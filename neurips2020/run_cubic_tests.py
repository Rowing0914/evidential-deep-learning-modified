import sys
sys.path.append("../")
import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from pathlib import Path
from sklearn.model_selection import train_test_split

import evidential_deep_learning as edl
import data_loader
import trainers
import models

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", default=False, action="store_true")
parser.add_argument("--visualise", default=False, action="store_true")
parser.add_argument("--save", default=False, action="store_true")
parser.add_argument("--prefix", default="cubic", type=str)
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--iterations", default=2000, type=int)
parser.add_argument("--num_samples", default=5000, type=int)
parser.add_argument("--num_features", default=2, type=int)
parser.add_argument("--data_name", default="cubic", type=str)
parser.add_argument("--run_name", default="asdf", type=str)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

save_fig_dir = f"./figs/{args.data_name}"


if args.data_name == "cubic":
    # ========= Original cubic dataset
    train_bounds = [[-4, 4]]
    X_train = np.concatenate([np.linspace(xmin, xmax, args.num_samples) for (xmin, xmax) in train_bounds]).reshape(-1,1)
    Y_train, _ = data_loader.generate_cubic(X_train, noise=True)

    test_bounds = [[-7, +7]]
    X_test = np.concatenate([np.linspace(xmin, xmax, args.num_samples) for (xmin, xmax) in test_bounds]).reshape(-1,1)
    Y_test, _ = data_loader.generate_cubic(X_test, noise=False)

elif args.data_name == "varying":
    # ========= Uncertain labels
    X = np.round(np.random.randn(args.num_samples), decimals=2)

    # Insert the duplicate entries
    threshold = 0.4
    mask = np.random.rand(args.num_samples) < threshold

    # duplication frequency score
    p = np.random.rand(sum(~mask))
    p = np.exp(p - np.max(p))
    p = p / p.sum(axis=0)

    X[mask] = np.random.choice(X[~mask], size=sum(mask), p=p)
    # mask = np.random.rand(args.num_samples) < threshold
    # X[mask] = np.random.choice(X[~mask], size=sum(mask))
    Y_1 = X.copy()

    # Perturb labels on duplicate x
    Y_1[mask] += (np.random.randn(sum(mask)) * 0.4)
    X = X[:, None]
    y = Y_1[:, None]
elif args.data_name == "multivariate":
    # ========= sklearn: https://scikit-learn.org/0.15/auto_examples/linear_model/plot_ransac.html#example-linear-model-plot-ransac-py
    import numpy as np
    from sklearn.datasets import make_regression

    n_outliers = 50
    X, y, coef = make_regression(n_samples=args.num_samples, n_features=args.num_features, n_informative=1, noise=10, coef=True, random_state=0)

    # Add outlier data
    X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

if args.data_name != "cubic":
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    ind = np.argsort(X_test[:, 0])  # sort for visualisation
    X_test = X_test[ind, :]
    Y_test = Y_test[ind, :]
print(X_train.shape, Y_train.shape)


### Plotting helper functions ###
def plot_scatter_with_var(mu, var, path, n_stds=3):
    if args.data_name == "varying":
        plt.plot(X_test, mu, color='#007cab')
        for k in np.linspace(0, n_stds, 4):
            plt.fill_between(X_test[:,0], (mu-k*var)[:,0], (mu+k*var)[:,0], alpha=0.3, edgecolor=None, facecolor='#00aeef', linewidth=0, antialiased=True)
        plt.scatter(X_test, Y_test, s=2., c='#463c3c')
    elif args.data_name == "cubic":
        plt.scatter(X_train, Y_train, s=1., c='#463c3c', zorder=0)
        for k in np.linspace(0, n_stds, 4):
            plt.fill_between(X_test[:,0], (mu-k*var)[:,0], (mu+k*var)[:,0], alpha=0.3, edgecolor=None, facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1)
        plt.plot(X_test, Y_test, 'r--', zorder=2)
        plt.plot(X_test, mu, color='#007cab', zorder=3)
        plt.gca().set_xlim(*test_bounds)
        plt.gca().set_ylim(-150,150)
    plt.title(path)
    plt.savefig(path)
    plt.clf()

def plot_ng(model, save="ng", ext=".png"):
    X_test_input = tf.convert_to_tensor(X_test, tf.float32)
    outputs = model(X_test_input)
    mu, v, alpha, beta = tf.split(outputs, 4, axis=1)

    epistemic = np.sqrt(beta/(v*(alpha-1)))
    epistemic = np.minimum(epistemic, 1e3) # clip the unc for vis
    plot_scatter_with_var(mu, epistemic, path=save+ext, n_stds=3)

def plot_ensemble(models, save="ensemble", ext=".png"):
    X_test_input = tf.convert_to_tensor(X_test, tf.float32)
    preds = tf.stack([model(X_test_input, training=False) for model in models], axis=0) #forward pass
    mus, sigmas = tf.split(preds, 2, axis=-1)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(sigmas, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_dropout(model, save="dropout", ext=".png"):
    X_test_input = tf.convert_to_tensor(X_test, tf.float32)
    preds = tf.stack([model(X_test_input, training=True) for _ in range(15)], axis=0) #forward pass
    mus, logvar = tf.split(preds, 2, axis=-1)
    var = tf.exp(logvar)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(var**0.5, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_bbbp(model, save="bbbp", ext=".png"):
    X_test_input = tf.convert_to_tensor(X_test, tf.float32)
    preds = tf.stack([model(X_test_input, training=True) for _ in range(15)], axis=0) #forward pass

    mean_mu = tf.reduce_mean(preds, axis=0)
    epistemic = tf.math.reduce_std(preds, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_gaussian(model, save="gaussian", ext=".png"):
    X_test_input = tf.convert_to_tensor(X_test, tf.float32)
    preds = model(X_test_input, training=False) #forward pass
    mu, sigma = tf.split(preds, 2, axis=-1)
    plot_scatter_with_var(mu, sigma, path=save+ext, n_stds=3)


#### Different toy configurations to train and plot
def evidence_reg_2_layers_50_neurons(return_dict=None):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=50, num_layers=2)
    args.group_name = f"{args.prefix}-evidence_reg_2_layers_50_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_2_layer_50_neurons"))
    if return_dict is not None:
        return_dict["evidence_reg_2_layers_50_neurons"] = rmse

def evidence_reg_2_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=2)
    args.group_name = f"{args.prefix}-evidence_reg_2_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_2_layers_100_neurons"))
    if return_dict is not None:
        return_dict["evidence_reg_2_layers_100_neurons"] = rmse

def evidence_reg_4_layers_50_neurons(return_dict=None):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=50, num_layers=4)
    args.group_name = f"{args.prefix}-evidence_reg_4_layers_50_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_4_layers_50_neurons"))
    if return_dict is not None:
        return_dict["evidence_reg_4_layers_50_neurons"] = rmse

def evidence_reg_4_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=4)
    args.group_name = f"{args.prefix}-evidence_reg_4_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_4_layers_100_neurons"))
    if return_dict is not None:
        return_dict["evidence_reg_4_layers_100_neurons"] = rmse

def evidence_noreg_4_layers_50_neurons(return_dict=None):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=50, num_layers=4)
    args.group_name = f"{args.prefix}-evidence_noreg_4_layers_50_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3, lam=0., maxi_rate=0.)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ng(model, os.path.join(save_fig_dir,"evidence_noreg_4_layers_50_neurons"))
    if return_dict is not None:
        return_dict["evidence_noreg_4_layers_50_neurons"] = rmse

def evidence_noreg_4_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=4)
    args.group_name = f"{args.prefix}-evidence_noreg_4_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3, lam=0., maxi_rate=0.)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ng(model, os.path.join(save_fig_dir,"evidence_noreg_4_layers_100_neurons"))
    if return_dict is not None:
        return_dict["evidence_noreg_4_layers_100_neurons"] = rmse

def ensemble_4_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.Ensemble
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=4)
    args.sigma = opts["sigma"]
    args.num_ensembles = opts["num_ensembles"]
    args.group_name = f"{args.prefix}-ensemble_4_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_ensemble(model, os.path.join(save_fig_dir,"ensemble_4_layers_100_neurons"))
    if return_dict is not None:
        return_dict["ensemble_4_layers_100_neurons"] = rmse

def gaussian_4_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.Gaussian
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=4)
    args.group_name = f"{args.prefix}-gaussian_4_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_gaussian(model, os.path.join(save_fig_dir,"gaussian_4_layers_100_neurons"))
    if return_dict is not None:
        return_dict["gaussian_4_layers_100_neurons"] = rmse

def dropout_4_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.Dropout
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=4, sigma=True)
    args.l = opts["l"]
    args.drop_prob = opts["drop_prob"]
    args.sigma = opts["sigma"]
    args.lam = opts["lam"]
    args.group_name = f"{args.prefix}-dropout_4_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=5e-3)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_dropout(model, os.path.join(save_fig_dir,"dropout_4_layers_100_neurons"))
    if return_dict is not None:
        return_dict["dropout_4_layers_100_neurons"] = rmse

def bbbp_4_layers_100_neurons(return_dict=None):
    trainer_obj = trainers.BBBP
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=X_train.shape[-1], num_neurons=100, num_layers=4)
    args.group_name = f"{args.prefix}-bbbp_4_layers_100_neurons"
    trainer = trainer_obj(model, args, learning_rate=1e-3)
    model, rmse, nll = trainer.train(X_train, Y_train, X_train, Y_train, np.array([[1.]]), iters=args.iterations, batch_size=args.batch_size)
    if args.visualise:
        plot_bbbp(model, os.path.join(save_fig_dir,"bbbp_4_layers_100_neurons"))
    if return_dict is not None:
        return_dict["bbbp_4_layers_100_neurons"] = rmse


### Main file to run the different methods and compare results ###
if __name__ == "__main__":
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)

    if args.run_name == "evidence_reg_2_layers_50_neurons":
        evidence_reg_2_layers_50_neurons()
    elif args.run_name == "evidence_reg_4_layers_100_neurons":
        evidence_reg_4_layers_100_neurons()
    elif args.run_name == "evidence_noreg_4_layers_100_neurons":
        evidence_noreg_4_layers_100_neurons()
    elif args.run_name == "ensemble_4_layers_100_neurons":
        ensemble_4_layers_100_neurons()
    elif args.run_name == "gaussian_4_layers_100_neurons":
        gaussian_4_layers_100_neurons()
    elif args.run_name == "dropout_4_layers_100_neurons":
        dropout_4_layers_100_neurons()
    elif args.run_name == "bbbp_4_layers_100_neurons":
        bbbp_4_layers_100_neurons()

    # print(f"Done! Figures saved to {save_fig_dir}")
    # from multiprocessing import Process, Manager
    # return_dict = Manager().dict()

    # jobs = list()
    # for f in [
    #     evidence_reg_2_layers_50_neurons,
    #     evidence_reg_4_layers_100_neurons,
    #     evidence_noreg_4_layers_100_neurons,
    #     ensemble_4_layers_100_neurons,
    #     gaussian_4_layers_100_neurons,
    #     dropout_4_layers_100_neurons,
    #     bbbp_4_layers_100_neurons,
    # ]:
    #     p = Process(target=f, args=(return_dict,))
    #     jobs.append(p)
    #     p.start()
    
    # for p in jobs:
    #     p.join()
    # print(return_dict)
