import sys
sys.path.append("../")

import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

import evidential_deep_learning as edl
import data_loader
import trainers
import models
from models.toy.h_params import h_params

import os
if not os.path.exists("uci-results/"):
    os.makedirs("uci-results/")

parser = argparse.ArgumentParser()
parser.add_argument("--num_trials", default=20, type=int, help="Number of trials to repreat training")
parser.add_argument("--num_epochs", default=40, type=int)
parser.add_argument('--datasets', nargs='+', default=["boston", "concrete"], choices=['boston', 'concrete', 'energy-efficiency', 'kin8nm', 'naval', 'power-plant', 'protein', 'wine', 'yacht'])
args = parser.parse_args()

"""" ================================================"""
args.wandb = args.save = False
training_schemes = [trainers.Dropout, trainers.Ensemble, trainers.Evidential]
datasets = args.datasets
num_trials = args.num_trials
num_epochs = args.num_epochs
dev = "/cpu:0" # for small datasets/models cpu is faster than gpu
"""" ================================================"""

RMSE = np.zeros((len(datasets), len(training_schemes), num_trials))
NLL = np.zeros((len(datasets), len(training_schemes), num_trials))
for di, dataset in enumerate(datasets):
    for ti, trainer_obj in enumerate(training_schemes):
        for n in range(num_trials):
            print("================================================")
            print(f"{dataset} {trainer_obj.__name__} {n}")
            print("================================================")
            args.seed = n
            random.seed(args.seed)
            np.random.seed(args.seed)
            tf.random.set_seed(args.seed)
            (x_train, y_train), (x_test, y_test), y_scale = data_loader.load_dataset(dataset, return_as_tensor=False)
            batch_size = h_params[dataset]["batch_size"]
            num_iterations = num_epochs * x_train.shape[0]//batch_size
            done = False
            while not done:
                with tf.device(dev):
                    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
                    model, opts = model_generator.create(input_shape=x_train.shape[1:])
                    if trainer_obj.__name__.lower() == "ensemble":
                        args.sigma = opts["sigma"]
                        args.num_ensembles = opts["num_ensembles"]
                    elif trainer_obj.__name__.lower() == "dropout":
                        args.l = opts["l"]
                        args.drop_prob = opts["drop_prob"]
                        args.sigma = opts["sigma"]
                        args.lam = opts["lam"]
                    trainer = trainer_obj(model, args, dataset, learning_rate=h_params[dataset]["learning_rate"])
                    model, rmse, nll = trainer.train(x_train, y_train, x_test, y_test, y_scale, iters=num_iterations, batch_size=batch_size)
                    del model
                    tf.keras.backend.clear_session()
                    done = False if np.isinf(nll) or np.isnan(nll) else True
            
            print("saving {} {}".format(rmse, nll))
            RMSE[di, ti, n] = rmse
            NLL[di, ti, n] = nll

mu = RMSE.mean(axis=-1)
error = RMSE.std(axis=-1)

print("TRAINERS: {}\nDATASETS: {}".format([trainer.__name__ for trainer in training_schemes], datasets))
print("MEAN: \n{}".format(mu))
print("ERROR: \n{}".format(error))

with open(f"uci-results/result-{dataset}.txt", "w") as file:
    file.write("TRAINERS: {}\nDATASETS: {}".format([trainer.__name__ for trainer in training_schemes], datasets))
    file.write("\nMEAN: {}".format(mu))
    file.write("\nERROR: {}".format(error))