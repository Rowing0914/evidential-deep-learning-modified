source .setup-evidential
cd neurips2020

# == test
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=evidence_reg_2_layers_50_neurons

python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=evidence_reg_2_layers_50_neurons
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=evidence_reg_4_layers_100_neurons
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=evidence_noreg_4_layers_100_neurons
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=ensemble_4_layers_100_neurons
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=gaussian_4_layers_100_neurons
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=dropout_4_layers_100_neurons
python run_cubic_tests.py --seed=1 --iterations=20 --num_samples=200 --run_name=bbbp_4_layers_100_neurons


# == UCI
nohup python run_uci_dataset_tests.py --datasets=boston >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=concrete >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=energy-efficiency >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=kin8nm >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=naval >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=power-plant >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=protein >&/dev/null &
nohup python run_uci_dataset_tests.py --datasets=yacht >&/dev/null &

# == toy exp
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=evidence_reg_2_layers_50_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=evidence_reg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=evidence_noreg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=ensemble_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=gaussian_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=dropout_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=1 --wandb --visualise --run_name=bbbp_4_layers_100_neurons >&/dev/null &

nohup python run_cubic_tests.py --seed=2 --wandb --run_name=evidence_reg_2_layers_50_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=2 --wandb --run_name=evidence_reg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=2 --wandb --run_name=evidence_noreg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=2 --wandb --run_name=ensemble_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=2 --wandb --run_name=gaussian_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=2 --wandb --run_name=dropout_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=2 --wandb --run_name=bbbp_4_layers_100_neurons >&/dev/null &

nohup python run_cubic_tests.py --seed=3 --wandb --run_name=evidence_reg_2_layers_50_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=3 --wandb --run_name=evidence_reg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=3 --wandb --run_name=evidence_noreg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=3 --wandb --run_name=ensemble_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=3 --wandb --run_name=gaussian_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=3 --wandb --run_name=dropout_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=3 --wandb --run_name=bbbp_4_layers_100_neurons >&/dev/null &

nohup python run_cubic_tests.py --seed=4 --wandb --run_name=evidence_reg_2_layers_50_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=4 --wandb --run_name=evidence_reg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=4 --wandb --run_name=evidence_noreg_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=4 --wandb --run_name=ensemble_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=4 --wandb --run_name=gaussian_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=4 --wandb --run_name=dropout_4_layers_100_neurons >&/dev/null &
nohup python run_cubic_tests.py --seed=4 --wandb --run_name=bbbp_4_layers_100_neurons >&/dev/null &
