"""Parametric Ensemble with flat model structure using the BNE class. """
import os
import sys
from importlib import reload

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.extend([os.getcwd()])

import pandas as pd
import numpy as np

import util.data as data
import controller

SAVE_ADDR = "./experiment_result/bne"
DATA_FUNC_DICT = {"hetero": data.sin_cos_curve_skew_noise_1d}

data_types = ("hetero",)
data_sizes = np.concatenate([np.arange(50, 200, step=25),
                             np.arange(200, 500, step=50),
                             np.arange(500, 1001, step=100), ])
num_reps_trials = 20

trial_id = 0
data_size = 100
data_type = "hetero"


def run_experiment(trial_id, data_size, data_type):
    # default data parameters
    N_train = 50
    N_calib_induce = 50
    N_valid = 1000

    # experiment settings
    N_test = N_calib = data_size
    data_gen_func = DATA_FUNC_DICT[data_type]

    #
    experiment_name = ("{}_train_{}_calib_{}_{}".format(data_type, N_test,
                                                        N_calib, trial_id))

    experiment = controller.Experiment(experiment_name=experiment_name,
                                       N_train=N_train,
                                       N_test=N_test,
                                       N_calib=N_calib,
                                       N_calib_induce=N_calib_induce,
                                       N_valid=N_valid,
                                       data_gen_func=data_gen_func,
                                       save_addr=SAVE_ADDR)
    experiment.run(seed_test=None)

    return experiment.eval_metrics.l1, experiment.eval_metrics.rmse


if __name__ == "__main__":
    record = pd.DataFrame(columns=["data_type", "sample_size",
                                   "system_l1", "random_l1",
                                   "system_rmse", "random_rmse"])

    for data_type in data_types:
        for sample_size in data_sizes:
            for trial_id in range(num_reps_trials):
                l1, rmse = run_experiment(trial_id, sample_size, data_type)

                record = record.append({"data_type": data_type,
                                        "sample_size": sample_size,
                                        "system_l1": l1.system,
                                        "random_l1": l1.random,
                                        "system_rmse": rmse.system,
                                        "random_rmse": rmse.random
                                        }, ignore_index=True)
                record.to_csv("{}/record.csv".format(SAVE_ADDR))
                record.to_csv("/home/jeremiah/Dropbox/record.csv".format(SAVE_ADDR))

