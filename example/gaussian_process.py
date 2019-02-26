"""Playground scripts for system testing with Gaussian Process."""
import os
import sys

# sys.path.extend([os.getcwd()])

from importlib import reload
from functools import partial

import numpy as np

import model.gp as model
import inference.vi as vi
import inference.predictor as predictor

import util.data as data_util
import util.visual as visual_util
import util.experiment as experiment_util

import matplotlib.pyplot as plt

_MULTISCALE_DATA = True

_DEFAULT_LOG_LS_SCALE = np.log(0.075).astype(np.float32)

_SAVE_ADDR_PREFIX = "./experiment_result/gpr/"

os.makedirs(_SAVE_ADDR_PREFIX, exist_ok=True)

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""
if not _MULTISCALE_DATA:
    N = 50
    X_train, y_train = data_util.generate_1d_data(N=N,
                                                  f=data_util.sin_curve_1d,
                                                  noise_sd=0.03, seed=100)
    X_train = np.expand_dims(X_train, 1).astype(np.float32)
    y_train = y_train.astype(np.float32)
    std_y_train = np.std(y_train)

    X_test = np.expand_dims(np.linspace(-1, 2, 100), 1).astype(np.float32)
    y_test = data_util.sin_curve_1d(X_test)

    X_valid = X_test
    y_valid = y_test
    calib_sample_id = None

    N, D = X_train.shape
else:
    N_train = 50
    N_test = 50
    N_valid = 500

    data_gen_func_list = [
        partial(data_util.sin_curve_1d, freq=(3, 6), x_rate=0.1),
        partial(data_util.sin_curve_1d_fast_local, bound=[0.1, 0.6],
                freq=40., scale=0.5)
    ]

    (X_train, y_train,
     X_test, y_test,
     X_valid, y_valid, calib_sample_id) = experiment_util.generate_data_1d_multiscale(
        N_train=N_train, N_test=N_test, N_valid=N_valid, noise_sd=0.005,
        data_gen_func_list=data_gen_func_list,
        data_range=(0., 1.), valid_range=(-0.5, 1.5),
        seed_train=1500, seed_test=2500, seed_calib=100)

#
plt.plot(X_valid, y_valid, c='black')
plt.plot(X_train.squeeze(), y_train.squeeze(),
         'o', c='red', markeredgecolor='black')
plt.xlim([-0.5, 1.5])
plt.ylim([-4.5, 4.5])
plt.savefig("{}/data.png".format(_SAVE_ADDR_PREFIX))
plt.close()

"""""""""""""""""""""""""""""""""
# 2. Set up model
"""""""""""""""""""""""""""""""""
# define model
gp_model = model.GaussianProcess(X=X_test,
                                 y=y_test,
                                 log_ls=_DEFAULT_LOG_LS_SCALE)

# set up estimator, run inference
gp_estimator = vi.VIEstimator(model=gp_model)
gp_estimator.config(step_size=5e-3, Z=X_test)

model_session = gp_estimator.run(max_steps=int(3e4),
                                 verbose=True)

# set up predictor
gp_predictor = predictor.Predictor(estimator=gp_estimator)

gp_predictor.config(sample_type="post_sample", n_sample=1000)
sample_dict = gp_predictor.run(sess=model_session)

gp_predictor.config(sample_type="pred_sample",
                    X_new=X_valid,
                    f_sample=sample_dict["post_sample"]["gp"].T)
sample_dict = gp_predictor.run(sess=model_session)

# visualize
y_pred = sample_dict["pred_sample"]["gp"]

mu = np.mean(y_pred, axis=1)
cov = np.var(y_pred, axis=1)

visual_util.gpr_1d_visual(mu, cov,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          title="RBF, ls={:.3f}".format(np.exp(_DEFAULT_LOG_LS_SCALE)),
                          save_addr="{}/gpr_fit.png".format(_SAVE_ADDR_PREFIX),
                          smooth_quantile=False)
