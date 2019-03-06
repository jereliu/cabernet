"""Class definition for running experiments."""
import os
import functools

import numpy as np

import critique

import util.io as io_util
import util.data as data_util
import util.gp_flow as gp_util
import util.dtype as dtype_util
import util.experiment as experiment_util

import controller.bne as bne
import controller.config as config

import matplotlib.pyplot as plt

DEFAULT_KERN_FUNC_NAMES = ("rbf_0.25", "rbf_1", "period1.5")

DEFAULT_KERN_FUNC_DICT = {
    key: value for
    key, value in gp_util.DEFAULT_KERN_FUNC_DICT_GPY.items()
    if key in DEFAULT_KERN_FUNC_NAMES}

DEFAULT_N_INDUCE_POINTS = 20

DEFAULT_GP_LOG_LS_RESID = np.log(0.2).astype(dtype_util.NP_DTYPE)
DEFAULT_CDF_LOG_LS_RESID = np.log(0.3).astype(dtype_util.NP_DTYPE)

DEFAULT_PERCENTILES_TRAIN = np.linspace(.0001, .9999, num=15).astype(dtype_util.NP_DTYPE)
DEFAULT_PERCENTILES_PRED = np.linspace(.0001, .9999, num=100).astype(dtype_util.NP_DTYPE)


class Experiment(object):
    def __init__(self,
                 experiment_name,
                 N_train, N_test, N_calib, N_calib_induce, N_valid,
                 data_gen_func, save_addr):
        self.name = experiment_name

        self.N_train = N_train
        self.N_test = N_test
        self.N_calib = N_calib
        self.N_calib_induce = N_calib_induce
        self.N_valid = N_valid

        self.data_gen_func = data_gen_func

        self.save_addr = save_addr
        self.save_addr_data_plot = "{}/plot_data".format(self.save_addr)
        self.save_addr_base_models = (
            "{}/base/n_train_{}_n_test_{}_n_calib_{}/".format(
                self.save_addr, N_train,  N_test, N_calib))

    def run(self, **kwargs):
        """Executes full experiment."""
        self.prepare_data(**kwargs)

        self.plot_data()

        self.get_base_models()

        self.prepare_vi_data(**kwargs)

        self.run_bne_model(**kwargs)

        self.compute_metrics()

    def prepare_data(self,
                     seed_train=1000, seed_test=1000,
                     seed_calib=1000, **kwargs):
        # feature generation functions
        data_gen_func_x = data_util.gaussian_mix
        data_gen_func_x_test = functools.partial(data_util.gaussian_mix,
                                                 sd_scale=2.5)

        (self.X_train, self.y_train,
         self.X_test, self.y_test,
         self.X_valid, self.y_valid_sample,
         self.calib_sample_id,
         self.calib_sample_id_induce) = experiment_util.generate_data_1d(
            N_train=self.N_train, N_test=self.N_test,
            N_calib=self.N_calib, N_calib_induce=self.N_calib_induce,
            N_valid=self.N_valid,
            noise_sd=None,
            data_gen_func=self.data_gen_func,
            data_gen_func_x=data_gen_func_x,
            data_gen_func_x_test=data_gen_func_x_test,
            data_range=(-6., 6.), valid_range=(-6., 6.),
            seed_train=seed_train, seed_test=seed_test,
            seed_calib=seed_calib)

        self.y_valid = self.y_valid_sample[:, 0]

        # calibration data
        self.X_calib = self.X_valid[self.calib_sample_id]
        self.y_calib = self.y_valid[self.calib_sample_id]
        self.X_calib_induce = self.X_valid[self.calib_sample_id_induce]

        self.X_test = self.X_calib
        self.y_test = self.y_calib
        
    def plot_data(self):
        os.makedirs(self.save_addr_data_plot, exist_ok=True)

        plt.ioff()
        # plot data
        plt.figure(figsize=(12, 6))
        plt.scatter(np.repeat(self.X_valid, 100),
                    self.y_valid_sample[:, :100].flatten(), marker=".", s=1)
        plt.savefig("{}/data_valid_{}".format(self.save_addr_data_plot,
                                              self.name))
        plt.close()

        plt.scatter(self.X_train, self.y_train, marker="o", s=5.)
        plt.savefig("{}/data_train_{}".format(self.save_addr_data_plot,
                                              self.name))
        plt.close()

        plt.scatter(self.X_test, self.y_test, marker="o", s=5.)
        plt.savefig("{}/data_test_{}".format(self.save_addr_data_plot,
                                             self.name))
        plt.close()
        plt.ion()

    def get_base_models(self):
        load_res = None
        while load_res is None:
            try:
                load_res = io_util.load_results(["base_test_pred.pkl",
                                                 "base_valid_pred.pkl"],
                                                file_addr=self.save_addr_base_models)
            except FileNotFoundError:
                self.prepare_base_models()

        self.pred_dict_test = load_res["base_test_pred.pkl"]
        self.pred_dict_valid = load_res["base_valid_pred.pkl"]

        self.pred_dict_calib_induce = {
            key: value[self.calib_sample_id_induce]
            for key, value in self.pred_dict_valid.items()}

    def prepare_base_models(self):
        os.makedirs(self.save_addr_base_models, exist_ok=True)

        y_valid_mean = np.mean(self.y_valid_sample, axis=1)
        gp_util.fit_base_gp_models(self.X_train, self.y_train,
                                   self.X_test, self.y_test,
                                   self.X_valid, y_valid_mean,
                                   kern_func_dict=DEFAULT_KERN_FUNC_DICT,
                                   n_valid_sample=1000,
                                   save_addr_prefix=self.save_addr_base_models,
                                   y_range=[-2.5, 2.5])

    def prepare_vi_data(self, n_induce=DEFAULT_N_INDUCE_POINTS, **kwargs):
        induce_index = np.linspace(0, self.N_valid - 1, n_induce).astype(np.int)

        self.X_induce = self.X_valid[induce_index, ...]
        self.X_induce_mean = self.X_test

    def run_bne_model(self,
                      system_model_kwargs=None, random_model_kwargs=None,
                      **kwargs):
        """Configure and executes BNE model."""
        if system_model_kwargs is None or random_model_kwargs is None:
            self.system_model_kwargs, self.random_model_kwargs = (
                config.default_kwargs(self.X_induce, self.X_induce_mean))

        # initiate model
        self.bne_model = bne.BNE(X=self.X_test, y=self.y_test, base_pred=self.pred_dict_test,
                                 X_new=self.X_valid, base_pred_new=self.pred_dict_valid,
                                 X_calib=self.X_calib, y_calib=self.y_calib,
                                 X_calib_induce=self.X_calib_induce,
                                 base_pred_calib_induce=self.pred_dict_calib_induce,
                                 log_ls_system=DEFAULT_GP_LOG_LS_RESID,
                                 log_ls_random=DEFAULT_CDF_LOG_LS_RESID,
                                 calib_percentiles_train=DEFAULT_PERCENTILES_TRAIN,
                                 calib_percentiles_pred=DEFAULT_PERCENTILES_PRED)

        # run inference
        self.bne_model.run_model(self.system_model_kwargs,
                                 self.random_model_kwargs)

    def compute_metrics(self):
        self.eval_metrics = critique.EvalMetrics(bne_model=self.bne_model,
                                                 X_valid=self.X_valid,
                                                 y_valid_sample=self.y_valid_sample)
