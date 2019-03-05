"""Class definition for agent to manage data and run models."""
import collections

import numpy as np

import model

import inference.vi as vi
import inference.predictor as predictor
import inference.cdf as cdf

import util.dtype as dtype_util

DEFAULT_GP_LOG_LS_RESID = np.log(0.2).astype(np.float32)
DEFAULT_CDF_LOG_LS_RESID = np.log(0.3).astype(np.float32)

DEFAULT_N_POST_SAMPLE = 500

DEFAULT_CALIB_PERCENTILES_TRAIN = np.linspace(.0001, .9999, num=15).astype(dtype_util.NP_DTYPE)
DEFAULT_CALIB_PERCENTILES_PRED = np.linspace(.0001, .9999, num=100).astype(dtype_util.NP_DTYPE)
DEFAULT_OUTPUT_PERCENTILES = np.array([10, 25, 50, 75, 90]).astype(dtype_util.NP_DTYPE)

BNE_SUMMARY_NAMES = ("mean", "median", "var", "quantiles", "mean_cdf")

BNESummary = collections.namedtuple("BNESummary",
                                    field_names=BNE_SUMMARY_NAMES)
BNESummary.__new__.__defaults__ = (None,) * len(BNESummary._fields)


class BNE(object):
    def __init__(self,
                 X, y, base_pred,
                 X_new, base_pred_new,
                 X_calib=None, y_calib=None,
                 X_calib_induce=None, base_pred_calib_induce=None,
                 log_ls_system=DEFAULT_GP_LOG_LS_RESID,
                 log_ls_random=DEFAULT_CDF_LOG_LS_RESID,
                 calib_percentiles_train=DEFAULT_CALIB_PERCENTILES_TRAIN,
                 calib_percentiles_pred=DEFAULT_CALIB_PERCENTILES_PRED,
                 pred_percentiles=DEFAULT_OUTPUT_PERCENTILES):
        """Initializer."""
        # names of summary statistics
        self.summary_names = BNE_SUMMARY_NAMES
        self.posterior_summary = BNESummary()

        # initialize data
        self.X_train = X
        self.y_train = y
        self.base_pred_train = base_pred

        self.X_calib = self.X_train if X_calib is None else X_calib
        self.y_calib = self.y_train if y_calib is None else y_calib
        self.X_calib_induce = self.X_calib if X_calib_induce is None else X_calib_induce
        self.base_pred_calib_induce = (self.base_pred_train if base_pred_calib_induce is None
                                       else base_pred_calib_induce)

        self.X_test = X_new
        self.base_pred_test = base_pred_new

        # initialize hyper parameters
        self.log_ls_system = log_ls_system
        self.log_ls_random = log_ls_random

        self.calib_percentiles_train = calib_percentiles_train
        self.calib_percentiles_pred = calib_percentiles_pred
        self.pred_percentiles = pred_percentiles

        # initialize internal parameters
        # model
        self.system_model = None
        self.random_model = None

        # initialize model estimator, predictor and sessions
        self.system_estimator = None
        self.random_estimator = None

        self.system_predictor = None
        self.random_predictor = None
        self.random_summarizer = None

        self.system_session = None
        self.random_session = None

        # initialize containers for posterior samples at training/predict locations
        self.system_model_sample_train = None
        self.random_model_sample_train = None

        self.system_model_sample_calib = None  # for training random component model
        self.system_model_sample_pred = None
        self.random_model_sample_pred = None

        self.system_model_quantile_calib = None  # for training random component model
        self.system_model_quantile_pred = None
        self.random_model_quantile_pred = None

    def run_model(self):
        """Estimates and generates predictive samples/quantiles for full model."""
        ...

    def run_system_model(self,
                         log_ls_system=None, restart_model=True,
                         n_sample=DEFAULT_N_POST_SAMPLE,
                         **estimate_kwargs):
        """Estimates and generates predictive samples/quantiles for sys model."""
        # define model
        if log_ls_system:
            self.log_ls_system = log_ls_system

        if not self.system_model or restart_model:
            self.system_model = (
                model.HierarchicalGP(X=self.X_train, y=self.y_train,
                                     base_pred=self.base_pred_train,
                                     resid_log_ls=self.log_ls_system)
            )

        # ESTIMATION
        self._estimate_system_model(**estimate_kwargs)

        # PREDICTION
        self.system_predictor = predictor.Predictor(estimator=self.system_estimator)

        # sample posterior, train, calibration and pred location
        self.system_model_sample_train = self._sample_system_model(n_sample)

        self.system_model_sample_calib = (
            self._pred_sample_system_model(X_new=self.X_calib_induce,
                                           base_pred_new=self.base_pred_calib_induce))
        self.system_model_sample_pred = (
            self._pred_sample_system_model(X_new=self.X_test,
                                           base_pred_new=self.base_pred_test))

        # sample quantiles, calibration location (for training random component model)
        self.system_model_quantile_calib = (
            self._pred_quantiles_system_model(sample_dict=self.system_model_sample_calib,
                                              perc_eval=self.calib_percentiles_train))

        # sample quantiles, predictive location (for prediction output)
        self.system_model_quantile_pred = (
            self._pred_quantiles_system_model(sample_dict=self.system_model_sample_pred,
                                              perc_eval=self.calib_percentiles_pred))

    def run_random_model(self,
                         log_ls_random=None, restart_model=True,
                         n_sample=DEFAULT_N_POST_SAMPLE,
                         **estimate_kwargs):
        """Estimates and generates predictive samples/quantiles for rand model."""
        # define model
        if log_ls_random:
            self.log_ls_random = log_ls_random

        if not self.random_model or restart_model:
            self.random_model = (
                model.MonoGP(X=self.X_calib, y=self.y_calib,
                             X_induce=self.X_calib_induce,
                             cdf_sample_induce=self.system_model_quantile_calib,
                             log_ls=self.log_ls_random)
            )

        # ESTIMATION
        self._estimate_random_model(**estimate_kwargs)

        # PREDICTION
        self.random_predictor = predictor.Predictor(estimator=self.random_estimator)

        # sample posterior, train and pred locations
        self.random_model_sample_train = self._sample_random_model(n_sample)
        self.random_model_sample_pred = (
            self._pred_sample_random_model(X_new=self.X_test,
                                           quant_pred_new=self.system_model_quantile_pred))

        # SUMMARIZE
        self.random_summarizer = (
            cdf.CDFMoments(estimator=self.random_estimator,
                           cdf_sample_dict=self.random_model_sample_pred))
        self.posterior_summary = BNESummary(**self._pred_summary_random_model())

    def _estimate_system_model(self,
                               step_size=1e-2,
                               max_steps=50000, model_dir=None,
                               save_step=500, verbose=True,
                               restart_estimate=False,
                               **vi_kwargs):
        """Estimates systematic component model."""
        # estimation
        if not self.system_estimator:
            # in case of first run, configure estimator graph
            self.system_estimator = vi.VIEstimator(model=self.system_model)
            self.system_estimator.config(step_size=step_size, **vi_kwargs)

        elif restart_estimate:
            # in case of re-run and want to restart, re-config
            # estimator graph and erase current session.
            self.system_estimator.config(step_size=step_size, **vi_kwargs)
            self.system_session = None
        else:
            # in case of re-run and want to reuse current setting, do nothing
            pass

        self.system_session = self.system_estimator.run(sess=self.system_session,
                                                        max_steps=max_steps,
                                                        model_dir=model_dir,
                                                        save_step=save_step,
                                                        verbose=verbose)

    def _estimate_random_model(self,
                               step_size=1e-2,
                               max_steps=10000, model_dir=None,
                               save_step=500, verbose=True,
                               restart_estimate=False,
                               **vi_kwargs):
        """Estimates random component model."""
        # estimation
        if not self.random_estimator:
            # in case of first run, configure estimator graph
            self.random_estimator = vi.VIEstimator(model=self.random_model)
            self.random_estimator.config(step_size=step_size, **vi_kwargs)
        elif restart_estimate:
            # in case of re-run and want to restart, re-config
            # estimator graph and erase current session.
            self.random_estimator.config(step_size=step_size, **vi_kwargs)
            self.random_session = None
        else:
            # in case of re-run and want to reuse current setting, do nothing
            pass

        self.random_session = self.random_estimator.run(sess=self.random_session,
                                                        max_steps=max_steps,
                                                        model_dir=model_dir,
                                                        save_step=save_step,
                                                        verbose=verbose)

    def _sample_system_model(self, n_sample):
        """Generates posterior samples for model parameters at training locations."""
        if not self.system_predictor:
            raise ValueError("Predictor for systematic component model empty.")
        if not self.system_session:
            raise ValueError("Session for systematic component model empty.")

        # posterior sampling for training sample
        # in-sample posterior sample, training locations
        self.system_predictor.config(sample_type="post_sample",
                                     rv_dict=self.system_estimator.model.model_param,
                                     n_sample=n_sample)
        sample_dict = self.system_predictor.run(sess=self.system_session)
        return sample_dict["post_sample"]

    def _sample_random_model(self, n_sample):
        """Generates posterior samples for model parameters at training locations."""
        if not self.random_predictor:
            raise ValueError("Predictor for random component model is empty.")
        if not self.random_session:
            raise ValueError("Session for systematic component model is empty.")

        # posterior sampling for training sample
        # in-sample posterior sample, training locations
        self.random_predictor.config(sample_type="post_sample",
                                     rv_dict=self.random_estimator.model.model_param,
                                     n_sample=n_sample)
        sample_dict = self.random_predictor.run(sess=self.random_session)
        return sample_dict["post_sample"]

    def _pred_sample_system_model(self, X_new, base_pred_new):
        """Generates posterior samples at predictive locations."""
        if not self.system_model_sample_train:
            raise ValueError("Train sample for systematic component model empty.")
        if not self.system_session:
            raise ValueError("Session for systematic component model empty.")

        self.system_predictor.config(sample_type="pred_sample",
                                     X_new=X_new,
                                     base_pred_new=base_pred_new,
                                     post_sample_dict=self.system_model_sample_train)

        sample_dict = self.system_predictor.run(sess=self.system_session)

        return sample_dict["pred_sample"]

    def _pred_sample_random_model(self, X_new, quant_pred_new):
        """Generates posterior samples at predictive locations."""
        if not self.system_model_sample_train:
            raise ValueError("Train sample for systematic component model empty.")
        if not self.system_session:
            raise ValueError("Session for systematic component model empty.")

        self.random_predictor.config(sample_type="pred_sample",
                                     X_new=X_new,
                                     quant_dict_new=quant_pred_new,
                                     post_sample_dict=self.random_model_sample_train,
                                     verbose=True)

        sample_dict = self.random_predictor.run(sess=self.random_session)

        return sample_dict["pred_sample"]

    def _pred_quantiles_system_model(self, sample_dict, perc_eval):
        """Generates predictive quantiles at predictive locations."""
        if not self.system_model_sample_pred:
            raise ValueError("Predictive sample for systematic component model empty.")
        if not self.system_session:
            raise ValueError("Session for systematic component model empty.")

        self.system_predictor.config(sample_type="pred_quant",
                                     perc_eval=perc_eval,
                                     sample_dict=sample_dict)

        sample_dict = self.system_predictor.run(sess=self.system_session)
        return sample_dict["pred_quant"]

    def _pred_summary_random_model(self):
        """Computes summary statistics from the estimated CDFs."""
        for summary_name in self.summary_names:
            self.random_summarizer.config(summary_name,
                                          percentiles=self.pred_percentiles / 100.)

        return self.random_summarizer.run()

    @property
    def log_ls_system(self):
        return self.__log_ls_system

    @log_ls_system.setter
    def log_ls_system(self, value):
        self.__log_ls_system = value.astype(dtype_util.NP_DTYPE)

    @property
    def log_ls_random(self):
        return self.__log_ls_random

    @log_ls_random.setter
    def log_ls_random(self, value):
        self.__log_ls_random = value.astype(dtype_util.NP_DTYPE)
