"""Estimator for computing distribution moments using CDF samples."""
import numpy as np

import meta.estimator as estimator_template

import inference.vi as vi

import tensorflow as tf

_MEAN_NAME = "mean"
_MEDIAN_NAME = "median"

_VAR_NAME = "var"
_SKEW_NAME = "skewness"
_KURT_NAME = "kurtosis"

_QUANT_NAME = "quantiles"
_QUANT_SINGLE_NAME = "quantile_single"

_MEAN_CDF_NAME = "mean_cdf"


class CDFMoments(estimator_template.Estimator):
    """Class instance to estimate distribution moments using CDF samples."""

    def __init__(self, estimator, cdf_sample_dict):
        """

        Args:
            estimator: (Estimator) The estimator used to generate cdf sample.
            cdf_sample: (dict) Dictionary of CDF samples, see model.MonoGP.
        """
        # define model and initialize other attributes
        super().__init__(model=estimator.model)

        # register estimator
        if not isinstance(estimator, vi.VIEstimator):
            raise ValueError("`estimator` must be a inference estimator "
                             "(i.e. VIEstimator)")
        self.estimator = estimator
        self.graph = estimator.graph

        self.y_eval = cdf_sample_dict["y_eval"]  # shape (n_eval, n_obs)
        self.cdf_val = cdf_sample_dict["cdf"]  # shape (n_sample, n_eval, n_obs)
        self.mean_cdf_val = self._compute_mean_cdf()  # shape (n_eval, n_obs)
        self.cdf_val_orig = cdf_sample_dict["cdf_orig"]  # shape (n_eval, n_obs)

        self.n_sample, self.n_eval, self.n_obs = self.cdf_val.shape
        # register moment functions and containers
        self.moment_funcs = {_MEAN_NAME: self._config_post_mean,
                             _MEDIAN_NAME: self._config_post_median,
                             _VAR_NAME: self._config_post_var,
                             _SKEW_NAME: self._config_post_skew,
                             _KURT_NAME: self._config_post_kurt,
                             _QUANT_NAME: self._config_mean_quantiles,
                             _QUANT_SINGLE_NAME: self._config_mean_single_quantile,
                             _MEAN_CDF_NAME: self._config_mean_cdf
                             }

        self.moment_dict = dict()

    def config(self, moment_type, **sample_kwargs):
        """Interface function for setting up estimator graph.

        Adds a new sample dictionary to self.sample_dict

        Args:
            moment_type: (str) Type of predictive samples to draw, must be one
                of the self.pred_types.
            **sample_kwargs: Keyword arguments to pass to corresponding
                sample functions in self.sample_funcs.

        Raises:
            (ValueError) If sample_type does not belong to
                self.sample_funcs.keys
        """
        if moment_type not in self.moment_funcs.keys():
            raise ValueError(
                "moment_type `{}` not supported.\n"
                "Can only be one of {}".format(
                    moment_type, tuple(self.moment_funcs.keys())))

        # adds sampling node to computation graph
        with self.graph.as_default():
            with tf.name_scope("sample_{}".format(moment_type)):
                param_samples = (
                    self.moment_funcs[moment_type](**sample_kwargs))

        self.moment_dict[moment_type] = param_samples

    def run(self):
        """Executes estimator graph in a given session.

        Returns:
            (dict) Evaluated self.sample_dict.
        """
        return self.moment_dict

    def _config_post_mean(self, **kwargs):
        """Computes posterior mean using CDF samples."""
        moment1_deriv = 1.
        return self._compute_statistic_mean(moment1_deriv)

    def _config_post_median(self, **kwargs):
        """Computes posterior mean using CDF samples."""
        return self._config_mean_single_quantile(percentile=.5)

    def _config_post_var(self, **kwargs):
        """Computes posterior variance using CDF samples."""
        # first get posterior mean, shape (1, n_obs)
        moment1_mean = np.mean(self.moment_dict[_MEAN_NAME],
                               axis=0, keepdims=True)

        moment2_deriv = 2 * self.y_eval
        moment2_mean = self._compute_statistic_mean(moment2_deriv)

        return moment2_mean - moment1_mean ** 2

    def _config_post_skew(self, **kwargs):
        """Computes posterior skewness using CDF samples."""
        ...

    def _config_post_kurt(self, **kwargs):
        """Computes posterior kurtosis using CDF samples."""
        ...

    def _config_mean_cdf(self, **kwargs):
        """Computes posterior mean of CDF samples."""
        return self.mean_cdf_val

    def _config_mean_quantiles(self, percentiles=[.1, .25, .5, .75, .9], **kwargs):
        """Computes posterior kurtosis using CDF samples."""
        quantile_val_list = [self._config_mean_single_quantile(val)
                             for val in percentiles]

        return np.array(quantile_val_list)

    def _config_mean_single_quantile(self, percentile=.5, **kwargs):
        """Computes posterior kurtosis using CDF samples."""
        # nearest neighbor approach
        y_val_index = np.argmin(np.abs(self.mean_cdf_val - percentile), axis=0)

        quantile_vals = np.array([self.y_eval[y_index, obs_id] for
                                  obs_id, y_index in enumerate(y_val_index)])
        return quantile_vals

    def _compute_statistic_mean(self, statistic):
        """Computes posterior mean of given statistic using Darth Vadar rule.

        Args:
            statistic: (tf.Tensor) statistics to compute mean over, shape (n_eval, n_obs)

        Returns:
            (tf.Tensor) mean of statistics, shape (n_sample, n_obs)
        """

        return np.mean(statistic * ((self.y_eval > 0) - self.cdf_val),
                       axis=1)

    def _compute_mean_cdf(self):
        """Computes sample mean of cdf_val array and post process.

            Takes in cdf array of dimension (n_sample, n_eval, n_obs)

        Returns:
            (np.ndarray) mean cdf array of shape (n_eval, n_obs)

        """
        mean_cdf_val = np.mean(self.cdf_val, axis=0)

        mean_cdf_val[mean_cdf_val > 1.] = 1.
        mean_cdf_val[mean_cdf_val < 0.] = 0.

        return np.maximum.accumulate(mean_cdf_val, axis=0)
