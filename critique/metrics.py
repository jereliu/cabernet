"""Class definition for computing evaluation metrics."""
import collections

import meta.critique as critique_template

import numpy as np

METRIC_NAME_L1_DIST = "l1"
METRIC_NAME_RMSE = "rmse"
METRIC_NAME_COVERAGE_INDEX = "ci"

METRIC_NAMES = (METRIC_NAME_L1_DIST,
                METRIC_NAME_RMSE,)

BNEMetric = collections.namedtuple("BNEMetric",
                                   field_names=["system", "random"])


class EvalMetrics(critique_template.Critique):
    def __init__(self, bne_model, X_valid, y_valid_sample):
        """Initializer.

        Args:
            bne_model: (bne.BNE) Fitted result.
            X_valid: (np.ndarray) Validation features, shape (n_obs_valid, n_dim).
            y_valid_sample: (np.ndarray) A large validation sample from the
                data-generation mechanism, shape (n_obs_valid, n_sample_valid)
        """
        self.metric_names = METRIC_NAMES

        self.metric_funcs = {
            METRIC_NAME_L1_DIST: self.compute_l1_distance,
            METRIC_NAME_RMSE: self.compute_rmse, }

        self.metric_value = dict()

        # extract cdfs and medians
        super().__init__(bne_model, X_valid, y_valid_sample)

        # compute and extract metrics
        self.compute_all_metrics()

        self.l1 = self.metric_value[METRIC_NAME_L1_DIST]
        self.rmse = self.metric_value[METRIC_NAME_RMSE]

    def compute_all_metrics(self):
        for name in self.metric_names:
            self.metric_value[name] = self.metric_funcs[name]()

    def compute_l1_distance(self):
        """Computes L1 distance between model and data CDFs."""
        # shape (n_obs_valid, n_eval)
        model_cdf_diff = self.model_cdf.system - self.model_cdf.data
        random_cdf_diff = self.model_cdf.random - self.model_cdf.data

        return BNEMetric(
            system=_mean_absolute_value(model_cdf_diff),
            random=_mean_absolute_value(random_cdf_diff))

    def compute_rmse(self):
        """Computes RMSE for model medians."""
        model_median_diff = self.model_median.system - self.model_median.data
        random_median_diff = self.model_median.random - self.model_median.data

        return BNEMetric(
            system=_root_mean_square(model_median_diff),
            random=_root_mean_square(random_median_diff))

    def compute_coverage_index(self):
        ...


def _mean_absolute_value(x):
    return np.mean(np.abs(x))


def _root_mean_square(x):
    return np.sqrt(np.mean(x ** 2))
