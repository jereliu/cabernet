"""Class definition to compute evaluation metrics.

    Computes L1 distance, Median RMSE, Coverage Index
    for both the system-component model and the full model against the
    ground truth.

    Optionally, plots ground-truth / predicted CDFs, and also coverage plot.
"""
import collections

import numpy as np

BNE_PARAM_FIELD_NAMES = ("x", "y_eval", "system", "random", "data")

BNEParam = collections.namedtuple("BNEParam",
                                  field_names=BNE_PARAM_FIELD_NAMES)
BNEParam.__new__.__defaults__ = (None,) * len(BNEParam._fields)


class Critique(object):
    def __init__(self, bne_model, X_valid, y_valid_sample):
        """Initializer.

        Args:
            bne_model: (bne.BNE) Fitted result.
            X_valid: (np.ndarray) Validation features, shape (n_obs_valid, n_dim).
            y_valid_sample: (np.ndarray) A large validation sample from the
                data-generation mechanism, shape (n_obs_valid, n_sample_valid)
        """
        self.bne_model = bne_model
        self.valid_feature = X_valid
        self.valid_sample = y_valid_sample

        self.model_cdf = self._extract_all_cdfs()
        self.model_median = self._extract_all_medians()

    def _extract_all_cdfs(self):
        """Extracts model CDFs for sys and full model, and also true CDF"""
        y_eval = self.bne_model.random_summarizer.y_eval

        # extract cdfs, shape (n_obs_valid, n_eval)
        system_cdf = self.bne_model.random_summarizer.cdf_val_orig
        random_cdf = self.bne_model.random_summarizer.mean_cdf_val
        data_cdf = self._compute_empirical_cdf()

        return BNEParam(x=self.valid_feature, y_eval=y_eval,
                        system=system_cdf, random=random_cdf,
                        data=data_cdf)

    def _extract_all_medians(self):
        """Extracts model CDFs for sys and full model, and also true CDF"""
        # extract medians, shape (n_obs_valid, )

        system_median = np.median(self.bne_model.system_model_sample_pred["y"],
                                  axis=0)
        random_median = self.bne_model.posterior_summary.median
        data_median = np.median(self.valid_sample, axis=-1)

        return BNEParam(x=self.valid_feature,
                        system=system_median,
                        random=random_median,
                        data=data_median)

    def _compute_empirical_cdf(self):
        """Computes eCDF of valid_sample using BNE's eval points. """
        # y evaluation points for each obs in valid_sample,
        # shape (n_eval, n_obs_valid, 1)
        y_eval = self.bne_model.random_summarizer.y_eval[:, :, np.newaxis]

        # y sample values for each obs in valid_sample,
        # shape (1, n_obs_valid, n_sample_valid)
        y_sample = self.valid_sample[np.newaxis, :, :]

        # final empirical CDF by averaging over indicators,
        # shape (n_eval, n_obs_valid)
        return np.mean(y_sample < y_eval, axis=-1)