"""Estimator for computing predictive samples / CDFs."""
import meta.estimator as estimator_template
import inference.vi as vi

import tensorflow as tf


class Predictor(estimator_template.Estimator):
    """Draws predictive sample/cdfs by adding sampling node to Graph."""

    def __init__(self, estimator):
        """Initializer.

        Args:
            estimator: (Estimator) A estimator configured for estimating
                parameters for a model.
        """
        # define model and initialize other attributes
        super().__init__(model=estimator.model)

        # register estimator
        if not isinstance(estimator, vi.VIEstimator):
            raise ValueError("`estimator` must be a inference estimator "
                             "(i.e. VIEstimator)")
        self.estimator = estimator
        self.graph = estimator.graph

        # register sample functions and containers
        self.sample_funcs = {"post_sample": self._config_posterior_sample_graph,
                             "pred_sample": self._config_predictive_sample_graph,
                             "pred_cdf": self._config_predictive_cdf_graph,
                             "pred_quant": self._config_predictive_quant_graph,}

        self.sample_dict = dict()

    def config(self, sample_type, **sample_kwargs):
        """Interface function for setting up estimator graph.

        Adds a new sample dictionary to self.sample_dict

        Args:
            sample_type: (str) Type of predictive samples to draw, must be one
                of the self.pred_types.
            **sample_kwargs: Keyword arguments to pass to corresponding
                sample functions in self.sample_funcs.

        Raises:
            (ValueError) If sample_type does not belong to
                self.sample_funcs.keys
        """
        if sample_type not in self.sample_funcs.keys():
            raise ValueError(
                "sample_type `{}` not supported.\n"
                "Can only be one of {}".format(
                    sample_type, tuple(self.sample_funcs.keys())))

        # adds sampling node to computation graph
        with self.graph.as_default():
            with tf.name_scope("sample_{}".format(sample_type)):
                param_samples = (
                    self.sample_funcs[sample_type](**sample_kwargs))

        self.sample_dict[sample_type] = param_samples

    def run(self, sess):
        """Executes estimator graph in a given session.

        Args:
            sess: (tf.Session) A session that was previously used
                to perform inference with self.estimator. It should
                contain all the parameter estimates.

        Returns:
            (dict) Evaluated self.sample_dict.
        """
        return sess.run(self.sample_dict)

    def _config_posterior_sample_graph(self, **kwargs):
        """Adds graph nodes for drawing predictive samples."""
        return self.model.posterior_sample(**kwargs)

    def _config_predictive_sample_graph(self, **kwargs):
        """Adds graph nodes for drawing predictive samples."""
        return self.model.predictive_sample(**kwargs)

    def _config_predictive_cdf_graph(self, **kwargs):
        """Adds graph nodes for computing predictive CDFs.

        Raises:
            (ValueError): If model does not contain predictive_cdf method.
        """
        if not hasattr(self.model, "predictive_cdf"):
            raise ValueError("Predictive CDF not implemented "
                             "for current model ({}).".format(self.model.model_name))

        return self.model.predictive_cdf(**kwargs)

    def _config_predictive_quant_graph(self, **kwargs):
        """Adds graph nodes for computing predictive CDFs.

        Raises:
            (ValueError): If model does not contain predictive_cdf method.
        """
        if not hasattr(self.model, "predictive_quantile"):
            raise ValueError("Predictive Quantile not implemented "
                             "for current model ({}).".format(self.model.model_name))

        return self.model.predictive_quantile(**kwargs)
