"""Class template for models."""
import collections

from abc import ABC, abstractmethod

MODEL_PARAM_DEFAULT = None
SAMPLE_PARAM_DEFAULT = None
VI_PARAM_DEFAULT = {"loc": None, "scale": None}


class Model(ABC):
    """Class template for models"""

    def __init__(self, param_name, sample_name):
        """Initializer."""
        # check model/parameter/sample names are given
        if not self.model_name or not self.param_name or not self.sample_name:
            raise ValueError("`model_name`/`param_name`/`sample_name` empty.")

        # initialize parameter containers.
        self.model_param = dict(zip(param_name,
                                    [MODEL_PARAM_DEFAULT, ] * len(param_name)))
        self.vi_param = dict(zip(param_name,
                                 [VI_PARAM_DEFAULT, ] * len(param_name)))
        self.sample_dict = dict(zip(sample_name,
                                    [SAMPLE_PARAM_DEFAULT, ] * len(sample_name)))

    @abstractmethod
    def definition(self):
        """Adds model parameter nodes to graph."""
        ...

    @staticmethod
    def likelihood(outcome_rv, outcome_value):
        """Returns tensor of model likelihood.

        Args:
            outcome_rv: (ed.RandomVariable) A random variable representing model outcome.
            outcome_value: (np.ndarray) Values of the training data.

        Returns:
            (tf.Tensor) A tf.Tensor representing likelihood values to be optimized.
        """
        return outcome_rv.distribution.log_prob(outcome_value)

    @abstractmethod
    def variational_family(self):
        """Defines variational family and parameters."""
        ...

    @abstractmethod
    def posterior_sample(self):
        """Sample posterior distribution for training sample."""
        ...

    @abstractmethod
    def predictive_sample(self):
        """Samples new observations."""
        ...
