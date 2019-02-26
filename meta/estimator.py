"""Class template for models."""
import collections

from abc import ABC, abstractmethod

import meta.model as model_template

# set up EstimatorOps container with defaults
EstimatorOps = collections.namedtuple("EstimatorOps",
                                      ["init", "train", "loss",
                                       "save", "pred", "summary"])
EstimatorOps.__new__.__defaults__ = (None,) * len(EstimatorOps._fields)


class Estimator(ABC):
    """Class template for model parameter/statistic estimators."""

    def __init__(self, model):
        """Initializer."""
        self.graph = None
        self.param = None
        self.ops = None

        self.__model = None
        self.model = model

    @abstractmethod
    def config(self):
        """Sets up estimator graph."""
        ...

    @abstractmethod
    def run(self, sess):
        """Executes estimator graph in a given session.

        Args:
            sess: (tf.Session) A session to run graph in.
        """
        ...

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model):
        """Setter method to make model property immutable."""
        if self.model:
            raise ValueError("model already defined and cannot be changed.")

        if not isinstance(new_model, model_template.Model):
            raise ValueError("The input model is not a valid Model class.")

        self.__model = new_model
