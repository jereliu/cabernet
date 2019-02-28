"""Model definitions for Monotonic Gaussian Process with Identity Mean Function."""
import meta.model as model_template

import numpy as np

import model

import util.dtype as dtype_util
import util.model as model_util
import util.kernel as kernel_util

import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

_WEIGHT_PRIOR_SDEV = np.array(1.).astype(dtype_util.NP_DTYPE)
_LOG_NOISE_PRIOR_MEAN = np.array(-1.).astype(dtype_util.NP_DTYPE)
_LOG_NOISE_PRIOR_SDEV = np.array(1.).astype(dtype_util.NP_DTYPE)


class MonoGP(model_template.Model):
    def __init__(self, X, cdf_dict, y,
                 log_ls, kern_func=kernel_util.rbf):
        """Initializer.

        Args:
            X: (np.ndarray of float32) Training label, shape (n_obs, n_dim),
            cdf_dict: (dict of np.ndarray) A dictionary of two items:
                `y_eval`:   y locations where CDF are evaluated,
                            shape (n_eval, ).
                `cdf`:      predictive CDF values for n_obs locations
                            in sample_dict, evaluated at y_eval,
                            shape (n_eval, n_obs).
            y: (np.ndarray of float32) Training labels, shape (n_obs, ).
            log_ls: (float32) length scale parameter.
            kern_func: (function) kernel function for the gaussian process.
                Default to rbf.
        """

        self.model_name = "Monotonic Gaussian Process"
        self.param_names = ("gp", "log_sigma",)
        self.sample_names = ("gp", "log_sigma",
                             "noise", "y")

        # initiate parameter dictionaries.
        super().__init__(self.param_names, self.sample_names)

        # data handling
        self.X = X
        self.y = y
        self.y_eval = cdf_dict["y_eval"]
        self.cdf_val = cdf_dict["cdf"]

        self.ls = tf.exp(log_ls)
        self.kern_func = kern_func

        # record statistics
        self.n_obs, self.n_dim = self.X.shape
        self.n_eval = len(self.y_eval)

        self.param_dims = {"gp": (self.n_eval * self.n_obs,),
                           "log_sigma": ()}

        # check data
        Ny = self.y.size
        if self.n_obs != Ny:
            raise ValueError("Sample sizes in X ({}) and "
                             "y ({}) not equal".format(self.n_obs, Ny))

        # make model and empirical cdfs, shape (n_eval*n_obs, ...)
        (self.model_cdf,
         self.cdf_feature) = self._make_cdf_features(self.cdf_val, self.X)
        self.empir_cdf = self._make_cdf_labels(self.y_eval, self.y)

        # initiate a zero-mean GP.
        self.gp_model = model.GaussianProcess(X=self.cdf_feature,
                                              y=self.empir_cdf,
                                              log_ls=log_ls,
                                              kern_func=kern_func)

    def definition(self):
        ...

    def variational_family(self):
        """Defines variational family and parameters."""
        ...

    def posterior_sample(self):
        """Sample posterior distribution for training sample."""
        ...

    def predictive_sample(self):
        """Samples new observations."""
        ...

    def predictive_cdf(self):
        """Samples new observations."""
        ...

    @staticmethod
    def _make_cdf_features(cdf_array, X):
        """Produces CDF features [F(y|X), X].

        Outputs an array [F(y|x), x] of shape (n_eval * n_obs, 1 + n_dim).

        Args:
            cdf_array: (np.ndarray) CDF values of shape (n_eval, n_obs)
            X: (np.ndarray) Features of shape (n_obs, n_dim)

        Returns:
            cdf_feature (np.ndarray) CDF feature only, shape (n_eval * n_obs, )
            feature_all (np.ndarray) CDF and original input features of shape (n_eval * n_obs, 1 + n_dim).
        """

        n_eval, n_obs_cdf = cdf_array.shape
        n_obs, n_dim = X.shape

        if n_obs_cdf != n_obs:
            raise ValueError("Sample sizes in cdf_array ({})"
                             " and X ({}) doesn't match.".format(n_obs_cdf, n_obs))

        cdf_feature = np.expand_dims(cdf_array, -1)  # shape (n_eval, n_obs, 1)
        X_feature = np.tile(np.expand_dims(X, 0),
                            (n_eval, 1, 1))  # shape (n_eval, n_obs, n_dim)

        # assemble features to wide format, shape (n_eval, n_obs, 1 + n_dim)
        feature_all = np.concatenate([cdf_feature, X_feature], axis=-1)

        # convert features to long format, shape (n_eval * n_obs, 1 + n_dim)
        feature_all = feature_all.reshape(n_eval * n_obs, 1 + n_dim)
        feature_cdf = feature_all[:, 0]

        return feature_cdf, feature_all

    @staticmethod
    def _make_cdf_labels(y_eval, y):
        """Makes empirical cdf I(y < y_eval).

        Args:
            y_eval: (np.ndarray of float32) y locations where CDF
                are evaluated, shape (n_eval, ).
            y: (np.ndarray of float32) Training labels, shape (n_obs, ).

        Returns:
            (n_eval, n_obs) Evaluated empirical cdf.
        """
        return _make_empirical_cdf(y_eval, y)


def _make_empirical_cdf(y_eval, y, flatten=True):
    """Makes empirical cdf I(y < y_eval).

    Args:
        y_eval: (np.ndarray of float32) y locations where CDF
            are evaluated, shape (n_eval, ).
        y: (np.ndarray of float32) Training labels, shape (n_obs, ).

    Returns:
        (np.ndarray) Evaluated empirical cdf,
            shape (n_eval, n_obs) if flatten = False,
            or (n_eval * n_obs, ) if flatten = True
    """
    # reshape input for broadcasting
    y_eval = y_eval.reshape((y_eval.size, 1))  # shape (n_eval, 1)
    y_obs = y.reshape((1, y.size))  # shape (1, n_obs)

    # compute empirical CDF using broadcasting
    emp_cdf_array = (y_eval > y_obs)  # shape (n_eval, n_obs)
    emp_cdf_array = emp_cdf_array.astype(dtype_util.NP_DTYPE)

    if flatten:
        emp_cdf_array = emp_cdf_array.flatten()

    return emp_cdf_array


def _process_cdf_array(cdf_array, process_type="long"):
    """Reshapes a CDF array for training/plotting.

    Converts shape of cdf array between (n_eval, n_obs) (wide format)
        and (n_eval*n_obs,) (long format).

    Args:
        cdf_array: (np.ndarray) Input cdf_array in long / wide format.
            shape (n_eval, n_obs) if wide, or (n_eval*n_obs, ) if long.
        process_type: (str) Should be `long`.

    Returns:
        (np.ndarray) cdf array in converted shape.
    """
    if process_type == 'long':
        return cdf_array.flatten()
