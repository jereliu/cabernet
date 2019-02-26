"""Model definitions for Hierarchical Gaussian Process with Linear Mean Function."""
import meta.model as model_template

import numpy as np

import model

import util.dtype as dtype_util
import util.model as model_util
import util.kernel as kernel_util

import tensorflow as tf
from tensorflow_probability import edward2 as ed

_WEIGHT_PRIOR_SDEV = np.array(1.).astype(dtype_util.NP_DTYPE)
_LOG_NOISE_PRIOR_MEAN = np.array(-1.).astype(dtype_util.NP_DTYPE)
_LOG_NOISE_PRIOR_SDEV = np.array(1.).astype(dtype_util.NP_DTYPE)


class HierarchicalGP(model_template.Model):
    def __init__(self, X, base_pred, y,
                 add_resid=True,
                 resid_log_ls=None,
                 resid_kern_func=kernel_util.rbf):
        """Initializer.

        Args:
            X: (np.ndarray) Input features of dimension (N, D).
            base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
                from base models. For each item in the dictionary,
                key is the model name, and value is the model prediction with
                dimension (N, ).
            y: (np.ndarray) Input labels of dimension (N, ).
            add_resid: (bool) Whether to add residual process to model.
            log_ls_resid: (float32) length-scale parameter for residual GP.
                If None then will estimate with normal prior.

        Raises:
            (ValueError) If number of predictions in base_pred does not
                match sample size.
            (ValueError) If log_ls_resid is empty when add_resid=True.
        """
        self.model_name = "Hierarchical Gaussian Process"

        self.param_name = ("mean_weight", "resid_func", "sigma")

        self.sample_name = ("mean_func", "resid_func", "sigma")

        # initiate parameter dictionaries.
        super().__init__(self.param_name, self.sample_name)

        # data handling
        self.X = X
        self.y = y
        self.base_pred = base_pred

        self.add_resid = add_resid
        self.resid_log_ls = resid_log_ls
        self.resid_kern_func = resid_kern_func
        self.resid_model = None

        # record statistics
        self.n_sample, self.n_dim = self.X.shape
        self.n_model = len(base_pred)

        self.param_dim = {"mean_weight": (self.n_model, 1),
                          "resid_func": (self.n_sample,),
                          "sigma": ()}

        # check data
        Ny = self.y.size

        if self.n_sample != Ny:
            raise ValueError("Sample sizes in X ({}) and "
                             "y ({}) not equal".format(self.n_sample, Ny))

        for key, value in base_pred.items():
            if not value.shape == (self.n_sample,):
                raise ValueError(
                    "All base-model predictions should have shape ({},), but"
                    "observed {} for '{}'".format(self.n_sample, value.shape, key))

        if self.add_resid and not self.resid_log_ls:
            raise ValueError("log_ls_resid cannot be None if add_resid=True")

        # initiate residual model if add_resid = True.
        if self.add_resid:
            self.resid_model = model.GaussianProcess(X=self.X, y=self.y,
                                                     log_ls=self.resid_log_ls,
                                                     kern_func=self.resid_kern_func)
            self.resid_model.param_name = ("resid_func",)

    def definition(self, **resid_kwargs):
        """Sets up model definition and parameters.

        Args:
            **resid_kwargs: Keyword arguments for GaussianProcess model
                definition.

        Returns:
            (ed.RandomVariable) outcome random variable.
        """
        # convert data type
        F = np.asarray(list(self.base_pred.values())).T
        F = tf.convert_to_tensor(F, dtype=dtype_util.TF_DTYPE)

        # specify mean function
        W = ed.MultivariateNormalDiag(loc=tf.zeros(shape=(self.n_model, 1)),
                                      scale_identity_multiplier=_WEIGHT_PRIOR_SDEV,
                                      name="mean_weight")

        FW = tf.matmul(F, W)
        mean_func = tf.reduce_sum(FW, axis=1, name="mean_func")

        # specify residual function
        resid_func = 0.
        if self.add_resid:
            resid_func = self.resid_model.definition(gp_only=True,
                                                     name="resid_func",
                                                     **resid_kwargs)

        # specify observational noise
        sigma = ed.Normal(loc=_LOG_NOISE_PRIOR_MEAN,
                          scale=_LOG_NOISE_PRIOR_SDEV, name="sigma")

        # specify outcome
        y = ed.MultivariateNormalDiag(loc=mean_func + resid_func,
                                      scale_identity_multiplier=tf.exp(sigma),
                                      name="y")

        return y

    def variational_family(self, **resid_kwargs):
        """Defines variational family and parameters.

        Args:
            **resid_kwargs: Keyword arguments for GaussianProcess model's
                variational family.
        """
        param_dict_all = dict()

        for param_name, param_dim in self.param_dim.items():
            if param_name == "resid_func":
                continue

            param_dict_all[param_name] = (
                model_util.normal_variational_family(shape=param_dim,
                                                     name=param_name))

        # compile rv and param dicts
        self.model_param, self.vi_param = model_util.make_param_dict(param_dict_all)

        # Optionally, also define vi family for resid_gp
        if self.add_resid:
            resid_model_param, resid_vi_param = (
                self.resid_model.variational_family(**resid_kwargs,
                                                    name="resid_func",
                                                    return_vi_param=True)
            )
            self.model_param.update(resid_model_param)
            self.vi_param.update(resid_vi_param)

        return self.model_param

    def posterior_sample(self):
        """Sample posterior distribution for training sample."""
        ...

    def predictive_sample(self):
        """Samples new observations."""
        ...
