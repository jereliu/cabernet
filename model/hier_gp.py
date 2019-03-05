"""Model definitions for Hierarchical Gaussian Process with Linear Mean Function."""
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
            resid_log_ls: (float32) length-scale parameter for residual GP.
                If None then will estimate with normal prior.

        Raises:
            (ValueError) If number of predictions in base_pred does not
                match sample size.
            (ValueError) If log_ls_resid is empty when add_resid=True.
        """
        self.model_name = "Hierarchical Gaussian Process"

        self.param_names = ("mean_weight", "resid_func", "log_sigma")

        self.sample_names = ("mean_func", "resid_func",
                             "noise", "log_sigma",
                             "y")

        # initiate parameter dictionaries.
        super().__init__(self.param_names, self.sample_names)

        # data handling
        self.X = X
        self.y = y
        self.base_pred = base_pred
        self.base_pred_array = _make_base_pred_array(self.base_pred)
        self.outcome_obs = self.y

        self.add_resid = add_resid
        self.resid_log_ls = resid_log_ls
        self.resid_kern_func = resid_kern_func
        self.resid_model = None

        # record statistics
        self.n_obs, self.n_dim = self.X.shape
        self.n_model = len(base_pred)

        self.param_dims = {"mean_weight": (self.n_model,),
                           "resid_func": (self.n_obs,),
                           "log_sigma": ()}

        # check data
        Ny = self.y.size

        if self.n_obs != Ny:
            raise ValueError("Sample sizes in X ({}) and "
                             "y ({}) not equal".format(self.n_obs, Ny))

        for key, value in base_pred.items():
            if not value.shape == (self.n_obs,):
                raise ValueError(
                    "All base-model predictions should have shape ({},), but"
                    "observed {} for '{}'".format(self.n_obs, value.shape, key))

        if self.add_resid and not self.resid_log_ls:
            raise ValueError("log_ls_resid cannot be None if add_resid=True")

        # initiate residual model if add_resid = True.
        if self.add_resid:
            self.resid_model = model.GaussianProcess(X=self.X, y=self.y,
                                                     log_ls=self.resid_log_ls,
                                                     kern_func=self.resid_kern_func)
            self.resid_model.param_names = ("resid_func",)

    def definition(self, **resid_kwargs):
        """Sets up model definition and parameters.

        Args:
            **resid_kwargs: Keyword arguments for GaussianProcess model
                definition.

        Returns:
            (ed.RandomVariable) outcome random variable.
        """
        # convert data type
        F = tf.convert_to_tensor(self.base_pred_array,
                                 dtype=dtype_util.TF_DTYPE)

        # specify mean function
        W = ed.MultivariateNormalDiag(loc=tf.zeros(shape=(self.n_model,)),
                                      scale_identity_multiplier=_WEIGHT_PRIOR_SDEV,
                                      name="mean_weight")

        FW = tf.matmul(F, tf.expand_dims(W, -1))
        mean_func = tf.reduce_sum(FW, axis=1, name="mean_func")

        # specify residual function
        resid_func = 0.
        if self.add_resid:
            resid_func = self.resid_model.definition(gp_only=True,
                                                     name="resid_func",
                                                     **resid_kwargs)

        # specify observational noise
        sigma = ed.Normal(loc=_LOG_NOISE_PRIOR_MEAN,
                          scale=_LOG_NOISE_PRIOR_SDEV, name="log_sigma")

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

        for param_name, param_dim in self.param_dims.items():
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

    def posterior_sample(self, rv_dict, n_sample):
        """Sample posterior distribution for training sample.

        Args:
            rv_dict: (dict of RandomVariable) Dictionary of RandomVariables
                following same structure as self.model_param
            n_sample: (int) Number of samples.

        Returns:
            (dict of tf.Tensor) A dictionary of tf.Tensor representing
             sampled values, shape (n_sample, param_dims)
        """
        post_sample_dict = dict()

        # fill in parameter samples, shape (n_sample, param_dims)
        for param_name in self.param_names:
            if not self.add_resid and param_name == "resid_func":
                continue

            post_sample_dict[param_name] = (
                rv_dict[param_name].distribution.sample(n_sample))

        # add mean_func, shape (n_sample, n_obs)
        post_sample_dict["mean_func"] = tf.matmul(
            post_sample_dict["mean_weight"],  # shape (n_sample, n_model)
            self.base_pred_array,  # shape (n_obs, n_model)
            transpose_b=True
        )

        # make noise, shape (n_sample, n_obs)
        post_sample_dict["noise"] = model_util.sample_noise_using_sigma(
            log_sigma_sample=post_sample_dict["log_sigma"],
            n_obs=self.n_obs)

        # add y, shape (n_sample, n_obs)
        post_sample_dict["y"] = (
                post_sample_dict["mean_func"] +
                post_sample_dict["noise"])

        if self.add_resid:
            post_sample_dict["y"] += post_sample_dict["resid_func"]

        return post_sample_dict

    def predictive_sample(self,
                          X_new, base_pred_new,
                          post_sample_dict,
                          **resid_kwargs):
        """Samples new observations.

        Args:
            X_new: (np.ndarray) New observations of shape (n_obs_new, n_dim).
            base_pred_new: (dict) Dictionary of n_model new predictions,
                each of shape (n_obs_new, ).
            post_sample_dict: (dict of np.ndarray) Dictionary of sampled values.
                following same format as output of posterior_sample.
            **resid_kwargs: Keyword arguments to pass to
                GaussianProcess.predictive_sample

        Returns:
            (dict of tf.Tensor) Dictionary of predictive samples,
                with keys containing those in self.sample_names.
        """

        # first make base_pred_array
        base_pred_array_new = _make_base_pred_array(base_pred_new)
        n_obs_new, n_model_new = base_pred_array_new.shape

        if n_model_new != self.n_model:
            raise ValueError(
                "Number of models in base_pred_"
                "new should be {}, observed {}.".format(self.n_model,
                                                        n_model_new))

        # make mean_func prediction, shape (n_sample, n_obs_new)
        pred_sample_dict = dict()

        pred_sample_dict["mean_func"] = tf.matmul(
            post_sample_dict["mean_weight"],  # shape (n_sample, n_model)
            base_pred_array_new,  # shape (n_obs_new, n_model)
            transpose_b=True
        )

        # make noise, shape (n_sample, n_obs)
        pred_sample_dict["log_sigma"] = (
            tf.convert_to_tensor(post_sample_dict["log_sigma"],
                                 dtype=dtype_util.TF_DTYPE))
        pred_sample_dict["noise"] = model_util.sample_noise_using_sigma(
            log_sigma_sample=post_sample_dict["log_sigma"],
            n_obs=n_obs_new)

        # add y, shape (n_sample, n_obs)
        pred_sample_dict["y"] = (
                pred_sample_dict["mean_func"] +
                pred_sample_dict["noise"])

        # make resid_func prediction, shape (n_sample, n_obs_new)
        if self.add_resid:
            pred_sample_dict["resid_func"] = (
                self.resid_model.predictive_sample(
                    X_new, f_sample=post_sample_dict["resid_func"],
                    return_dict=False,
                    **resid_kwargs))

            pred_sample_dict["y"] += pred_sample_dict["resid_func"]

        return pred_sample_dict

    def predictive_cdf(self, y_eval, sample_dict):
        """Produces predictive CDFs.

        Args:
            y_eval (tf.Tensor) y locations to evaluate CDF, shape (n_eval, )
            sample_dict: (dict of tf.Tensor) Dictionary of posterior/predictive
                samples, with keys containing those in self.sample_names.

        Returns:
            (dict of tf.Tensor) A dictionary of two items:
                `perc_eval`:   y locations where CDF are evaluated.
                `cdf`:      predictive CDF values for n_obs locations
                            in sample_dict, evaluated at perc_eval,
                            shape (n_eval, n_obs).
        """

        # type handling
        y_eval = tf.convert_to_tensor(y_eval,
                                      dtype=dtype_util.TF_DTYPE)

        # eval over cdf functions, shape (n_eval, n_sample, n_obs)
        pred_dist = self._make_sample_distribution(sample_dict)
        cdf_func = pred_dist.cdf
        cdf_val_samples = tf.map_fn(cdf_func, y_eval)

        # average over posterior sample, shape  (n_eval, n_obs)
        cdf_vals = tf.reduce_mean(cdf_val_samples, axis=1)

        # return dictionary
        pred_cdf_dict = dict()

        pred_cdf_dict["perc_eval"] = y_eval
        pred_cdf_dict["cdf"] = cdf_vals

        return pred_cdf_dict

    def predictive_quantile(self, perc_eval, sample_dict):
        """Produces predictive quantiles.

        Args:
            perc_eval (tf.Tensor) percentage values to compute quantiles for,
                shape (n_eval, )
            sample_dict: (dict of tf.Tensor) Dictionary of posterior/predictive
                samples, with keys containing those in self.sample_names.

        Returns:
            (dict of tf.Tensor) A dictionary of two items:
                `perc_eval`:   y locations where CDF are evaluated.
                `cdf`:      predictive CDF values for n_obs locations
                            in sample_dict, evaluated at perc_eval,
                            shape (n_eval, n_obs).
        """

        # type handling
        perc_eval = tf.convert_to_tensor(perc_eval,
                                         dtype=dtype_util.TF_DTYPE)

        # eval over cdf functions, shape (n_eval, n_sample, n_obs)
        pred_dist = self._make_sample_distribution(sample_dict)
        quant_func = pred_dist.quantile
        quant_val_samples = tf.map_fn(quant_func, perc_eval)

        # average over posterior sample, shape  (n_eval, n_obs)
        quant_vals = tf.reduce_mean(quant_val_samples, axis=1)

        # return dictionary
        pred_quant_dict = dict()

        pred_quant_dict["perc_eval"] = perc_eval
        pred_quant_dict["quantile"] = quant_vals

        return pred_quant_dict

    @staticmethod
    def _make_sample_distribution(sample_dict):
        """Produce a tf.Distribution object from sample dictionary."""
        # check sample_dict
        required_keys = ("y", "noise", "log_sigma")
        for key in required_keys:
            try:
                sample_dict[key]
            except KeyError:
                raise ValueError(
                    "`sample_dict` must contain key {}".format(key))

        # compute posterior sample, shape (n_sample, n_obs)
        pred_sample_mean = sample_dict["y"] - sample_dict["noise"]
        pred_sample_scale = tf.exp(sample_dict["log_sigma"])

        # tile pred_sample_scale to match shape with sample_mean
        n_sample, n_obs = pred_sample_mean.shape
        pred_sample_scale = tf.tile(tf.expand_dims(pred_sample_scale, -1),
                                    multiples=[1, n_obs])

        # eval over cdf functions, shape (n_eval, n_sample, n_obs)
        return tfd.Normal(loc=pred_sample_mean,
                          scale=pred_sample_scale)


def _make_base_pred_array(base_pred):
    """Makes a ndarray from dictionary of base-model predictive values.

    Args:
        base_pred: (dict of np.ndarray) A dictionary of length n_model
         recording prediction for n_obs.

    Returns:
        (np.ndarray) A array recording prediction values of base models,
            shape (n_obs, n_model)
    """
    return np.asarray(list(base_pred.values()),
                      dtype=dtype_util.NP_DTYPE).T
