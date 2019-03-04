"""Model definitions for Monotonic Gaussian Process with Identity Mean Function."""
import tqdm

import numpy as np

import meta.model as model_template
import model

import util.dtype as dtype_util
import util.model as model_util
import util.kernel as kernel_util

import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

_CDF_PENALTY_MEAN_SHIFT = np.array(-5e-3).astype(dtype_util.NP_DTYPE)
_CDF_PENALTY_SCALE = np.array(1e-3).astype(dtype_util.NP_DTYPE)

_WEIGHT_PRIOR_SDEV = np.array(1.).astype(dtype_util.NP_DTYPE)
_LOG_NOISE_PRIOR_MEAN = np.array(-1.).astype(dtype_util.NP_DTYPE)
_LOG_NOISE_PRIOR_SDEV = np.array(1.).astype(dtype_util.NP_DTYPE)


class MonoGP(model_template.Model):
    """Class definition for CDF calibration model.

    Prior:
        noise ~ N(0, sigma)
        F     ~ GP(F_S, kern_func(F_S, X))

    Model:
        F_emp = F + noise
    """

    def __init__(self, X, quant_dict, y,
                 log_ls,
                 kern_func=kernel_util.rbf,
                 activation=model_util.relu1):
        """Initializer.

        Args:
            X: (np.ndarray of float32) Training label, shape (n_obs, n_dim),
            quant_dict: (dict of np.ndarray) A dictionary of two items:
                `perc_eval`:    y locations where CDF are evaluated,
                                shape (n_eval, ).
                `quantile`:     predictive CDF values for n_obs locations
                                in sample_dict, evaluated at perc_eval,
                                shape (n_eval, n_obs).
            y: (np.ndarray of float32) Training labels, shape (n_obs, ).
            log_ls: (float32) length scale parameter.
            kern_func: (function) kernel function for the gaussian process.
                Default to rbf.
        """

        self.model_name = "Monotonic Gaussian Process"
        self.param_names = ("gp", "log_sigma",)
        self.sample_names = ("gp", "log_sigma",
                             "noise", "cdf")

        # initiate parameter dictionaries.
        super().__init__(self.param_names, self.sample_names)

        # data handling
        self.X = X
        self.y = y
        self.perc_eval = quant_dict["perc_eval"]
        self.quant_val = quant_dict["quantile"]

        self.ls = tf.exp(log_ls)
        self.kern_func = kern_func
        self.activation = activation

        # record statistics
        self.n_obs, self.n_dim = self.X.shape
        self.n_eval = len(self.perc_eval)

        self.n_cdf_obs = self.n_eval * self.n_obs
        self.param_dims = {"gp": (self.n_cdf_obs,),
                           "log_sigma": ()}

        # check data
        Ny = self.y.size
        if self.n_obs != Ny:
            raise ValueError("Sample sizes in X ({}) and "
                             "y ({}) not equal".format(self.n_obs, Ny))

        # make model and empirical cdfs, shape (n_eval*n_obs, ...)
        (self.model_cdf,
         self.cdf_feature) = self._make_cdf_features(self.perc_eval, self.X)
        self.empir_cdf = self._make_cdf_labels(self.quant_val, self.y)

        # initiate a zero-mean GP.
        self.gp_model = model.GaussianProcess(X=self.cdf_feature,
                                              y=self.empir_cdf,
                                              log_ls=log_ls,
                                              kern_func=kern_func)
        self.outcome_obs = self.empir_cdf

    def definition(self, **resid_kwargs):
        """Defines Gaussian process with identity mean function.

        Args:
            **resid_kwargs: Keyword arguments for GaussianProcess model
                definition.

        Returns:
            (ed.RandomVariable) outcome random variable.
        """
        # specify identity mean function
        mean_func = self.model_cdf

        # specify residual function
        gp = self.gp_model.definition(gp_only=True,
                                      name="gp",
                                      **resid_kwargs)

        # specify observational noise
        sigma = ed.Normal(loc=_LOG_NOISE_PRIOR_MEAN,
                          scale=_LOG_NOISE_PRIOR_SDEV, name="log_sigma")

        # specify outcome
        cdf_mean = mean_func + gp

        if self.activation:
            cdf_mean = self.activation(cdf_mean)

        cdf = ed.MultivariateNormalDiag(loc=cdf_mean,
                                        scale_identity_multiplier=tf.exp(sigma),
                                        name="cdf")

        return cdf

    def likelihood(self, outcome_rv, outcome_value,
                   cdf_constraint=False,
                   constraint_penalty=_CDF_PENALTY_SCALE):
        """Returns tensor of constrained model likelihood.

        Adds Probit range constraints to Gaussian process in log likelihood.

        Note:
            Currently cdf_constraint will over-constraint CDF-estimate to be
                away from 0 and 1. More research needed.

        Args:
            outcome_rv: (ed.RandomVariable) A random variable representing model outcome.
            outcome_value: (np.ndarray) Values of the training data.
            cdf_constraint: (bool) Whether to constraint cdf.
            constraint_penalty: (float) Penalty factor for likelihood constraints.

        Returns:
            (tf.Tensor) A tf.Tensor representing likelihood values to be optimized.
        """
        log_penalties = 0.

        if cdf_constraint:
            # construct penalties
            cdf_ge_zero = tfd.Normal(
                loc=_CDF_PENALTY_MEAN_SHIFT,
                scale=constraint_penalty).log_cdf(outcome_rv)
            cdf_le_one = tfd.Normal(
                loc=_CDF_PENALTY_MEAN_SHIFT,
                scale=constraint_penalty).log_cdf(1 - outcome_rv)

            log_penalties = [cdf_ge_zero, cdf_le_one]
            log_penalties = tf.reduce_mean(log_penalties)

        # define likelihood
        log_likehood = outcome_rv.distribution.log_prob(outcome_value)

        return log_likehood + log_penalties

    def variational_family(self, **resid_kwargs):
        """Defines variational family and parameters.

        Args:
            **resid_kwargs: Keyword arguments for GaussianProcess model's
                variational family.
        """
        param_dict_all = dict()

        for param_name, param_dim in self.param_dims.items():
            if param_name == "gp":
                continue

            param_dict_all[param_name] = (
                model_util.normal_variational_family(shape=param_dim,
                                                     name=param_name))

        # compile rv and param dicts
        self.model_param, self.vi_param = model_util.make_param_dict(param_dict_all)

        # Add vi family for resid_gp
        gp_model_param, gp_vi_param = (
            self.gp_model.variational_family(**resid_kwargs,
                                             name="gp",
                                             return_vi_param=True, )
        )
        self.model_param.update(gp_model_param)
        self.vi_param.update(gp_vi_param)

        return self.model_param

    def posterior_sample(self, rv_dict, n_sample):
        """Sample posterior distribution for training sample.

        Args:
            rv_dict: (dict of RandomVariable) Dictionary of RandomVariables
                following same structure as self.model_param
            n_sample: (int) Number of samples.

        Returns:
            (dict of tf.Tensor) A dictionary of tf.Tensor representing
             sampled values, shape (n_sample, n_cdf_obs)
        """
        post_sample_dict = dict()

        # fill in parameter samples, shape (n_sample, param_dims)
        for param_name in self.param_names:
            post_sample_dict[param_name] = (
                rv_dict[param_name].distribution.sample(n_sample))

        # add mean_func, shape (n_sample, n_cdf_obs)
        post_sample_dict["mean_func"] = tf.tile(
            tf.expand_dims(self.model_cdf, 0),
            multiples=[n_sample, 1]
        )

        # make noise, shape (n_sample, n_cdf_obs)
        post_sample_dict["noise"] = model_util.sample_noise_using_sigma(
            log_sigma_sample=post_sample_dict["log_sigma"],
            n_obs=self.n_cdf_obs)

        # make cdf prediction, shape (n_sample, n_cdf_obs)
        post_sample_dict["y_eval"] = tf.convert_to_tensor(
            self.quant_val, dtype=dtype_util.TF_DTYPE)

        cdf_mean = post_sample_dict["mean_func"] + post_sample_dict["gp"]
        if self.activation:
            cdf_mean = self.activation(cdf_mean)

        post_sample_dict["cdf"] = cdf_mean + post_sample_dict["noise"]
        post_sample_dict["cdf_orig"] = tf.reshape(self.model_cdf,
                                                  shape=(self.n_eval, self.n_obs))

        return post_sample_dict

    def predictive_sample(self,
                          X_new, quant_dict_new,
                          post_sample_dict,
                          reshape=True,
                          verbose=False,
                          **resid_kwargs):
        """Samples new observations.

        Args:
            X_new: (np.ndarray) New observations of shape (n_obs_new, n_dim).
            quant_dict_new: (dict) Dictionary of cdf values at prediction locations,
                contains:
                    `perc_eval`: Number of eval locations (n_eval_new, )
                    `quantile`: (n_eval_new, n_obs_new).
            post_sample_dict: (dict of np.ndarray) Dictionary of sampled values.
                following same format as output of posterior_sample.
            **resid_kwargs: Keyword arguments to pass to
                GaussianProcess.predictive_sample

        Returns:
            (dict of tf.Tensor) Dictionary of predictive cdf samples,
                with keys containing those in self.sample_names.

            Specifically, pred_sample_dict["cdf"] is of shape
                (n_sample, n_eval_new, n_obs_new, )
        """
        # prepare predictive features, shape (n_cdf_obs_new, 1 + n_dim)
        # where n_cdf_obs_new = n_eval_new * n_obs_new
        cdf_val_new = quant_dict_new["perc_eval"]
        y_eval_new = quant_dict_new["quantile"]

        n_sample, n_cdf_obs = post_sample_dict["cdf"].shape
        n_eval_new, n_obs_new = y_eval_new.shape
        n_obs_X_new, n_dim_X_new = X_new.shape
        n_cdf_obs_new = n_eval_new * n_obs_new

        if n_dim_X_new != self.n_dim:
            raise ValueError(
                "Feature dimension in X_new ({}) and "
                "model dimension ({}) not equal!".format(n_dim_X_new, self.n_dim))

        if n_obs_X_new != n_obs_new:
            raise ValueError(
                "Sample size in X_new ({}) and "
                "quant_dict_new ({}) not equal!".format(n_obs_X_new, n_obs_new))

        model_cdf_new, cdf_feature_new = (
            self._make_cdf_features(cdf_val_new, X_new))

        # prepare prediction dictionary, all shape (n_sample, n_cdf_obs_new)
        pred_sample_dict = dict()

        # make gp prediction (looping over chunks to avoid large matrix computation)
        gp_pred_list = []
        gp_feature_iter = np.split(cdf_feature_new, n_eval_new)
        if verbose:
            gp_feature_iter = tqdm.tqdm(gp_feature_iter)

        for cdf_feature_chunk in gp_feature_iter:
            pred_sample_chunk = (
                self.gp_model.predictive_sample(
                    cdf_feature_chunk,
                    f_sample=post_sample_dict["gp"],
                    return_dict=False,
                    **resid_kwargs))
            gp_pred_list.append(pred_sample_chunk)

        pred_sample_dict["gp"] = tf.concat(gp_pred_list, axis=-1)

        # make mean_func prediction
        pred_sample_dict["mean_func"] = tf.tile(
            tf.expand_dims(model_cdf_new, 0),
            multiples=[n_sample, 1]
        )

        # make noise prediction.
        pred_sample_dict["log_sigma"] = (
            tf.convert_to_tensor(post_sample_dict["log_sigma"],
                                 dtype=dtype_util.TF_DTYPE))
        pred_sample_dict["noise"] = model_util.sample_noise_using_sigma(
            log_sigma_sample=post_sample_dict["log_sigma"],
            n_obs=n_cdf_obs_new)

        # make cdf prediction, shape (n_sample, n_cdf_obs_new)
        pred_sample_dict["y_eval"] = tf.convert_to_tensor(
            y_eval_new, dtype=dtype_util.TF_DTYPE)

        cdf_mean = pred_sample_dict["mean_func"] + pred_sample_dict["gp"]
        if self.activation:
            cdf_mean = self.activation(cdf_mean)

        pred_sample_dict["cdf"] = cdf_mean + pred_sample_dict["noise"]

        if reshape:
            # reshape cdf to (n_sample, n_eval_new, n_obs_new)
            pred_sample_dict["cdf"] = tf.reshape(
                pred_sample_dict["cdf"],
                shape=(n_sample, n_eval_new, n_obs_new))

        # make original cdf prediction, shape (n_eval_new, n_obs_new)
        pred_sample_dict["cdf_orig"] = tf.reshape(model_cdf_new,
                                                  shape=(n_eval_new, n_obs_new))

        return pred_sample_dict

    @staticmethod
    def _make_cdf_features(perc_array, X, flatten=True):
        """Produces CDF features [F(y|X), X].

        Outputs an array [F(y|x), x] of shape (n_eval * n_obs, 1 + n_dim).

        Args:
            perc_array: (np.ndarray) CDF values of shape (n_eval, ).
            X: (np.ndarray) Features of shape (n_obs, n_dim).
            flatten: (bool) Whether to flatten output.

        Returns:
            cdf_feature (np.ndarray) CDF feature only, shape (n_eval * n_obs, )
            feature_all (np.ndarray) CDF and original input features of shape (n_eval * n_obs, 1 + n_dim).
        """

        n_eval, = perc_array.shape
        n_obs, n_dim = X.shape

        # repeat
        cdf_feature = np.tile(np.reshape(perc_array, [perc_array.size, 1, 1]),
                              (1, n_obs, 1))  # shape (n_eval, n_obs, 1)
        X_feature = np.tile(np.expand_dims(X, 0),
                            (n_eval, 1, 1))  # shape (n_eval, n_obs, n_dim)

        # assemble features to wide format, shape (n_eval, n_obs, 1 + n_dim)
        feature_all = np.concatenate([cdf_feature, X_feature], axis=-1)

        # convert features to long format, shape (n_eval * n_obs, 1 + n_dim)
        if flatten:
            feature_all = feature_all.reshape(n_eval * n_obs, 1 + n_dim)

        feature_cdf = feature_all[..., 0]

        return feature_cdf, feature_all

    @staticmethod
    def _make_cdf_labels(y_eval, y, flatten=True):
        """Makes empirical cdf I(y < perc_eval).

        Args:
            y_eval: (np.ndarray of float32) y locations where CDF
                are evaluated, shape (n_eval, n_obs).
            y: (np.ndarray of float32) Training labels, shape (n_obs, ).

        Returns:
            (n_eval, n_obs) Evaluated empirical cdf.
        """
        return _make_empirical_cdf(y_eval, y, flatten=flatten)


def _make_empirical_cdf(y_eval, y, flatten=True):
    """Makes empirical cdf I(y < perc_eval).

    Args:
        y_eval: (np.ndarray of float32) y locations where CDF
            are evaluated, shape (n_eval, n_obs).
        y: (np.ndarray of float32) Training labels, shape (n_obs, ).

    Returns:
        (np.ndarray) Evaluated empirical cdf,
            shape (n_eval, n_obs) if flatten = False,
            or (n_eval * n_obs, ) if flatten = True
    """
    # reshape input for broadcasting
    n_eval, n_obs = y_eval.shape
    if y.size != n_obs:
        raise ValueError("Sample size of y_eval ({}) and "
                         "y ({}) doesn't match!".format(n_obs, y.size))

    y_obs = np.tile(np.expand_dims(y, 0), (n_eval, 1))  # shape (n_eval, n_obs)

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
