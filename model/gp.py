"""Model definitions for Gaussian Process """
import meta.model as model_template

import numpy as np

import util.dtype as dtype_util
import util.model as model_util
import util.kernel as kernel_util

import tensorflow as tf
from tensorflow_probability import edward2 as ed


class GaussianProcess(model_template.Model):
    def __init__(self, X, y, log_ls, kern_func=kernel_util.rbf):
        """Initializer.

        Args:
            X: (np.ndarray of float32) input training features, shape (N, D).
            log_ls: (float32) length scale parameter.
            kern_func: (callable) kernel function for the gaussian process.
                Default to rbf.
        """
        self.model_name = "Gaussian Process"
        self.param_names = ("gp",)
        self.sample_names = ("gp",)

        # initiate parameter dictionaries.
        super().__init__(self.param_names, self.sample_names)

        # data handling
        self.X = X
        self.y = y
        self.outcome_obs = self.y

        self.ls = np.exp(log_ls)

        self.kern_func = kern_func

        # check data dimension
        self.n_obs, self.n_dim = self.X.shape
        Ny = self.y.size

        self.param_dims = {self.param_names[0]: (self.n_obs,)}

        if self.n_obs != Ny:
            raise ValueError("Sample sizes in X ({}) and "
                             "y ({}) not equal".format(self.n_obs, Ny))

    def definition(self, ridge_factor=1e-3, name="gp",
                   gp_only=False):
        """Defines Gaussian Process prior with kernel_func.

        Args:
            ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
            name: (str) name of the random variable
            gp_only: (bool) Whether only return gp.

        Returns:
            (ed.RandomVariable) A random variable representing the Gaussian Process,
                dimension (N,)

        """
        X = tf.convert_to_tensor(self.X, dtype=dtype_util.TF_DTYPE)
        ls = tf.convert_to_tensor(self.ls, dtype=dtype_util.TF_DTYPE)

        gp_mean = tf.zeros(self.n_obs, dtype=dtype_util.TF_DTYPE)

        # covariance
        K_mat = self.kern_func(X, ls=ls,
                               ridge_factor=ridge_factor)

        gp = ed.MultivariateNormalTriL(
            loc=gp_mean,
            scale_tril=tf.cholesky(K_mat),
            name=name)

        if gp_only:
            return gp

        y = ed.MultivariateNormalDiag(loc=gp,
                                      scale_identity_multiplier=.01,
                                      name="y")
        return y

    def variational_family(self, vi_family="dgpr",
                           ridge_factor=1e-3,
                           name="q_gp",
                           return_vi_param=False,
                           **vi_family_args):
        """Defines the decoupled GP variational family for GPR.

        Args:
            vi_family: (str) `dgpr` (Decoupled GP) or `mfvi` (mean-field VI).
            ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
            name: (str) name for the variational parameter/random variables.
            return_vi_param: (bool) If True then also return var parameters.
        """
        param_dict_all = dict()

        assert len(self.param_names) == 1, "Found {} > 1 params".format(len(self.param_names))

        if vi_family == "dgpr":
            vi_family_function = model_util.dgpr_variational_family
        elif vi_family == "mfvi":
            vi_family_function = model_util.mfvi_variational_family
        else:
            raise ValueError("`vi_family can only be `dgpr` or `mfvi`.")

        for name in self.param_names:
            param_dict_all[name] = vi_family_function(
                X=self.X, ls=self.ls,
                kernel_func=self.kern_func,
                ridge_factor=ridge_factor,
                **vi_family_args,
                name=name)

        self.model_param, self.vi_param = model_util.make_param_dict(param_dict_all)

        if return_vi_param:
            return self.model_param, self.vi_param

        return self.model_param

    def posterior_sample(self, outcome_rv, n_sample):
        """Sample posterior distribution for training sample.

        Args:
            outcome_rv: (ed.RandomVariable) A random variable representing model outcome.
            n_sample: (int) Number of samples to draw.

        Returns:
            (dict of tf.Tensor) A dict of tf.Tensor representing posterior sample.
        """
        post_sample_dict = dict()

        assert len(self.sample_names) == 1, "Found {} > 1 params".format(len(self.sample_names))

        for name in self.sample_names:
            post_sample_dict[name] = outcome_rv.distribution.sample(n_sample)

        return post_sample_dict

    def predictive_sample(self,
                          X_new, f_sample,
                          kernel_func_xn=None,
                          kernel_func_nn=None,
                          ridge_factor=1e-3,
                          return_mean=False, return_vcov=False,
                          return_dict=True):
        """Sample posterior predictive distribution.

        Sample posterior conditional from f^* | f ~ MVN, where:

            E(f*|f) = K(X*, X)K(X, X)^{-1}f
            Var(f*|f) = K(X*, X*) - K(X*, X)K(X, X)^{-1}K(X, X*)

        Args:
            X_new: (np.ndarray of float32) testing locations, (n_obs_new, n_sample)
            f_sample: (np.ndarray of float32) M samples of posterior GP sample,
                N_obs x N_sample
            kernel_func_xn: (function or None) kernel function for distance between X and X_new,
                if None then set to kernel_func.
            kernel_func_nn: (function or None) kernel function for distance among X_new,
                if None then set to kernel_func.
            ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition.

        Returns:
             (np.ndarray of float32) posterior predictive mean samples,
                shape (n_sample, n_obs_new)
        """
        ls = tf.convert_to_tensor(self.ls, dtype=dtype_util.TF_DTYPE)
        X = tf.convert_to_tensor(self.X, dtype=dtype_util.TF_DTYPE)
        X_new = tf.convert_to_tensor(X_new, dtype=dtype_util.TF_DTYPE)
        f_sample = tf.convert_to_tensor(f_sample, dtype=dtype_util.TF_DTYPE)

        N_new, _ = X_new.shape.as_list()
        M, N = f_sample.shape.as_list()

        if kernel_func_xn is None:
            kernel_func_xn = self.kern_func
        if kernel_func_nn is None:
            kernel_func_nn = self.kern_func

        # compute basic components
        Kxx = kernel_func_nn(X_new, X_new, ls=ls)
        Kx = kernel_func_xn(X, X_new, ls=ls)
        K = self.kern_func(X, ls=ls, ridge_factor=ridge_factor)
        K_inv = tf.matrix_inverse(K)

        # compute conditional mean and variance.
        mu_sample = tf.matmul(Kx,
                              tf.matmul(K_inv, f_sample, transpose_b=True),
                              transpose_a=True)
        Sigma = Kxx - tf.matmul(Kx, tf.matmul(K_inv, Kx), transpose_a=True)

        # sample
        with tf.Session() as sess:
            cond_means, cond_cov, Kxx_val = sess.run([mu_sample, Sigma, Kxx])

        if return_mean:
            return cond_means.astype(dtype_util.NP_DTYPE)

        if return_vcov:
            return cond_cov.astype(dtype_util.NP_DTYPE)

        f_new_centered = np.random.multivariate_normal(
            mean=[0] * N_new, cov=cond_cov, size=M).T
        f_new = f_new_centered + cond_means

        # convert to tensor and organize shape to (n_sample, n_obs_new)
        f_new = tf.convert_to_tensor(f_new, dtype=dtype_util.TF_DTYPE)
        f_new = tf.transpose(f_new, perm=[1, 0])

        # finally, produce outcome dictionary
        if not return_dict:
            return f_new

        pred_sample_dict = dict()

        assert len(self.sample_names) == 1, "Found {} > 1 params".format(len(self.sample_names))

        for name in self.sample_names:
            pred_sample_dict[name] = f_new

        return pred_sample_dict
