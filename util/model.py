"""Utility and helper functions for building models."""

import tensorflow as tf
from tensorflow_probability import edward2 as ed

from tensorflow.python.ops.distributions.util import fill_triangular

import util.dtype as dtype_util
import util.kernel as kernel_util

import util.distribution as dist_util


def make_param_dict(param_val_dict):
    """Makes dictionary for VI RandomVariable and parameters.

    :param param_val_dict: (dict of list) A dictionary with key being
        parameter names, and values being a list of
        [RandomVariable, mean_vi_param, scale_vi_param]

    :return:
        model_param: (dict of RandomVariable)
            Keys are variable names, Values are RandomVariable output
            from VI families.
        vi_param: (dict) Keys are variable names, Values are
            dict of loc and scale VI parameters.
    """
    model_param = dict()
    vi_param = dict()

    for key, value in param_val_dict.items():
        model_param[key] = value[0]
        vi_param[key] = {"loc": value[1], "scale": value[2]}

    return model_param, vi_param


def make_value_setter(**model_kwargs):
    """Creates a value-setting interceptor.

    Args:
      **model_kwargs: dict of str to Tensor. Keys are the names of random variable
        in the model to which this interceptor is being applied. Values are
        Tensors to set their value to.

    Returns:
      set_values: Function which sets the value of intercepted ops.
    """

    def set_values(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        return ed.interceptable(f)(*args, **kwargs)

    return set_values


def normal_variational_family(shape, name=None,
                              init_loc=None,
                              init_scale=None,
                              trainable=True):
    """Defines mean-field variational family.

    Args:
        shape: (tuple of int) Defines shape of variational random variable.
        name: (str) Name of the random variable.
        init_loc: (float32) Initial values for q_mean.
        init_scale: (float32) Initial values for q_log_sd.
        trainable: (bool) Whether the variational family is trainable.

    Returns:
        q_rv (ed.RandomVariable) RandomVariable representing the variational family.
        q_mean, q_log_sd (tf.Variable) Variational parameters.

    Raises:
        (ValueError) init_loc/init_scale is None when trainable=False.
    """
    if not trainable:
        if init_loc is None or init_scale is None:
            raise ValueError("Initial values cannot be None if trainable=False.")

    with tf.variable_scope("q_{}_scope".format(name),
                           default_name="normal_variational"):
        # TODO(jereliu): manipulate such that we can initiate these values.
        #  allow user to set trainable=False and give value through initializer.
        q_mean = tf.get_variable("q_mean",
                                 shape=shape,
                                 dtype=dtype_util.TF_DTYPE,
                                 initializer=init_loc,
                                 trainable=trainable)
        q_log_sd = tf.get_variable("q_log_sd",
                                   shape=shape,
                                   dtype=dtype_util.TF_DTYPE,
                                   initializer=init_scale,
                                   trainable=trainable)

        rv_shape = shape if shape is not None else init_loc.shape

        if len(rv_shape) == 0:
            q_rv = ed.Normal(loc=q_mean,
                             scale=tf.exp(q_log_sd),
                             name="q_{}".format(name))
        else:
            q_rv = ed.MultivariateNormalDiag(
                loc=q_mean,
                scale_diag=tf.exp(q_log_sd),
                name="q_{}".format(name)
            )

        return q_rv, q_mean, q_log_sd


def dgpr_variational_family(X, Z, Zm=None, ls=1.,
                            kernel_func=kernel_util.rbf, ridge_factor=1e-3,
                            name="", **kwargs):
    """Defines the decoupled GP variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, shape (Ns, D).
        Zm: (np.ndarray of float32 or None) inducing points for mean, shape (Nm, D).
            If None then same as Z
        ls: (float32) length scale parameter.
        kernel_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
        mfvi_mixture: (float32) Whether to output variational family with a
            mixture of MFVI.
        n_mixture: (int) Number of MFVI mixture component to add.
        name: (str) name for the variational parameter/random variables.
        kwargs: Dict of other keyword variables.
            For compatibility purpose with other variational family.

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    X = tf.convert_to_tensor(X)
    Zs = tf.convert_to_tensor(Z)
    Zm = tf.convert_to_tensor(Zm) if Zm is not None else Zs

    Nx, Nm, Ns = X.shape.as_list()[0], Zm.shape.as_list()[0], Zs.shape.as_list()[0]

    # 1. Prepare constants
    # compute matrix constants
    Kxx = kernel_func(X, ls=ls)
    Kmm = kernel_func(Zm, ls=ls)
    Kxm = kernel_func(X, Zm, ls=ls)
    Kxs = kernel_func(X, Zs, ls=ls)
    Kss = kernel_func(Zs, ls=ls, ridge_factor=ridge_factor)

    # 2. Define variational parameters
    # define free parameters (i.e. mean and full covariance of f_latent)
    m = tf.get_variable(shape=[Nm, 1], name='{}_mean_latent'.format(name))
    s = tf.get_variable(shape=[Ns * (Ns + 1) / 2], name='{}_cov_latent_s'.format(name))
    L = fill_triangular(s, name='{}_cov_latent_chol'.format(name))

    # components for KL objective
    H = tf.eye(Ns) + tf.matmul(L, tf.matmul(Kss, L), transpose_a=True)
    cond_cov_inv = tf.matmul(L, tf.matrix_solve(H, tf.transpose(L)))

    func_norm_mm = tf.matmul(m, tf.matmul(Kmm, m), transpose_a=True)
    log_det_ss = tf.log(tf.matrix_determinant(H))
    cond_norm_ss = tf.reduce_sum(tf.multiply(Kss, cond_cov_inv))

    # compute sparse gp variational parameter (i.e. mean and covariance of P(f_obs | f_latent))
    qf_mean = tf.squeeze(tf.tensordot(Kxm, m, [[1], [0]]), name='{}_mean'.format(name))
    qf_cov = (Kxx -
              tf.matmul(Kxs, tf.matmul(cond_cov_inv, Kxs, transpose_b=True)) +
              ridge_factor * tf.eye(Nx, dtype=tf.float32)
              )

    # 3. Define variational family
    q_f = dist_util.VariationalGaussianProcessDecoupled(loc=qf_mean,
                                                        covariance_matrix=qf_cov,
                                                        func_norm_mm=func_norm_mm,
                                                        log_det_ss=log_det_ss,
                                                        cond_norm_ss=cond_norm_ss,
                                                        name=name)
    return q_f, qf_mean, qf_cov
