import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def square_dist(X, X2=None, ls=1.):
    """Computes Square distance between two sets of features.

    Referenced from GPflow.kernels.Stationary.

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale.

    Returns:
        (tf.Tensor) A N x N2 tensor for ||x-x'||^2 / ls**2

    Raises:
        (ValueError) If feature dimension of X and X2 disagrees.
    """
    N, D = X.shape

    X = X / ls
    Xs = tf.reduce_sum(tf.square(X), axis=1)

    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        return tf.clip_by_value(dist, 0., np.inf)

    N2, D2 = X2.shape
    if D != D2:
        raise ValueError('Dimension of X and X2 does not match.')

    X2 = X2 / ls
    X2s = tf.reduce_sum(tf.square(X2), axis=1)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
    return tf.clip_by_value(dist, 0., np.inf)


def rbf(X, X2=None, ls=1., ridge_factor=0.):
    """Defines RBF kernel.

     k(x, x') = - exp(- |x-x'| / ls**2)

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        (tf.Tensor) A N x N2 tensor for exp(-||x-x'||**2 / 2 * ls**2)
    """
    N, _ = X.shape.as_list()
    if ridge_factor and X2 is None:
        ridge_mat = ridge_factor * tf.eye(N, dtype=tf.float32)
    else:
        ridge_mat = 0

    return tf.exp(-square_dist(X, X2, ls=ls) / 2) + ridge_mat
