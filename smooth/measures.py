from typing import Callable

import tensorflow as tf
import numpy as np


def average_l2(model: tf.keras.Model):
    total_l2 = 0
    n_weights = 0
    for weight_matrix in model.get_weights():
        total_l2 += np.sum(weight_matrix ** 2)
        n_weights += weight_matrix.size

    return total_l2 / n_weights


def gradient_norm(model: tf.keras.Model, x_input):
    with tf.GradientTape() as g:
        x = tf.constant(x_input)
        g.watch(x)
        y = model(x)
    dy_dx = g.batch_jacobian(y, x)
    # We consider each function x -> prob. that x is a "1" separately,
    # take the Frobenius norms of the Jacobians and sum them
    # Should we perhaps take the norm of the entire 10x28x28 tensor?
    res = np.mean(np.linalg.norm(dy_dx, ord="fro", axis=(2, 3)))
    return res


def _total_variation(samples):
    """
    Given evenly spaced samples of a function's values, computes an approximation
    of the total variation, that is the sum of the distances of consecutive samples.

    For scalar samples, this means the sum of absolute values of the first difference,
    for vector-valued functions we sum the l2 norms of the first difference.

    >>> _total_variation([1, 2, 3, 1])
    4.0
    >>> print("{:.3f}".format(_total_variation([[0, 0], [1, 1], [1, 2]])))
    2.414
    """
    res = np.diff(samples, axis=0)
    if res.ndim == 1:
        res = res[:, np.newaxis]
    res = np.linalg.norm(res, axis=1)
    res = np.sum(res)
    return res


def _interpolate(a, b, n_samples):
    """
    >>> _interpolate(1, 2, 3).tolist()
    [1.0, 1.5, 2.0]
    >>> _interpolate([0, 3], [3, 0], 4).tolist()
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    >>> _interpolate([[0, 2], [1, 1]], [[2, 0], [2, 2]], 3).tolist()
    [[[0.0, 2.0], [1.0, 1.0]], [[1.0, 1.0], [1.5, 1.5]], [[2.0, 0.0], [2.0, 2.0]]]
    """
    a, b = np.array(a), np.array(b)
    assert a.shape == b.shape
    w = np.linspace(0, 1, n_samples)
    res = np.outer(1 - w, a) + np.outer(w, b)
    res = np.reshape(res, (-1,) + a.shape)
    return res


def _segment_total_variation(model: tf.keras.Model, x1, x2, n_samples, derivative):
    samples = _interpolate(x1, x2, n_samples)
    if not derivative:
        output = model.predict(samples)
    else:
        with tf.GradientTape() as g:
            x = tf.constant(samples)
            g.watch(x)
            y = model(x)
        dy_dx = g.batch_jacobian(y, x)
    return _total_variation(output)


def segments_total_variation(
    model: tf.keras.Model, x_input, n_segments=1000, n_samples_per_segment=100, derivative=False,
):
    """
    Takes two random points from `x_input` and calculates the total variation
    of the model's prediction along the line segment between the two points.
    This is repeated `n_segments` times and the average total variation is taken.
    """

    res = 0
    for i in range(n_segments):
        x1, x2 = x_input[np.random.randint(len(x_input), size=(2,))]
        res += _segment_total_variation(x1, x2, n_samples_per_segment, derivative)

    return res / n_segments

    # Attempted vectorized version:
    # sample_is = np.random.randint(len(x_input), size=n_segments * 2)
    # endpoints = np.reshape(x_input[sample_is], (n_segments, 2, -1))
    #
    # segments = _interpolate(endpoints[:,0], endpoints[:,1], n_samples_per_segment)
    # print(segments.shape)
    # segments_flat = np.ravel(segments)
    # y = model.predict(segments_flat)
    # y = np.reshape(y, segments.shape)
    # print(y.shape)
    # res = 0
    # for i in range(n_segments):
    #     res += _total_variation(y[:,i])
    #     print(y[:,i], _total_variation(y[:,i]))
    #
    # return res / n_segments
