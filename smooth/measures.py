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


def _total_variation(samples, batch=False):
    """
    Given evenly spaced samples of a function's values, computes an approximation
    of the total variation, that is the sum of the distances of consecutive samples.
    For scalar samples, this means the sum of absolute values of the first difference,
    for vector-valued functions we sum the l2 norms of the first difference.

    If `batch` is set, we interpret the first axis as the batch axis
    and treat batches separately.

    >>> _total_variation([1, 2, 3, 1])
    4.0
    >>> print("{:.3f}".format(_total_variation([[0, 0], [1, 1], [1, 2]])))
    2.414
    >>> _total_variation([[0, 0], [1, 1], [1, 2]], batch=True)
    array([0., 0., 1.])
    """
    if not batch:
        samples = np.array([samples])
    res = np.diff(samples, axis=1)
    if res.ndim == 2:
        res = res[:, :, np.newaxis]
    res = np.linalg.norm(res, axis=2)
    res = np.sum(res, axis=1)

    if not batch:
        assert len(res) == 1
        return res[0]
    else:
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


def _segment_total_variation(
    model: tf.keras.Model, x1, x2, n_samples: int, derivative: bool, batch=False
):
    if not batch:
        x1 = [x1]
        x2 = [x2]
    x1 = np.array(x1)
    x2 = np.array(x2)
    n_segments = len(x1)
    assert x1.shape == x2.shape
    samples = _interpolate(x1, x2, n_samples)
    samples_flat = np.reshape(samples, (n_samples * n_segments,) + samples.shape[2:])

    if not derivative:
        output_flat = model.predict(samples_flat, batch_size=1024)
    else:
        with tf.GradientTape() as g:
            x = tf.constant(samples_flat)
            g.watch(x)
            y = model(x)
        output_flat = g.batch_jacobian(y, x)
        # We just stretch the Jacobian into a single vector and take its total variation
        # (meaning we sum the Frobenius norms of the first difference)
        # Does this make any sense mathematically?
        output_flat = np.reshape(output_flat, (len(samples_flat), -1))

    output = np.reshape(output_flat, (n_samples, n_segments) + output_flat.shape[1:])
    output = np.swapaxes(output, 0, 1)
    # at this point, `output` has shape (n_segments, n_samples, n_classes)
    res = _total_variation(output, batch=True)

    if not batch:
        assert len(res) == 1
        return res[0]
    else:
        return res


def segments_total_variation(
    model: tf.keras.Model,
    x_input,
    segments=1000,
    samples_per_segment=100,
    segments_per_batch=10,
    derivative=False,
):
    """
    Takes two random points from `x_input` and calculates the total variation
    of the model's prediction along the line segment between the two points.
    This is repeated `n_segments` times and the average total variation is taken.
    """

    x = x_input[np.random.randint(len(x_input), size=(2 * segments,))]
    x = np.array([x[:segments], x[segments:]])
    x = np.swapaxes(x, 0, 1)

    batches = np.array_split(x, segments // segments_per_batch)

    results = []
    for batch in batches:
        x1, x2 = np.swapaxes(batch, 0, 1)
        cur = _segment_total_variation(model, x1, x2, samples_per_segment, derivative,
                                       batch=True)
        results.append(cur)

    return np.mean(results)

    #
    # res = 0
    # for i in range(n_segments):
    #     x1, x2 = x_input[np.random.randint(len(x_input), size=(2,))]
    #     res += _segment_total_variation(
    #         model, [x1], [x2], n_samples_per_segment, derivative
    #     )
    #
    # return res / n_segments