from typing import Callable
from math import ceil

import tensorflow as tf
import numpy as np
import sklearn.kernel_ridge
import GPy

import smooth.datasets
import smooth.model


def get_measures(
    model: tf.keras.Model,
    dataset: smooth.datasets.Dataset,
    samples=1000,
    # If set, instead of sampling segments use one segment spanning the entire domain
    precise_in_1d=True,
    is_classification=False,
):
    res = {}
    x_train, y_train = dataset.x_train, dataset.y_train
    x_test, y_test = dataset.x_test, dataset.y_test

    if is_classification:
        # TODO: use cross-entropy loss for classification
        raise NotImplementedError

    for name, x, y in [
        ("train", x_train, y_train),
        ("test", x_test, y_test),
    ]:
        try:
            y_pred = model.predict(x[:samples], batch_size=64)

            res["loss_{}".format(name)] = sklearn.metrics.mean_squared_error(
                y[:samples],
                y_pred
            )
        except ValueError as e:
            return {"error": str(e)}

    if smooth.model.is_differentiable(model):
        res["gradient_norm"] = gradient_norm(model, x_test[:samples])
        # Technically, we need a keras model for this.
        # But for now differentiable == keras
        res["weights_rms"] = weights_rms(model)

    try:
        res["actual_epochs"] = len(model.history.history["loss"])
    except AttributeError:
        # If we're measuring a saved model, this information is no longer available
        # and `model.history` does not exist.
        pass

    is_1d = x_test[0].squeeze().shape == () and y[0].squeeze().shape == ()

    if is_1d and precise_in_1d:
        # 1D case - we can solve this deterministically
        x_min, x_max = np.min(x_test, axis=0), np.max(x_test, axis=0)
        res["path_length_f_test"] = path_length_one_sample(
            model, x_min, x_max, n_samples=samples, derivative=False,
        )

        if smooth.model.is_differentiable(model):
            res["path_length_d_test"] = path_length_one_sample(
                model, x_min, x_max, n_samples=samples, derivative=True,
            )
    else:
        for suffix, x in [("_train", x_train), ("_test", x_test)]:
            res["path_length_f" + suffix] = path_length(
                model, x, segments_per_batch=100, segments=samples
            )

            if smooth.model.is_differentiable(model):
                res["path_length_d" + suffix] = path_length(
                    model, x, derivative=True, segments_per_batch=10, segments=samples,
                )

            if is_classification:
                res["path_length_f_softmax" + suffix] = path_length(
                    model, x, segments_per_batch=100, segments=samples, softmax=True,
                )

                if smooth.model.is_differentiable(model):
                    res["path_length_d_softmax" + suffix] = path_length(
                        model,
                        x,
                        derivative=True,
                        segments_per_batch=10,
                        segments=samples,
                        softmax=True,
                    )

    return res


def weights_rms(model: tf.keras.Model):
    total_l2 = 0
    n_weights = 0
    for weight_matrix in model.get_weights():
        total_l2 += np.sum(weight_matrix ** 2)
        n_weights += weight_matrix.size

    return np.sqrt(total_l2 / n_weights)


def gradient_norm(model: tf.keras.Model, x_input):
    with tf.GradientTape() as g:
        x = tf.constant(x_input)
        g.watch(x)
        y = model(x)
    dy_dx = g.batch_jacobian(y, x)
    # For MNIST, we consider each function x -> prob. that x is a "1" separately,
    # take the Frobenius norms of the Jacobians and take the mean
    # Should we perhaps take the norm of the entire 10x28x28 tensor?
    axes_to_sum = tuple(range(2, len(dy_dx.shape)))
    res = np.mean(np.linalg.norm(dy_dx, axis=axes_to_sum))
    return res


def _path_length(samples, batch=False):
    """
    Given evenly spaced samples of a function's values, computes an approximation
    of the path length, that is the sum of the distances of consecutive samples.
    For scalar samples, this means the sum of absolute values of the first difference,
    for vector-valued functions we sum the l2 norms of the first difference.

    If `batch` is set, we interpret the first axis as the batch axis
    and treat batches separately.

    >>> _path_length([1, 2, 3, 1])
    4.0
    >>> print("{:.3f}".format(_path_length([[0, 0], [1, 1], [1, 2]])))
    2.414
    >>> _path_length([[0, 0], [1, 1], [1, 2]], batch=True)
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
    >>> _interpolate(1, 2, 3).round(1).tolist()
    [1.0, 1.5, 2.0]
    >>> _interpolate([0, 3], [3, 0], 4).round(1).tolist()
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    >>> _interpolate([[0, 2], [1, 1]], [[2, 0], [2, 2]], 3).round(1).tolist()
    [[[0.0, 2.0], [1.0, 1.0]], [[1.0, 1.0], [1.5, 1.5]], [[2.0, 0.0], [2.0, 2.0]]]
    """
    a, b = np.array(a), np.array(b)
    assert a.shape == b.shape
    w = np.linspace(0, 1, n_samples, dtype=np.float32)
    res = np.outer(1 - w, a) + np.outer(w, b)
    res = np.reshape(res, (-1,) + a.shape)
    return res


def path_length_one_sample(
    model: tf.keras.Model,
    x1,
    x2,
    n_samples: int,
    derivative: bool,
    batch=False,
    softmax=False,
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
        if softmax:
            output_flat = tf.nn.softmax(output_flat)
    else:
        with tf.GradientTape() as g:
            x = tf.constant(samples_flat)
            g.watch(x)
            y = model(x)
            if softmax:
                y = tf.nn.softmax(y)
        output_flat = g.batch_jacobian(y, x)
        # We just stretch the Jacobian into a single vector and take the path length
        # of this vector "moving around".
        # (meaning we sum the Frobenius norms of the first difference)
        # Does this make any sense mathematically?
        output_flat = np.reshape(output_flat, (len(samples_flat), -1))

    output = np.reshape(output_flat, (n_samples, n_segments) + output_flat.shape[1:])
    output = np.swapaxes(output, 0, 1)
    # at this point, `output` has shape (n_segments, n_samples, n_classes)
    res = _path_length(output, batch=True)

    if not batch:
        assert len(res) == 1
        return res[0]
    else:
        return res


def path_length(
    model: tf.keras.Model,
    x_input,
    segments=1000,
    samples_per_segment=100,
    segments_per_batch=10,
    derivative=False,
    softmax=False,
):
    """
    Takes two random points from `x_input` and calculates the path length of the image
    of the line segment between the two points when passed through the model.
    This is repeated `n_segments` times and the average is taken.
    """

    x = x_input[np.random.randint(len(x_input), size=(2 * segments,))]
    x = np.array([x[:segments], x[segments:]])
    x = np.swapaxes(x, 0, 1)

    batches = np.array_split(x, ceil(segments / segments_per_batch))

    results = []
    for batch in batches:
        x1, x2 = np.swapaxes(batch, 0, 1)
        cur = path_length_one_sample(
            model, x1, x2, samples_per_segment, derivative, batch=True, softmax=softmax
        )
        results.append(cur)

    return np.mean(results)


def path_length_f_lower_bound(dataset: smooth.datasets.Dataset, use_test_set=True):
    """
    Calculates what the lowest achievable `path_length_f` is for a given dataset.
    This is the expected value of the difference of two output points y1, y2.
    """
    y = dataset.y_test if use_test_set else dataset.y_train

    # Currently we can only do scalar outputs.
    assert y.shape[1:] == (1,)

    y = np.sort(y.reshape(-1))
    dy = np.diff(y, axis=0)

    # For datasets where the output is scalar, we can compute this in O(n log n):
    # We imagine the points on the real line and draw a line segment between every
    # pair of points. We want to calculate the total length of these line segments.
    # Notice that the number of segments which cross the segment between y[i] and y[i+1]
    # is (# points on the left) * (# points on the right). Thus we can calculate
    # the total length by weighting the first difference by these coefficients.
    n = len(y)
    w = np.arange(1, n) * (n - np.arange(1, n))

    return 2 * (dy * w).sum() / (n ** 2)
