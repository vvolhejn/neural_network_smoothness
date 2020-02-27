from typing import Callable
from math import ceil

import tensorflow as tf
import numpy as np


def get_measures(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    # dataset: ClassificationDataset,
    include_training_measures=True,
    samples=1000,
    # If set, instead of sampling segments use one segment spanning the entire domain
    precise_in_1d=True,
    is_classification=False,
):
    res = {}

    # For metrics recorded by keras itself, such as loss, accuracy/mae
    tf_metrics = dict(
        zip(model.metrics_names, model.evaluate(x, y, batch_size=256, verbose=0),)
    )

    for k, v in tf_metrics.items():
        res["test_{}".format(k)] = v

    # This might happen if the learning rate or init scale is too high
    if not np.isfinite(tf_metrics["loss"]):
        return res

    grad_norm = gradient_norm(model, x[:samples])
    l2 = average_l2(model)

    is_1d = x[0].squeeze().shape == () and y[0].squeeze().shape == ()

    if is_1d and precise_in_1d:
        # 1D case - we can solve this deterministically
        x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
        path_length_f = path_length_one_sample(
            model, x_min, x_max, n_samples=samples, derivative=False,
        )
        path_length_d = path_length_one_sample(
            model, x_min, x_max, n_samples=samples, derivative=True,
        )
    else:
        path_length_f = path_length(
            model, x, segments_per_batch=100, segments=samples,
        )
        path_length_d = path_length(
            model, x, derivative=True, segments_per_batch=10, segments=samples,
        )
        if is_classification:
            path_length_f_softmax = path_length(
                model, x, segments_per_batch=100, segments=samples, softmax=True,
            )
            path_length_d_softmax = path_length(
                model,
                x,
                derivative=True,
                segments_per_batch=10,
                segments=samples,
                softmax=True,
            )
            res.update(
                path_length_f_softmax=path_length_f_softmax,
                path_length_d_softmax=path_length_d_softmax,
            )

    res.update(
        gradient_norm=grad_norm,
        l2=l2,
        path_length_f=path_length_f,
        path_length_d=path_length_d,
    )

    if include_training_measures:
        history = model.history.history
        for k in model.metrics_names:
            res["train_{}".format(k)] = history[k][-1]

        res.update(
            # loss=history["loss"][-1],
            # accuracy=history["accuracy"][-1],
            # val_loss=history.get("val_loss", [None])[-1],
            # val_accuracy=history.get("val_accuracy", [None])[-1],
            actual_epochs=len(history["loss"]),
        )

    return res


def average_l2(model: tf.keras.Model):
    """ This is not really l2! """
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
