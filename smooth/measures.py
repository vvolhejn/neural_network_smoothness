"""
Functions used to measure trained models.
"""
from math import ceil

import tensorflow as tf
import numpy as np
import sklearn.kernel_ridge

import smooth.datasets
import smooth.model


def get_measures(
    model: tf.keras.Model,
    dataset: smooth.datasets.Dataset,
    samples=1000,
    # If set, instead of sampling segments use one segment spanning the entire domain
    precise_in_1d=False,
    is_classification=False,
):
    """
    Takes a model's measures on a dataset.
    Returns a dictionary with the measured values.
    """
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
                y[:samples], y_pred
            )
        except ValueError as e:
            return {"error": str(e)}

    if smooth.model.is_differentiable(model):
        # These measures can only be computed if we have access to the gradient.
        res["gradient_norm_train"] = gradient_norm(model, x_train[:samples])
        res["gradient_norm_test"] = gradient_norm(model, x_test[:samples])
        res["gradient_norm_squared_train"] = gradient_norm(model, x_train[:samples], power=2)
        res["gradient_norm_squared_test"] = gradient_norm(model, x_test[:samples], power=2)

        # Technically, we need a two-layer NN for these measures.
        # But in our case, differentiable == two-layer NN.
        res["weights_rms"] = weights_rms(model)
        res["weights_product"] = float(weights_product(model))

    try:
        res["actual_epochs"] = len(model.history.history["loss"])
    except AttributeError:
        # If we're measuring a saved model, this information is no longer available
        # and `model.history` does not exist.
        pass

    is_1d = x_test[0].squeeze().shape == () and y[0].squeeze().shape == ()

    if is_1d and precise_in_1d:
        # 1D case - we can compute measures deterministically
        # However we get slightly different measures because of the weighing
        x_min, x_max = np.min(x_test, axis=0), np.max(x_test, axis=0)
        res["path_length_f_test"] = total_variation_along_segment(
            model, x_min, x_max, n_samples=samples, derivative=False,
        )

        if smooth.model.is_differentiable(model):
            res["path_length_d_test"] = total_variation_along_segment(
                model, x_min, x_max, n_samples=samples, derivative=True,
            )
    else:
        # General case
        for suffix, x in [("_train", x_train), ("_test", x_test)]:
            res["path_length_f" + suffix] = path_length(
                model, x, segments_per_batch=100, segments=samples
            )

            if smooth.model.is_differentiable(model):
                res["path_length_d" + suffix] = path_length(
                    model, x, derivative=True, segments_per_batch=10, segments=samples,
                )

            if is_classification:
                # For classification, we also take the measures after softmax.
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
    """Returns the root mean square of the model's weights and biases."""
    total = 0
    n_weights = 0
    for weight_matrix in model.get_weights():
        total += np.sum(weight_matrix ** 2)
        n_weights += weight_matrix.size

    return np.sqrt(total / n_weights)


def weights_product(model: tf.keras.Model):
    """
    Takes a two-layer NN. For each hidden unit, computes ||w^(1)|| * |w^(2)|
    and returns the sum of these values.
    """
    # Warning: Returns a tf.Tensor
    # We take only the last 4 elements of `model.weights` since there might be
    # a PCA layer before that
    w1, _, w2, _ = model.weights[-4:]
    return tf.squeeze(tf.tensordot(tf.norm(w1, axis=0), tf.abs(w2), axes=1))


def gradient_norm(model: tf.keras.Model, x, power=1):
    """
    Evaluates `model` in data points `x`. For each point, calculates the norm
    of the gradient. Returns the mean of the norms.
    """
    was_tensor = True

    if not isinstance(x, tf.Tensor):
        was_tensor = False
        x = tf.constant(x)

    with tf.GradientTape() as g:
        g.watch(x)
        y = model(x)

    dy_dx = g.batch_jacobian(y, x)

    # Flatten each batch (matters when y is not just scalar, i.e. for classification)
    shape = (-1, np.prod(dy_dx.shape[1:]))
    dy_dx = tf.reshape(dy_dx, shape)
    # OLD:
    # For classification, we consider each function x -> prob. that x is a certain class
    # separately, take the norms of the functions' gradients and take the mean.
    # NEW:
    # We take the norm of the entire (d_out x d_in) matrix

    res = tf.reduce_mean(tf.norm(dy_dx, axis=-1) ** power)

    if not was_tensor:
        res = float(res)

    return res


@tf.custom_gradient
def stable_norm(x):
    """
    There are numerical issues when the taking the gradient of ||x|| when x is close to 0.
    This fixes them.
    """
    y = tf.norm(x, axis=2)

    def grad(dy):
        return tf.expand_dims(dy, axis=-1) * (x / tf.expand_dims(y + 1e-19, axis=-1))

    return y, grad


def total_variation(samples: tf.Tensor):
    """
    Given evenly spaced samples of a function's values, computes an approximation
    of the total variation, that is the sum of the distances of consecutive samples.
    For scalar samples, this means the sum of absolute values of the first difference,
    for vector-valued functions we sum the l2 norms of the first difference.

    The first axis is interpreted as the batch axis.

    >>> float(total_variation(tf.constant([[1, 2, 3, 1]], dtype=tf.float32)))
    4.0
    >>> l = total_variation(tf.constant([[[0, 0], [1, 1], [1, 2]]], dtype=tf.float32))
    >>> print("{:.3f}".format(float(l)))
    2.414
    >>> np.array(total_variation(tf.constant([[0, 0], [1, 1], [1, 2]], dtype=tf.float32)))
    array([0., 0., 1.], dtype=float32)
    """
    assert isinstance(samples, tf.Tensor)

    res = samples[:, 1:, :] - samples[:, :-1, :]
    res = stable_norm(res)
    res = tf.reduce_sum(res, axis=1)

    return res


def _interpolate(a: tf.Tensor, b: tf.Tensor, n_samples):
    """
    >>> _interpolate(1., 2., 3).round(1).tolist()
    [1.0, 1.5, 2.0]
    >>> _interpolate([0, 3], [3, 0], 4).round(1).tolist()
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    >>> _interpolate([[0, 2], [1, 1]], [[2, 0], [2, 2]], 3).round(1).tolist()
    [[[0.0, 2.0], [1.0, 1.0]], [[1.0, 1.0], [1.5, 1.5]], [[2.0, 0.0], [2.0, 2.0]]]
    """
    w = tf.linspace(0.0, 1.0, n_samples)
    res = tf.tensordot(w, a, axes=0) + tf.tensordot(1.0 - w, b, axes=0)
    return res


def total_variation_along_segment(
    model: tf.keras.Model,
    x1: tf.Tensor,
    x2: tf.Tensor,
    n_samples: int,
    derivative: bool,
    softmax=False,
):
    """
    Computes the total variation of a model's function along a segment.
    Optionally takes the derivative (gradient) first.
    """
    assert not softmax
    tf.debugging.assert_all_finite(x1, "x1")
    tf.debugging.assert_all_finite(x2, "x2")
    samples = _interpolate(x1, x2, n_samples)

    samples_flat = tf.reshape(
        samples, tf.concat([[-1], tf.shape(samples)[2:]], axis=0),
    )

    if not derivative:
        output_flat = model(samples_flat)
        tf.debugging.assert_all_finite(output_flat, "output_flat0")
        if softmax:
            output_flat = tf.nn.softmax(output_flat)
    else:
        with tf.GradientTape() as g:
            x = samples_flat
            g.watch(x)
            y = model(x)
            if softmax:
                y = tf.nn.softmax(y)

        output_flat = g.batch_jacobian(y, x)
        output_flat = tf.reshape(output_flat, [tf.shape(output_flat)[0], -1])

    tf.debugging.assert_all_finite(output_flat, "output_flat")

    output = tf.reshape(
        output_flat, tf.concat([[n_samples, -1], tf.shape(output_flat)[1:]], axis=0)
    )

    # Swap the first two axes so that the batch axis comes first.
    p = tf.concat([[1, 0], tf.range(2, tf.rank(output))], axis=0)
    output = tf.transpose(output, p)

    res = total_variation(output)
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
    Takes two random points from `x_input` and calculates the model's total variation
    along the segment between the two points.
    Repeats `n_segments` times and takes the mean.
    """
    x = x_input[np.random.randint(len(x_input), size=(2 * segments,))]
    x = np.array([x[:segments], x[segments:]])
    x = np.swapaxes(x, 0, 1)

    batches = np.array_split(x, ceil(segments / segments_per_batch))

    results = []
    for batch in batches:
        x1, x2 = np.swapaxes(batch, 0, 1)
        x1 = tf.constant(x1)
        x2 = tf.constant(x2)
        cur = total_variation_along_segment(
            model, x1, x2, samples_per_segment, derivative, softmax=softmax
        )

        results.append(np.array(cur))

    return np.mean(np.concatenate(results))


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
