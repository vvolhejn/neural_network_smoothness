import tensorflow as tf
import numpy as np


def average_l2(model: tf.keras.Model):
    total_l2 = 0
    n_weights = 0
    for weight_matrix in model.get_weights():
        total_l2 += np.sum(weight_matrix ** 2)
        n_weights += weight_matrix.size

    return total_l2 / n_weights


def gradient_norm(model, x_input):
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
