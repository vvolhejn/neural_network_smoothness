"""
Functions for analysis in Jupyter Notebooks.
"""
from typing import List

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd
import scipy.stats

import smooth


def get_kendall_coefs(
    df: pd.DataFrame,
    hyperparam_cols: List[str],
    result_col: str,
    measure_cols: List[str],
):
    """
    Given a dataframe containing the hyperparameters and measures of trained models,
    computes Kendall's tau coefficients between a measure to be predicted (result_col)
    and the measures used to predict it (measure_cols).

    Following "Fantastic Generalization Measures and Where to Find Them",
    we also compute the granulated Kendall's coefficient, meaning we split the dataset up
    into groups where only one hyperparameter changes, compute taus and then average them.
    We then define psi as the average of these coefficients.
    """

    res = pd.DataFrame(
        index=pd.Index(measure_cols, name="measures"),
        columns=pd.Index(hyperparam_cols + ["overall_tau", "psi"], name="Kendall"),
    )

    def get_kendall(df, measure):
        return scipy.stats.kendalltau(df[result_col], df[measure]).correlation

    for measure in measure_cols:
        res["overall_tau"][measure] = get_kendall(df, measure)

        psi_total = 0
        for hyperparam in hyperparam_cols:
            other_columns = hyperparam_cols.copy()
            other_columns.remove(hyperparam)
            groups = df.groupby(other_columns)
            groups = groups.apply(lambda group_df: get_kendall(group_df, measure))
            res[hyperparam][measure] = groups.mean()
            psi_total += res[hyperparam][measure]

        res["psi"][measure] = psi_total / len(hyperparam_cols)

    return res.astype("float32")


def pad_bounds(mn, mx, coef):
    assert mn <= mx
    width = mx - mn
    return (mn - width * coef, mx + width * coef)


def plot_shallow_relu(
    model: tf.keras.Sequential,
    dataset: smooth.datasets.Dataset,
    epochs=None,
    weights_history=None,
    title=None,
):
    """
    Plots a model trained by `smooth.model.train_shallow_relu`.

    :param model:
    :param dataset:
    :param epochs:
    :param weights_history: A dict mapping from epoch number to model weights
        (from model.get_weights). If provided, an animation is returned.
    :param title:
    :return:
    """
    xlim = pad_bounds(dataset.x_train.min(), dataset.x_train.max(), 0.5)
    ylim = pad_bounds(dataset.y_train.min(), dataset.y_train.max(), 1)
    x = np.linspace(xlim[0], xlim[1], 101)

    ax, ax_hist = None, None  # type: plt.Axes
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    elif hasattr(model, "plot_title"):
        ax.set_title(model.plot_title)

    ax.scatter(dataset.x_train, dataset.y_train, marker="o", color="green")
    (prediction_line,) = ax.plot([], [])
    (kinks_plot,) = ax.plot(
        [],
        [],
        marker="x",
        color="red",
        linestyle="None",
        alpha=(1.0 / np.size(model.get_weights()[0])) ** 0.5
    )
    epoch_text = ax.text(0.01, 0.95, "", transform=ax.transAxes)

    def get_kinks(weights, mn, mx):
        w = np.reshape(weights[0], (-1))
        b = weights[1]
        kinks = -b / w
        kinks = kinks[~np.isnan(kinks)]
        mask = (mn < kinks) & (kinks < mx)
        kinks_x = kinks[mask]
        kinks_y = np.squeeze(w[mask])
        return kinks_x, kinks_y

    ax_hist.hist(get_kinks(model.get_weights(), xlim[0], xlim[1])[0], bins=50)

    if epochs is not None:
        ax.text(0.01, 0.95, "Epochs: {}".format(epochs), transform=ax.transAxes)

    def update(weights, epoch=None):
        old_weights = model.get_weights()
        model.set_weights(weights)
        y = model.predict(x)
        prediction_line.set_data(x, y)

        kinks_x, kinks_y = get_kinks(weights, xlim[0], xlim[1])
        kinks_plot.set_data(kinks_x, kinks_y)

        if epoch is not None:
            epoch_text.set_text("Epochs: {}".format(epoch))

        model.set_weights(old_weights)
        return prediction_line, kinks_plot, epoch_text

    if weights_history is None:
        update(model.get_weights())
        plt.show()
        return None
    else:
        epochs = sorted(weights_history.keys())
        epochs.append(-1)
        weights_history[-1] = model.get_weights()

        def animation_update(i):
            return update(weights_history[epochs[i]], epoch=epochs[i])

        animation = matplotlib.animation.FuncAnimation(
            fig, animation_update, len(epochs), interval=50, blit=True
        )
        plt.close()
        return animation
