"""
Functions for analysis in Jupyter Notebooks.
"""
from typing import List, Callable
import os
import logging

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd
import scipy.stats
import tqdm.notebook
import seaborn as sns
import GPy

import smooth.datasets
import smooth.model
import smooth.measures


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


def plot_shallow(
    model: tf.keras.Sequential,
    dataset: smooth.datasets.Dataset,
    epochs=None,
    weights_history=None,
    title=None,
):
    """
    Plots a model trained by `smooth.model.train_shallow`. Mainly intended for ReLU
    models (else "kinks" are not really kinks), but applicable to other activations
    as well.

    :param model:
    :param dataset:
    :param epochs:
    :param weights_history: A dict mapping from epoch number to model weights
        (from model.get_weights). If provided, an animation is returned.
    :param title:
    :return:
    """
    xlim = pad_bounds(dataset.x_train.min(), dataset.x_train.max(), 0.25)
    ylim = pad_bounds(dataset.y_train.min(), dataset.y_train.max(), 0.5)
    x = np.linspace(xlim[0], xlim[1], 101, dtype=np.float32)

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

    ax.scatter(dataset.x_train, dataset.y_train, marker="o", color="green", alpha=0.5)
    (prediction_line,) = ax.plot([], [])
    (kinks_plot,) = ax.plot(
        [],
        [],
        marker="x",
        color="red",
        linestyle="None",
        alpha=(1.0 / np.size(model.get_weights()[0])) ** 0.5,
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
        # y = model.predict(x)
        y = model(x)
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


def remove_constant_columns(df: pd.DataFrame, verbose=False, to_keep: List[str]=[]):
    """
    Removes those columns of a DataFrame which are equal across all rows.
    The DF is modified in-place and also returned.
    """
    removed = []
    for col in df.columns:
        if col not in to_keep and df[col].nunique(dropna=False) == 1:
            removed.append(col)
            del df[col]

    if verbose:
        print("Removed columns:", ", ".join(removed))

    return df


def expand_dataset_columns(df: pd.DataFrame):
    """
    Given an analysis DataFrame where the dataset is a `GaussianProcessDataset`,
    expands the dataset's name into columns describing its properties
    (seed, lengthscale and samples_train).
    If not all datasets are `GaussianProcessDataset`s, returns `df` unchanged.
    """
    dataset_cols = df["dataset"].str.split("-", expand=True)

    if dataset_cols[0].nunique() > 1:
        # Not all datasets are of the same kind - do nothing
        return df

    dataset_name = dataset_cols.iloc[0, 0]
    del dataset_cols[0]  # A constant like "gp" or "mnist" etc.

    prefix = ""
    if dataset_name == "gp":
        # Naming format:
        # gp-{dim}-{seed}-{lengthscale}-{samples_train}[-{noise_var}[-{disjoint}]]
        params = [
            ("dim", np.int32),
            ("seed", np.int32),
            ("lengthscale", np.float32),
            ("samples_train", np.int32),
            ("noise_var", np.float32),
            ("disjoint", np.int32),
        ]
        dataset_cols.columns = [prefix + name for name, _ in params][
            : len(dataset_cols.columns)
        ]

        for name, t in params:
            if name in dataset_cols.columns:
                dataset_cols = dataset_cols.astype({prefix + name: t})
    else:
        dataset_cols.columns = [prefix + "samples_train"]
        dataset_cols = dataset_cols.astype({prefix + "samples_train": np.int32})

    res = df.join(dataset_cols)
    return res


def get_interpolation_measures(dataset_names, use_test_set=False, use_polynomial=False):
    """
    For the GP datasets in `dataset_names`, take the measures of certain special models.
    If `use_polynomial` is False, interpolates a piecewise linear function between
    either the training or test set (based on `use_test_set`).
    If `use_polynomial` is True, interpolates a polynomial between the training set
    (it would be pointless to do this for the test set).
    """
    res = []
    for dataset_name in tqdm.notebook.tqdm(dataset_names):
        dataset = smooth.datasets.GaussianProcessDataset.from_name(dataset_name)
        if use_polynomial:
            model = smooth.model.interpolate_polynomial(dataset)
        else:
            model = smooth.model.interpolate_relu_network(dataset, use_test_set)

        measures = smooth.measures.get_measures(
            model, dataset.x_test, dataset.y_test, include_training_measures=False,
        )
        res.append(measures)

    df = pd.DataFrame(res, index=pd.Index(dataset_names, name="dataset"))
    df["dataset"] = df.index
    df = expand_dataset_columns(df)
    return df


def make_palette(values):
    values = sorted(values)
    pal = dict(zip(values, sns.cubehelix_palette(len(values), light=0.75)))
    return pal


def get_gp_measures(datasets, from_params=False, kernel_f=None, lengthscale_coef=1.):
    """
    Compute "ground truth" measures for given GP datasets. This works by using the GP
    itself as a model. Also computes lower bounds on `path_length_f`, which can be
    computed from the outputs alone.
    """

    ms_gp_l = []
    for dataset_id in tqdm.notebook.tqdm(datasets):
        if from_params:
            dataset = smooth.datasets.from_params(**dataset_id)
        else:
            dataset = smooth.datasets.from_name(dataset_id)

        kernel = kernel_f(input_dim=dataset.x_shape()[0]) if kernel_f else None
        gp = GPy.models.GPRegression(
            dataset.x_train,
            dataset.y_train,
            noise_var=0.,
            kernel=kernel,
        )
        gp.kern.lengthscale = dataset.lengthscale * lengthscale_coef
        model = smooth.model.GPModel(gp)

        m = smooth.measures.get_measures(model, dataset)
        m.update(
            dim=dataset.dim,
            seed=dataset.seed,
            samples_train=dataset.samples_train,
            lengthscale=dataset.lengthscale,
            noise_var=int(dataset.noise_var),
            disjoint=dataset.disjoint,
            path_length_f_test_bound=smooth.measures.path_length_f_lower_bound(
                dataset, use_test_set=True,
            ),
            path_length_f_train_bound=smooth.measures.path_length_f_lower_bound(
                dataset, use_test_set=False
            ),
        )
        ms_gp_l.append(m)

    ms_gp = pd.DataFrame(ms_gp_l)
    return ms_gp


def compute_or_load_df(
    compute_f: Callable[[], pd.DataFrame], path: str, always_compute=False
):
    if os.path.isfile(path) and not always_compute:
        return pd.read_feather(path)
    else:
        res = compute_f()
        res.to_feather(path)
        return res
