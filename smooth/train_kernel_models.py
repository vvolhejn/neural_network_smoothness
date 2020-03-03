# For multiprocessing to work, TensorFlow must only be imported in the child processes
import multiprocessing
import datetime
import os
import itertools
import copy
from typing import List

import sacred
import numpy as np
import pandas as pd
import sklearn.kernel_ridge

import smooth.util

DEBUG = bool(os.environ.get("SMOOTH_DEBUG"))

ex = sacred.Experiment("gp_kernels")

if not DEBUG:
    observer = sacred.observers.MongoObserver(
        url="mongodb://mongochl.docker.ist.ac.at:9060", db_name="vv-smoothness"
    )
    ex.observers.append(observer)


@ex.config
def config():
    log_dir = smooth.util.get_logdir_name(debug=DEBUG)
    dry_run = False
    processes = 8
    datasets = [
        "gp-{}-{}-{}-{}".format(dim, seed, lengthscale * float(dim), samples_train)
        for (dim, seed, lengthscale, samples_train) in itertools.product(
            [2 ** x for x in range(1, 10)],
            range(1, 4),
            [0.3, 1.0],
            np.logspace(np.log10(10), np.log10(1000), 10).round().astype(int),
        )
    ]
    # datasets = ["gp-{}-{}-{}-{}".format(4, 123, 1.0, 100)]
    # alphas = [0.01, 0.0001, 1e-18]
    alphas = [0]
    gammas = [1.0]
    degrees = [1, 2, 3, 4, 5]
    continue_from = None


class Hyperparams:
    def __init__(
        self, dataset: str, degree: int, alpha: float, gamma: float,
    ):
        self.degree = degree
        self.dataset = dataset
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        return str(vars(self))


class SklearnModel:
    """
    A class that wraps a scikit-learn model and pretends it's a Keras model. This is
    useful for evaluating our measures.
    """

    def __init__(self, clf):
        self.clf = clf

    def predict(self, x, batch_size=None):
        # batch_size is a fake argument which is ignored
        return self.clf.predict(x)


def measure_krr(krr: sklearn.kernel_ridge.KernelRidge, dataset):
    import smooth.measures

    train_loss = sklearn.metrics.mean_squared_error(
        krr.predict(dataset.x_train), dataset.y_train
    )
    test_loss = sklearn.metrics.mean_squared_error(
        krr.predict(dataset.x_test), dataset.y_test
    )
    model = SklearnModel(krr)
    path_length_f_train = smooth.measures.path_length(model, dataset.x_train)
    path_length_f_test = smooth.measures.path_length(model, dataset.x_test)

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "path_length_f_train": path_length_f_train,
        "path_length_f_test": path_length_f_test,
    }


def train_model(hparams: Hyperparams, verbose: int = 0, dataset_cache={}):
    import os
    import smooth.util

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    smooth.util.tensorflow_init(gpu_indices=[])

    import smooth.datasets
    import smooth.measures

    try:
        dataset = smooth.datasets.from_name(hparams.dataset)
    except np.linalg.LinAlgError as e:
        # Sometimes "SVD did not converge" can happen when creating a GP dataset
        return {"error": str(e)}

    kwargs = copy.deepcopy(vars(hparams))
    del kwargs["dataset"]

    krr = sklearn.kernel_ridge.KernelRidge(kernel="poly", coef0=1, **kwargs)
    krr.fit(dataset.x_train, dataset.y_train)
    res = vars(hparams).copy()
    measures = measure_krr(krr, dataset)
    res.update(measures)
    print("Finished model training of", hparams)
    print("   ", res)
    return res


def train_models(hparams_list: List[Hyperparams], verbose: int = 0):
    dataset_cache = {}
    dataset_names = set(
        "-".join(hparams.dataset.split("-")[:-1]) for hparams in hparams_list
    )
    for d in dataset_names:
        d2 = d + "-1000"
        dataset_cache[d] = smooth.datasets.from_name(d2)

    res = []
    for hparams in hparams_list:
        res.append(train_model(hparams, dataset_cache=dataset_cache))
    return res


@ex.automain
def main(
    _run, processes, log_dir, datasets, dry_run, alphas, gammas, degrees, continue_from
):
    if DEBUG:
        log_dir = "logs_debug/"

    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%m%d-%H%M%S"))

    hyperparam_combinations = list(itertools.product(datasets, degrees, alphas, gammas))
    hyperparams_to_try = [Hyperparams(*hp) for hp in hyperparam_combinations]

    if continue_from is not None:
        ms = pd.read_feather(continue_from)
        filtered = []
        for hp in hyperparams_to_try:
            cur = ms
            for k, v in vars(hp).items():
                cur = cur.loc[cur[k] == v]

            if cur.empty:
                filtered.append(hp)

        if len(ms) + len(filtered) != len(hyperparams_to_try):
            raise ValueError(
                "Could not filter out some hyperparams using {}.".format(continue_from)
                + " Are the hyperparams compatible?"
            )
        print("Before: {}. After: {}".format(len(hyperparams_to_try), len(filtered)))
        hyperparams_to_try = filtered

    if dry_run:
        print("Models to train:", len(hyperparams_to_try))
        return

    measures_path = os.path.join(log_dir, "measures.feather")
    os.makedirs(log_dir)
    smooth.util.run_training_jobs(
        train_model, hyperparams_to_try, measures_path, processes, _run
    )
    ex.add_artifact(measures_path, content_type="application/feather")

    # # Shuffle to avoid having all of the "hard" hyperparameters at the end
    # np.random.shuffle(hyperparams_to_try)
    #
    # results = []
    # with multiprocessing.Pool(processes=processes) as pool:
    #     for res in pool.imap_unordered(train_models, hyperparams_to_try):
    #         results.append(res)
    #         df = pd.DataFrame(results)
    #         _run.result = "{}/{} models trained".format(
    #             len(results), len(hyperparams_to_try)
    #         )
    #         print(_run.result)
    #         df.to_feather(measures_path)
