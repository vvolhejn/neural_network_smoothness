# For multiprocessing to work, TensorFlow must only be imported in the child processes
import multiprocessing
import datetime
import os
import itertools
import copy

import sacred
import numpy as np
import pandas as pd

DEBUG = False

ex = sacred.Experiment("gp_nd_increasing_measures")

if not DEBUG:
    observer = sacred.observers.MongoObserver(
        url="mongodb://mongochl.docker.ist.ac.at:9060", db_name="vv-smoothness"
    )
    ex.observers.append(observer)


@ex.config
def config():
    GP = True

    log_dir = "logs/"
    activation = "relu"
    dry_run = False

    if GP:
        processes = 20
        learning_rates = [0.1, 0.01, 0.001]
        init_scales = [10.0, 1.0, 0.1]
        # hidden_sizes = [1, 2, 4, 8, 16, 32]
        hidden_sizes = [4, 16, 64]
        epochs = 100000
        batch_sizes = [64]

        iterations = 1
        datasets = [
            "gp-{}-{}-{}-{}-0-{}".format(dim, seed, lengthscale * float(dim), samples_train, disjoint)
            for (dim, seed, lengthscale, samples_train, disjoint) in itertools.product(
                [2 ** x for x in range(1, 10, 2)],
                range(1, 2),
                [1.0],
                # [0.3, 1.0],
                np.logspace(np.log10(10), np.log10(1000), 10).round().astype(int),
                range(2),
            )
        ]
        loss_threshold = 1e-5
        use_gpu = False
        train_val_split = 1.0
    else:
        processes = 9
        learning_rates = [0.01]
        init_scales = [1.0]
        # hidden_sizes = [10, 30, 100, 300]
        hidden_sizes = [1, 2, 4, 8, 16, 32]
        epochs = 20000
        batch_sizes = [128]

        iterations = 3

        datasets = [
            "mnist-{}".format(n_samples)
            for n_samples in np.logspace(np.log10(60), np.log10(60000), 20)
                .round()
                .astype(int)
        ]

        loss_threshold = 0.01
        use_gpu = True
        train_val_split = 0.9


class Hyperparams:
    def __init__(
        self,
        learning_rate,
        hidden_size,
        epochs,
        batch_size,
        log_dir,
        iteration,
        init_scale,
        dataset: str,
        loss_threshold: float,
        activation: str,
        use_gpu: bool,
        train_val_split: float,
    ):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.iteration = iteration
        self.init_scale = init_scale
        self.dataset = dataset
        self.loss_threshold = loss_threshold
        self.activation = activation
        self.use_gpu = use_gpu
        self.train_val_split = train_val_split

    def __repr__(self):
        return str(vars(self))


def train_model(hparams: Hyperparams, verbose: int = 0):
    import os
    import smooth.util

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

    process_id = smooth.util.get_process_id()
    smooth.util.tensorflow_init(
        gpu_indices=[process_id % 3 + 1] if hparams.use_gpu else []
    )

    import smooth.model
    import smooth.datasets
    import smooth.measures
    import tensorflow as tf

    try:
        dataset = smooth.datasets.from_name(hparams.dataset)
    except np.linalg.LinAlgError as e:
        # Sometimes "SVD did not converge" can happen when creating a GP dataset
        return {"error": str(e)}

    kwargs = copy.deepcopy(vars(hparams))
    non_training_hparams = ["dataset", "use_gpu"]
    for p in non_training_hparams:
        del kwargs[p]

    if kwargs["batch_size"] is None:
        # Batch size == None -> use GD (batch size is the training set's size)
        kwargs["batch_size"] = len(dataset.x_train)

    model = smooth.model.train_shallow(
        dataset=dataset,
        verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5000,
                                             min_delta=1e-5)
        ],
        **kwargs,
    )
    model.save(os.path.join(model.log_dir, "model.h5"))
    res = vars(hparams).copy()

    res["log_dir"] = model.log_dir
    measures = smooth.measures.get_measures(
        model, dataset, samples=1000
    )
    res.update(measures)
    print("Finished model training of", model.id)
    print("   ", res)
    return res


@ex.automain
def main(
    _run,
    learning_rates,
    hidden_sizes,
    epochs,
    batch_sizes,
    init_scales,
    processes,
    iterations,
    log_dir,
    datasets,
    loss_threshold,
    activation,
    dry_run,
    use_gpu,
    train_val_split,
):
    if DEBUG:
        log_dir = "logs_debug/"

    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%m%d-%H%M%S"))

    hyperparam_combinations = list(
        itertools.product(
            learning_rates,
            hidden_sizes,
            [epochs],
            batch_sizes,
            [log_dir],
            range(iterations),
            init_scales,
            datasets,
            [loss_threshold],
            [activation],
            [use_gpu],
            [train_val_split]
        )
    )
    hyperparams_to_try = [Hyperparams(*hp) for hp in hyperparam_combinations]
    if dry_run:
        print("Models to train:", len(hyperparams_to_try))
        return

    # Shuffle to avoid having all of the "hard" hyperparameters at the end
    np.random.shuffle(hyperparams_to_try)

    measures_path = os.path.join(log_dir, "measures.feather")
    results = []
    with multiprocessing.Pool(processes=processes) as pool:
        for res in pool.imap_unordered(train_model, hyperparams_to_try):
            results.append(res)
            df = pd.DataFrame(results)
            _run.result = "{}/{} models trained".format(
                len(results), len(hyperparams_to_try)
            )
            print(_run.result)
            df.to_feather(measures_path)

    ex.add_artifact(measures_path, content_type="application/feather")
