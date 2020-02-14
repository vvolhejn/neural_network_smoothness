# For multiprocessing to work, TensorFlow must only be imported in the child processes
import multiprocessing
import datetime
import os
import itertools

import sacred
import numpy as np
import pandas as pd

DEBUG = False

ex = sacred.Experiment("model_comparison")

observer = sacred.observers.MongoObserver(
    url="mongodb://mongochl.docker.ist.ac.at:9060", db_name="vv-smoothness"
)
if not DEBUG:
    ex.observers.append(observer)


@ex.config
def config():
    learning_rates = [0.01, 0.003]
    init_scales = [3.0, 1.0, 0.3]
    hidden_sizes = list(np.logspace(np.log10(50), np.log10(4000), 20).astype(int))
    epochs = 20000
    batch_sizes = [128, 256, 512]
    processes = 8
    iterations = 3
    log_dir = "logs/"


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
    ):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.iteration = iteration
        self.init_scale = init_scale


def init(gpu_indices):
    import tensorflow as tf

    # I ran into this issue when using model.save():
    # https://community.paperspace.com/t/storage-and-h5py-pytables-e-g-keras-save-weights-issues-heres-why-and-how-to-solve-it/430
    # This should hopefully fix it.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Only allocates GPU memory that is necessary. Makes it easier to run multiple
        # training jobs simultaneously
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_visible_devices(
            [gpus[i] for i in gpu_indices], "GPU"
        )


def train_model(hparams: Hyperparams, verbose: int = 0):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

    process_id = multiprocessing.current_process()._identity[0]
    # print("Process id:", process_id)
    init(gpu_indices=[process_id % 3 + 1])
    # init(gpu_indices=[2])

    from smooth.datasets import mnist
    import smooth.model

    mnist = smooth.datasets.mnist

    # print("Training model", vars(hparams))
    model = smooth.model.train(
        mnist,
        learning_rate=hparams.learning_rate,
        init_scale=hparams.init_scale,
        hidden_size=hparams.hidden_size,
        epochs=hparams.epochs,
        batch_size=hparams.batch_size,
        log_dir=hparams.log_dir,
        iteration=hparams.iteration,
        verbose=verbose,
    )
    model.save(os.path.join(model.log_dir, "model.h5"))
    res = vars(hparams).copy()
    res["log_dir"] = model.log_dir
    measures = smooth.model.get_measures(model, mnist)
    res.update(measures)
    print("Done with", vars(hparams))
    print("    ", res)
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
        )
    )
    hyperparams_to_try = [Hyperparams(*l) for l in hyperparam_combinations]

    # Shuffle to avoid having all of the "hard" hyperparameters at the end
    np.random.shuffle(hyperparams_to_try)

    results = []
    with multiprocessing.Pool(processes=processes) as pool:
        for res in pool.imap_unordered(train_model, hyperparams_to_try):
            results.append(res)
            df = pd.DataFrame(results)
            _run.result = "{}/{} models trained".format(
                len(results), len(hyperparams_to_try)
            )
            print(_run.result)
            measures_path = os.path.join(log_dir, "measures.feather")
            df.to_feather(measures_path)

    ex.add_artifact(measures_path, content_type="application/feather")
