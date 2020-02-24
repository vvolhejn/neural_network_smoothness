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

ex = sacred.Experiment("model_comparison_gp")

observer = sacred.observers.MongoObserver(
    url="mongodb://mongochl.docker.ist.ac.at:9060", db_name="vv-smoothness"
)
if not DEBUG:
    ex.observers.append(observer)


@ex.config
def config():
    learning_rates = [0.01]
    init_scales = [10.0]
    hidden_sizes = [100, 1000]
    epochs = 100000
    batch_sizes = [None]
    processes = 8
    iterations = 1
    log_dir = "logs/"
    datasets = []
    for seed in range(1, 6):
        for lengthscale in [0.1, 0.3, 1.0]:
            for samples_train in np.linspace(2 / lengthscale, 10 / lengthscale, 9).astype(int):
                datasets.append("gp-{}-{}-{}".format(seed, lengthscale, samples_train))
    # datasets = "gp_123"
    loss_threshold = 1e-5
    activation = "relu"


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

    def __repr__(self):
        return str(vars(self))


def get_process_id():
    process_id = multiprocessing.current_process()._identity
    if process_id == ():
        # This happens when we're not running inside of a multiprocessing.Pool
        return 0
    else:
        return process_id[0]


def train_model(hparams: Hyperparams, verbose: int = 0):
    import os
    import smooth.util

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

    # Hacky - we're relying on an undocumented internal variable
    process_id = get_process_id()
    # smooth.util.tensorflow_init(gpu_indices=[process_id % 3 + 1])
    smooth.util.tensorflow_init(gpu_indices=[])

    import smooth.model
    import smooth.datasets

    if hparams.dataset.startswith("gp-"):
        dataset = smooth.datasets.GaussianProcessDataset.from_name(hparams.dataset)
    else:
        dataset = smooth.datasets.get_keras_image_dataset(hparams.dataset)

    kwargs = copy.deepcopy(vars(hparams))
    del kwargs["dataset"]

    if kwargs["batch_size"] is None:
        # Batch size == None -> use GD (batch size is the training set's size)
        kwargs["batch_size"] = len(dataset.x_train)

    model = smooth.model.train_shallow(
        dataset=dataset,
        verbose=verbose,
        callbacks=[],
        train_val_split=1.0,
        # batch_size=len(dataset.x_train),
        **kwargs,
    )
    model.save(os.path.join(model.log_dir, "model.h5"))
    res = vars(hparams).copy()
    res["log_dir"] = model.log_dir
    measures = smooth.measures.get_measures(
        model, dataset.x_test, dataset.y_test, samples=1000
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
        )
    )
    hyperparams_to_try = [Hyperparams(*hp) for hp in hyperparam_combinations]

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
