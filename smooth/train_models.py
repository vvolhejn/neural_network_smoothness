# For multiprocessing to work, TensorFlow must only be imported in the child processes
import multiprocessing
import datetime
import os

import sacred
import numpy as np
import pandas as pd

ex = sacred.Experiment("model_comparison")

observer = sacred.observers.MongoObserver(
    url="mongodb://mongochl.docker.ist.ac.at:9060", db_name="vv-smoothness"
)
ex.observers.append(observer)


@ex.config
def config():
    learning_rates = [0.01]
    hidden_sizes = list(np.logspace(np.log10(10), np.log10(2000), 50).astype(int))
    epochs = 40000
    batch_size = 1000
    processes = 6
    iterations = 5
    log_dir = "logs/"


class Hyperparams:
    def __init__(
        self, learning_rate, hidden_size, epochs, batch_size, log_dir, iteration
    ):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.iteration = iteration


def train_model(hparams: Hyperparams):
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

    from smooth.datasets import mnist
    import smooth.model
    import smooth.main

    smooth.main.init(gpu_indices=[2, 3])

    mnist = smooth.datasets.mnist

    print("Training model", vars(hparams))
    model = smooth.model.train(
        mnist,
        learning_rate=hparams.learning_rate,
        init_scale=1.0,
        hidden_size=hparams.hidden_size,
        epochs=hparams.epochs,
        batch_size=hparams.batch_size,
        log_dir=hparams.log_dir,
        iteration=hparams.iteration,
    )
    model.save(os.path.join(hparams.log_dir, model.id, "model.h5"))
    res = dict(learning_rate=hparams.learning_rate, hidden_size=hparams.hidden_size,)
    metrics = smooth.model.get_metrics(model, mnist)
    res.update(metrics)
    print("Done with", vars(hparams))
    print("    ", res)
    return res


@ex.automain
def main(
    _run,
    learning_rates,
    hidden_sizes,
    epochs,
    batch_size,
    processes,
    iterations,
    log_dir,
):
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%m%d-%H%M%S"))

    hyperparams_to_try = []
    for iteration in range(iterations):
        for learning_rate in learning_rates:
            for hidden_size in hidden_sizes:
                hyperparams_to_try.append(
                    Hyperparams(
                        learning_rate,
                        hidden_size,
                        epochs,
                        batch_size,
                        log_dir,
                        iteration,
                    )
                )

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
            metrics_path = os.path.join(log_dir, "metrics.feather")
            df.to_feather(metrics_path)

    ex.add_artifact(metrics_path, content_type="application/feather")
