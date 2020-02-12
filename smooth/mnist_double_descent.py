import datetime
import os

import sacred
import pymongo
import numpy as np
import pandas as pd

from smooth.datasets import mnist
import smooth.model
import smooth.main

ex = sacred.Experiment("mnist_double_descent")

observer = sacred.observers.MongoObserver(
    url='mongodb://mongochl.docker.ist.ac.at:9060',
    db_name="vv-smoothness"
)
ex.observers.append(observer)


@ex.config
def config():
    learning_rates = [0.01, 0.003]
    hidden_sizes = list(np.logspace(np.log10(100), np.log10(10000), 20).astype(int))
    epochs = 50000


@ex.capture
def get_model_info(model, dataset):
    return smooth.model.get_metrics(model, dataset)


@ex.automain
def main(learning_rates, hidden_sizes, epochs):
    smooth.main.init(gpu_indices=[2])

    np.random.seed(8212)
    # 4000 samples following "Reconciling..." and label noise following "Deep Double Descent"
    small_noisy_mnist = mnist.subset(4000).add_label_noise(0.1)

    log_dir = os.path.join(
        smooth.model.LOG_DIR,
        datetime.datetime.now().strftime("%m%d-%H%M%S"),
    )

    results = []
    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes:
            model = smooth.model.train(
                small_noisy_mnist,
                learning_rate=learning_rate,
                init_scale=1.,
                hidden_size=hidden_size,
                epochs=epochs,
                batch_size=2000,
                log_dir=log_dir,
            )
            # model.save(os.path.join(log_dir, model.id, "model.h5"))
            res = dict(
                learning_rate=learning_rate,
                hidden_size=hidden_size,
            )
            metrics = smooth.model.get_metrics(model, small_noisy_mnist)
            res.update(metrics)

            results.append(res)
            df = pd.DataFrame(results)
            metrics_path = os.path.join(log_dir, "metrics.feather")
            df.to_feather(metrics_path)

            print("Done with {}".format(model.id))
            print("    ", res)

    ex.add_artifact(metrics_path, content_type="application/feather")