# For multiprocessing to work, TensorFlow must only be imported in the child processes
import multiprocessing
import datetime
import os
import itertools
import copy
import shutil
import logging

import sacred
import numpy as np
import pandas as pd

import smooth.config
import smooth.util

if __name__ == "__main__":
    try:
        _config_path = os.environ["SMOOTH_CONFIG"]
    except KeyError:
        raise RuntimeError(
            "Please set the SMOOTH_CONFIG environment variable"
            " to the path of the config YAML file."
        )

    _config = smooth.config.Config(_config_path)
    ex = sacred.Experiment(name=_config.name)

    if not _config.debug:
        observer = sacred.observers.MongoObserver(
            url="mongodb://mongochl.docker.ist.ac.at:9060", db_name="vv-smoothness"
        )
        ex.observers.append(observer)
else:
    # Needed so that doctests don't fail when they look into this module.
    ex = sacred.Experiment("dummy")


@ex.config
def experiment_config():
    dry_run = False
    log_dir = smooth.util.get_logdir_name(debug=_config.debug)
    config_path = _config_path


def train_model(hparams):
    import os
    import smooth.util

    # Sets to warning level. Disables TensorFlow's verbose messages about GPUs.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    smooth.util.tensorflow_init(gpu_indices=[])

    import smooth.datasets
    import smooth.measures
    import smooth.model

    try:
        dataset_hparams = smooth.config.hyperparams_by_prefix(hparams, "dataset.")
        dataset = smooth.datasets.from_params(**dataset_hparams)
    except np.linalg.LinAlgError as e:
        # Sometimes "SVD did not converge" can happen when creating a GP dataset
        return {"error": str(e)}

    model_hparams = smooth.config.hyperparams_by_prefix(hparams, "model.")
    model, updates = smooth.model.train_model(
        dataset=dataset, log_dir=hparams["log_dir"], **model_hparams
    )

    res = smooth.config.shorten_hyperparams(hparams)
    res.update(updates)

    measures = smooth.measures.get_measures(model, dataset)
    res.update(measures)
    # try:
    #
    # except ValueError as e:
    #     res["error"] = str(e)

    print("Finished model training of", hparams)
    print("   ", res)
    return res


@ex.automain
def main(_run, log_dir, config_path, dry_run):
    config = smooth.config.Config(config_path)
    if dry_run:
        print("Hyperparam combinations: {}".format(config.hyperparams_grid.grid_size()))
        return

    os.makedirs(log_dir)
    shutil.copy(config_path, os.path.join(log_dir, "run_config.yaml"))
    logging.info("Log dir: {}".format(log_dir))

    config.hyperparams_grid.add_axis({"log_dir": log_dir})
    hyperparams_to_try = list(config.hyperparams_grid.iterator())
    np.random.shuffle(hyperparams_to_try)

    measures_path = os.path.join(log_dir, "measures.feather")

    results = []
    with multiprocessing.Pool(processes=config.cpus) as pool:
        for res in pool.imap_unordered(train_model, hyperparams_to_try):
            results.append(res)
            df = pd.DataFrame(results)
            _run.result = "{}/{} models trained".format(
                len(results), len(hyperparams_to_try)
            )
            print(_run.result)
            df.to_feather(measures_path)

    ex.add_artifact(measures_path, content_type="application/feather")
