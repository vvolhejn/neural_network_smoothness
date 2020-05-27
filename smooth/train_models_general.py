"""
Trains multiple models in parallel using multiprocessing.
Data about the experiment is saved using Sacred.
"""

# For multiprocessing to work, TensorFlow must only be imported in the child processes,
# which is why we don't import it here
import multiprocessing
import os
import shutil
import logging

import sacred
import numpy as np
import pandas as pd

import smooth.config
import smooth.util


def confirmation_prompt(config: smooth.config.Config):
    print(config)
    ans = input("Begin training? (y/n): ")
    if ans not in ["y", "yes"]:
        print("Cancelling.")
        exit(1)


if __name__ == "__main__":
    try:
        _config_path = os.environ["SMOOTH_CONFIG"]
        _db_url = os.environ["SMOOTH_DB_URL"]
        _db_name = os.environ["SMOOTH_DB_NAME"]
    except KeyError:
        raise RuntimeError(
            "Please set the following environment variables:\n"
            "SMOOTH_CONFIG: path of the config YAML file\n"
            "SMOOTH_DB_URL: URL of the MongoDB database for Sacred\n"
            "SMOOTH_DB_NAME: name of the MongoDB database for Sacred"
        )

    _config = smooth.config.Config(_config_path)
    if _config.confirm:
        confirmation_prompt(_config)

    ex = sacred.Experiment(name=_config.name)

    if not _config.debug:
        observer = sacred.observers.MongoObserver(
            url=_db_url,
            db_name=_db_name,
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


def train_model(args):
    """
    Trains a single model with hyperparameters specified by a dictionary.
    Ran inside a process pool.
    """
    import os
    import smooth.util

    # Sets to warning level. Disables TensorFlow's verbose messages about GPUs.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if _config.gpus > 0:
        process_id = smooth.util.get_process_id()

        smooth.util.tensorflow_init(gpu_indices=[process_id % _config.gpus])
    else:
        smooth.util.tensorflow_init(gpu_indices=[])

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    import smooth.datasets
    import smooth.measures
    import smooth.model

    hparams, combination_id = args
    hparams["model.model_id"] = combination_id

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

    res = hparams
    res.update(updates)

    measures = smooth.measures.get_measures(model, dataset)
    res.update(measures)

    print("Finished model training:", res)
    return res


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@ex.automain
def main(_run, log_dir, config_path, dry_run):
    config = smooth.config.Config(config_path)
    if dry_run:
        print("Hyperparam combinations: {}".format(config.hyperparams_grid.grid_size()))
        return

    if os.path.abspath(log_dir) != os.getcwd():
        os.makedirs(log_dir)
        shutil.copy(config_path, os.path.join(log_dir, "run_config.yaml"))

    logging.info("Log dir: {}".format(os.path.abspath(log_dir)))

    config.hyperparams_grid.add_axis({"log_dir": log_dir})
    hyperparams_to_try = list(config.hyperparams_grid.iterator())
    np.random.shuffle(hyperparams_to_try)

    measures_path = os.path.join(log_dir, "measures.feather")

    results = []

    # I was getting OOM errors when training a large number of models.
    # Maybe restarting the processes once in a while could help.
    CHUNK_SIZE = 4 * config.cpus
    chunks = chunk_list(hyperparams_to_try, CHUNK_SIZE)

    for hyperparams_chunk in chunks:
        with multiprocessing.Pool(processes=config.cpus) as pool:
            for res in pool.imap_unordered(train_model, hyperparams_chunk):
                results.append(res)
                df = pd.DataFrame(results)
                _run.result = "{}/{} models trained".format(
                    len(results), len(hyperparams_to_try)
                )
                print(_run.result)
                df.to_feather(measures_path)

    ex.add_artifact(measures_path, content_type="application/feather")
