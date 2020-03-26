"""
Utility functions. Does not import TensorFlow (even indirectly) upon being imported.
"""

import os
import heapq
import itertools
import multiprocessing
import datetime
import re

import numpy as np
import pandas as pd
import tqdm


def tensorflow_init(gpu_indices):
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    if not gpu_indices:
        # Disable GPU usage altogether
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


class NumpyRandomSeed:
    """
    A context manager for temporarily setting a Numpy seed.
    If None is passed, the generator is not reseeded (the old RNG state is respected).
    """

    # TODO: this can be achieved more idiomatically using numpy's RandomState

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.old_state = np.random.get_state()
            np.random.seed(self.seed)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.seed is not None:
            np.random.set_state(self.old_state)


def sample_regularly(n):
    """
    Returns a permutation of range(n) whose prefixes are "evenly spaced". Specifically,
    let x = sample_regularly(n). x[0]=0, x[1]=n-1, and x[i] is chosen from the remaining
    numbers in range(n) so that it maximizes the distance from the numbers in x[:i].

    >>> list(sample_regularly(10))
    [0, 9, 4, 6, 2, 7, 1, 3, 5, 8]
    """
    assert n >= 2
    yield 0
    yield n - 1
    candidates = []
    heapq.heappush(candidates, (-(n - 1), (0, n - 1)))
    while candidates:
        _, (fr, to) = heapq.heappop(candidates)
        mid = (fr + to) // 2
        if mid == fr or mid == to:
            continue
        yield mid
        heapq.heappush(candidates, (-(mid - fr), (fr, mid)))
        heapq.heappush(candidates, (-(to - mid), (mid, to)))


def subsample_regularly(n, k):
    """
    Convenience wrapper for `sample_regularly()`.
    >>> subsample_regularly(10, 4)
    array([0, 4, 6, 9])
    """
    if k > n:
        raise ValueError(
            "trying to take more samples than available (n={}, k={})".format(n, k)
        )
    res = np.array(list(itertools.islice(sample_regularly(n), k)))
    res = np.sort(res)
    return res


def get_process_id():
    # Hacky - we're relying on an undocumented internal variable
    process_id = multiprocessing.current_process()._identity
    if process_id == ():
        # This happens when we're not running inside of a multiprocessing.Pool
        return 0
    else:
        return process_id[0]


def run_training_jobs(
    job_f, params_for_jobs, measures_path, processes, _run, shuffle=True
):
    if shuffle:
        # Shuffle to avoid having all of the "hard" hyperparameters at the end
        np.random.shuffle(params_for_jobs)

    results = []
    with multiprocessing.Pool(processes=processes) as pool:
        for res in tqdm.tqdm(
            pool.imap_unordered(job_f, params_for_jobs),
            total=len(params_for_jobs),
            smoothing=0.1,
        ):
            results.append(res)
            df = pd.DataFrame(results)
            _run.result = "{}/{} models trained".format(
                len(results), len(params_for_jobs)
            )
            print(_run.result)
            df.to_feather(measures_path)


def get_logdir_name(debug=False):
    return os.path.join(
        "logs/" if not debug else "logs_debug/",
        datetime.datetime.now().strftime("%m%d-%H%M%S"),
    )


def dict_to_short_string(d):
    """
    >>> dict_to_short_string({"foo.bar": 3, "baz": 1e-10})
    "b=1e-10_f.b=3"
    """
    return "_".join(
        "{}={}".format(re.sub(r"([^\.])[^_\.]*_?", r"\1", key), value)
        for key, value in sorted(d.items())
    )
