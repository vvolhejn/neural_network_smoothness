import os

import numpy as np


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
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.old_state = np.random.get_state()
            np.random.seed(self.seed)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.seed is not None:
            np.random.set_state(self.old_state)
