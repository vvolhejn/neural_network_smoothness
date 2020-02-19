import os


def tensorflow_init(gpu_indices):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

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
