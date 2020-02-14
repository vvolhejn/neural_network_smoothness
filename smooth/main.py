import os

import numpy as np
import tensorflow as tf

import smooth.model
from smooth.datasets import mnist


def smoothness_experiment_1():
    skip = 0
    for learning_rate in [0.03, 0.01, 0.003, 0.001]:
        for init_scale in [3.0, 1.0, 0.3, 0.1]:
            for hidden_size in [100, 300, 1000, 3000, 10000]:
                if skip > 0:
                    skip -= 1
                    continue
                model = smooth.model.train(
                    mnist,
                    learning_rate=learning_rate,
                    init_scale=init_scale,
                    hidden_size=hidden_size,
                    epochs=10000,
                    batch_size=2048,
                )
                model.save(os.path.join(smooth.model.LOG_DIR, model.id, "model.h5"))
                print("Done with {}".format(model.id))


def init(gpu_indices):
    # I ran into this issue when using model.save():
    # https://community.paperspace.com/t/storage-and-h5py-pytables-e-g-keras-save-weights-issues-heres-why-and-how-to-solve-it/430
    # This should hopefully fix it.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(
            [gpus[i] for i in gpu_indices], "GPU"
        )


def main():
    init(gpu_indices=[2])

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == "__main__":
    main()
