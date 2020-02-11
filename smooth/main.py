import os

import numpy as np
import tensorflow as tf

import smooth.model
from smooth.datasets import mnist


def smoothness_experiment_1():
    skip = 0
    for learning_rate in [0.03, 0.01, 0.003, 0.001]:
        for init_scale in [3., 1., 0.3, 0.1]:
            for hidden_size in [100, 300, 1000, 3000, 10000]:
                if skip > 0:
                    skip -= 1
                    continue
                model = smooth.model.train(mnist, learning_rate=learning_rate, init_scale=init_scale,
                                           hidden_size=hidden_size, epochs=10000, batch_size=2048)
                model.save(os.path.join(smooth.model.LOG_DIR, model.id, "model.h5"))
                print("Done with {}".format(model.id))


def mnist_double_descent():
    np.random.seed(8212)
    # 4000 samples following "Reconciling..." and label noise following "Deep Double Descent"
    small_noisy_mnist = mnist.subset(4000).add_label_noise(0.1)

    # for learning_rate in [0.03, 0.01, 0.003, 0.001]:
    #     for init_scale in [3., 1., 0.3, 0.1]:
    for learning_rate in [0.01, 0.003, 0.001]:
        for hidden_size in range(100, 3100, 100):
            model = smooth.model.train(small_noisy_mnist, learning_rate=learning_rate, init_scale=1.,
                                       hidden_size=hidden_size, epochs=20000, batch_size=2000)
            model.save(os.path.join(smooth.model.LOG_DIR, model.id, "model.h5"))
            print("Done with {}".format(model.id))


def main():
    # I ran into this issue when using model.save():
    # https://community.paperspace.com/t/storage-and-h5py-pytables-e-g-keras-save-weights-issues-heres-why-and-how-to-solve-it/430
    # This should hopefully fix it.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices([gpus[2]], 'GPU')

    mnist_double_descent()
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#     model = smooth.model.train(mnist, learning_rate=0.01, init_scale=1., hidden_size=10000, epochs=200, batch_size=2048)

# def train(dataset: ClassificationDataset, learning_rate, init_scale, hidden_size=200, epochs=1000, batch_size=512):

if __name__ == '__main__':
    main()
