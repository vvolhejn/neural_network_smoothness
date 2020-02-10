import os

import numpy as np
import tensorflow as tf

import smooth.model
from smooth.datasets import mnist


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices([gpus[2], gpus[3]], 'GPU')

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    for learning_rate in [0.03, 0.01, 0.003, 0.001]:
        for init_scale in [3., 1., 0.3, 0.1]:
            for hidden_size in [100, 300, 1000, 3000, 10000]:
                model = smooth.model.train(mnist, learning_rate=learning_rate, init_scale=init_scale,
                                           hidden_size=hidden_size, epochs=10000, batch_size=2048)
                model.save(os.path.join(smooth.model.LOG_DIR, model.id, "model.h5"))
                print("Done with {}".format(model.id))


#     model = smooth.model.train(mnist, learning_rate=0.01, init_scale=1., hidden_size=10000, epochs=200, batch_size=2048)

# def train(dataset: ClassificationDataset, learning_rate, init_scale, hidden_size=200, epochs=1000, batch_size=512):

if __name__ == '__main__':
    main()
