import numpy as np
import tensorflow as tf

import smooth.model
from smooth.datasets import mnist

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices([gpus[2], gpus[3]], 'GPU')

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    model = smooth.model.train(mnist, learning_rate=0.01, init_scale=1., epochs=6000, batch_size=2048)

if __name__ == '__main__':
    main()
