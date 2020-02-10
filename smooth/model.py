import datetime

import numpy as np
import tensorflow as tf

assert tf.__version__[0] == '2'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling

from smooth.datasets import ClassificationDataset

# from tqdm.keras import TqdmCallback

import matplotlib.pyplot as plt


def get_average_l2(model):
    total_l2 = 0
    n_weights = 0
    for weight_matrix in model.get_weights():
        total_l2 += np.sum(weight_matrix ** 2)
        n_weights += weight_matrix.size

    return total_l2 / n_weights


def get_roughness(model, x_input):
    with tf.GradientTape() as g:
        x = tf.constant(x_input)
        g.watch(x)
        y = model(x)
    dy_dx = g.batch_jacobian(y, x)
    # We consider each function x -> prob. that x is a "1" separately,
    # take the Frobenius norms of the Jacobians and sum them
    # Should we perhaps take the norm of the entire 10x28x28 tensor?
    res = np.mean(np.linalg.norm(dy_dx, ord='fro', axis=(2, 3)))
    return res


class MetricsCallback(keras.callbacks.Callback):

    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.max_roughness_samples = 1000

    def on_test_end(self, logs={}):
        history = self.model.history.history

        for k in ["val_roughness", "val_l2"]:
            if k not in history:
                history[k] = []

        roughness = get_roughness(self.model, self.x_val[:self.max_roughness_samples])
        l2 = get_average_l2(self.model)
        history["val_roughness"].append(roughness)
        history["val_l2"].append(l2)
        step = len(history["loss"]) + 1
        tf.summary.scalar("val_roughness", data=roughness, step=step)
        tf.summary.scalar("val_l2", data=l2, step=step)


#         predict = np.asarray(self.model.predict(self.validation_data[0]))
#         targ = self.validation_data[1]
#         self.f1s=f1(targ, predict)

# class TqdmCallbackFixed(TqdmCallback):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.on_train_batch_begin = self.on_batch_begin
#         self.on_train_batch_end = self.on_batch_end
#         setattr(self, 'on_test_begin', lambda x: None)
#         setattr(self, 'on_test_end', lambda x: None)
#         setattr(self, 'on_test_batch_begin', lambda x, y: None)
#         setattr(self, 'on_test_batch_end', lambda x, y: None)

def split_dataset(x, y, first_part=0.9):
    split = int(len(x) * first_part)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    return x_train, y_train, x_val, y_val


def train(dataset: ClassificationDataset, learning_rate, init_scale, hidden_size=200, epochs=1000, batch_size=512):
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=dataset.x_train[0].shape))
    model.add(layers.Dense(hidden_size, activation='relu',
                           kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_out"),
                           bias_initializer=VarianceScaling(scale=init_scale, mode="fan_out"),
                           ))
    model.add(layers.Dense(dataset.n_classes,
                           kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
                           bias_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
                           activation='softmax',
                           ))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_train, y_train, x_val, y_val = split_dataset(dataset.x_train, dataset.y_train, 0.9)

    #     n_frames = 100
    #     history_callback = HistoryCallback(epochs_per_save=int(epochs/n_frames))
    metrics_cb = MetricsCallback(x_val, y_val)

    # tqdm_cb = TqdmCallbackFixed(verbose=0)
    validation_freq = 50
    model.validation_freq = validation_freq

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    model.fit(x_train, y_train,
              epochs=epochs,
              shuffle=True,
              batch_size=batch_size,
              verbose=1,
              callbacks=[  # tqdm_cb,
                  metrics_cb,
                  tf.keras.callbacks.EarlyStopping("loss", min_delta=1e-5, patience=500),
                  tensorboard_callback
              ],
              validation_data=(x_val, y_val),
              validation_freq=validation_freq,  # evaluate once every <validation_freq> epochs
              )

    model.plot_title = ("lr={}, init_scale={:.2f}, ".format(learning_rate, init_scale)
                        + "h={}, ep={}".format(hidden_size, epochs))

    return model
