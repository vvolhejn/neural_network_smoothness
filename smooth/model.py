import re
import os

import numpy as np
import tensorflow as tf

assert tf.__version__[0] == "2"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling

from smooth.datasets import ClassificationDataset
from smooth import measures

# from tqdm.keras import TqdmCallback


def get_metrics(model, dataset, max_gradient_norm_samples=1000):
    gradient_norm = measures.gradient_norm(model, dataset.x_test[:max_gradient_norm_samples])
    l2 = measures.average_l2(model)
    history = model.history.history
    return dict(
        gradient_norm=gradient_norm,
        l2=l2,
        loss=history["loss"][-1],
        accuracy=history["accuracy"][-1],
        val_loss=history.get("val_loss", [None])[-1],
        val_accuracy=history.get("val_accuracy", [None])[-1],
    )


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.max_gradient_norm_samples = 1000

    def on_test_end(self, logs={}):
        history = self.model.history.history

        for k in ["gradient_norm", "l2"]:
            if k not in history:
                history[k] = []

        gradient_norm = measures.gradient_norm(self.model, self.x_val[: self.max_gradient_norm_samples])
        l2 = measures.average_l2(self.model)
        history["gradient_norm"].append(gradient_norm)
        history["l2"].append(l2)
        step = len(history["loss"]) + 1
        tf.summary.scalar("gradient_norm", data=gradient_norm, step=step)
        tf.summary.scalar("l2", data=l2, step=step)


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


def get_model_id(**kwargs):
    return "{}".format(
        # "exp0211-1",
        # datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "_".join(
            (
                "{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                for key, value in sorted(kwargs.items())
            )
        )
    )


def train(
    dataset: ClassificationDataset,
    learning_rate,
    init_scale,
    log_dir,
    hidden_size=200,
    epochs=1000,
    batch_size=512,
    iteration=None,
):
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=dataset.x_train[0].shape))
    model.add(
        layers.Dense(
            hidden_size,
            activation="relu",
            kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_out"),
            bias_initializer=VarianceScaling(scale=init_scale, mode="fan_out"),
        )
    )
    model.add(
        layers.Dense(
            dataset.n_classes,
            kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
            bias_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
            activation="softmax",
        )
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x_train, y_train, x_val, y_val = split_dataset(
        dataset.x_train, dataset.y_train, 0.9
    )

    #     n_frames = 100
    #     history_callback = HistoryCallback(epochs_per_save=int(epochs/n_frames))
    metrics_cb = MetricsCallback(x_val, y_val)

    # How many times do we want to validate and call and the Tensorboard callback
    n_updates = 100
    validation_freq = int(epochs / n_updates)

    model.validation_freq = validation_freq
    model.id = get_model_id(
        learning_rate=learning_rate,
        init_scale=init_scale,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        iteration=iteration,
    )

    log_dir = os.path.join(log_dir, model.id)
    # Warning: update_freq is in samples in TF 2.0.0, but in batches in TF 2.1
    tb_update_freq = update_freq = int((len(dataset.x_train) * epochs) / n_updates)
    # print("TB update freq:", tb_update_freq)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        profile_batch=0,
        write_graph=False,
        histogram_freq=validation_freq,
        update_freq=int(1e18),  # "batch"
    )
    tensorboard_callback._on_epoch_end = tensorboard_callback.on_epoch_end

    def tb_on_epoch_end(epoch, logs=None):
        if logs is not None and "val_loss" in logs:
            tensorboard_callback._on_epoch_end(epoch, logs)

    tensorboard_callback.on_epoch_end = tb_on_epoch_end

    file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    file_writer.set_as_default()

    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(LOG_DIR, model_id, "checkpoints"),
    #                                                          save_weights_only=False,
    #                                                          verbose=1)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        verbose=0,
        callbacks=[  # tqdm_cb,
            metrics_cb,
            tf.keras.callbacks.EarlyStopping("loss", min_delta=1e-5, patience=500),
            tensorboard_callback,
            # checkpoint_callback,
        ],
        validation_data=(x_val, y_val),
        validation_freq=validation_freq,  # evaluate once every <validation_freq> epochs
    )

    model.plot_title = "lr={}, init_scale={:.2f}, ".format(
        learning_rate, init_scale
    ) + "h={}, ep={}".format(hidden_size, epochs)

    return model
