import logging

import tensorflow as tf
from tensorflow import keras

from smooth import measures

# from tqdm.keras import TqdmCallback

class Stopping(keras.callbacks.Callback):
    def __init__(self, loss_threshold):
        self.loss_threshold = loss_threshold
        self.monitor = "loss"

    def on_epoch_end(self, epoch, logs=None):
        loss = self.get_monitor_value(logs)
        if loss < self.loss_threshold:
            self.model.stop_training = True

    def get_monitor_value(self, logs):
        # From TF's EarlyStoppingCallback
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on measure `%s` "
                "which is not available. Available measures are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value


class Measures(keras.callbacks.Callback):
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

        gradient_norm = measures.gradient_norm(
            self.model, self.x_val[: self.max_gradient_norm_samples]
        )
        l2 = measures.average_l2(self.model)
        history["gradient_norm"].append(gradient_norm)
        history["l2"].append(l2)
        step = len(history["loss"]) + 1
        tf.summary.scalar("gradient_norm", data=gradient_norm, step=step)
        tf.summary.scalar("l2", data=l2, step=step)


class TensorBoard(keras.callbacks.TensorBoard):
    """
    A variant of the tensorboard callback which only logs data during validation.
    This is to prevent the storing of unnecessary amounts of data, which slows TB down.
    """

    def __init__(self, log_dir, validation_freq):
        super().__init__(
            log_dir=log_dir,
            profile_batch=0,
            write_graph=False,
            histogram_freq=validation_freq,
            update_freq=int(1e18),
        )

    def on_epoch_end(self, epoch, logs={}):
        if logs is not None and "val_loss" in logs:
            super().on_epoch_end(epoch, logs)

# class Tqdm(TqdmCallback):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.on_train_batch_begin = self.on_batch_begin
#         self.on_train_batch_end = self.on_batch_end
#         setattr(self, 'on_test_begin', lambda x: None)
#         setattr(self, 'on_test_end', lambda x: None)
#         setattr(self, 'on_test_batch_begin', lambda x, y: None)
#         setattr(self, 'on_test_batch_end', lambda x, y: None)
