import logging
import copy

import tensorflow as tf
import smooth.measures

from tqdm.keras import TqdmCallback


class Stopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold, measure_name="mse"):
        self.threshold = threshold
        self.monitor = measure_name

    def on_epoch_end(self, epoch, logs=None):
        loss = self.get_monitor_value(logs)
        if loss < self.threshold:
            self.model.stop_training = True

    def get_monitor_value(self, logs):
        # From TF's EarlyStoppingCallback
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on measure `{}` "
                "which is not available. Available measures are: `{}`".format(
                    self.monitor, ",".join(list(logs.keys()))
                ),
            )
        return monitor_value


class Measures(tf.keras.callbacks.Callback):
    def __init__(self, dataset, samples=100):
        super().__init__()
        self.dataset = dataset
        self.samples = samples

    def on_test_end(self, logs={}):
        measure_names = [
            "gradient_norm_train",
            "gradient_norm_test",
            "weights_rms",
            "weights_product",
            "path_length_f_train",
            "path_length_f_test",
            "path_length_d_train",
            "path_length_d_test",
            "loss_train",
            "loss_test",
        ]
        history = self.model.history.history
        step = len(history["loss"]) + 1

        measures = smooth.measures.get_measures(
            self.model, self.dataset, samples=self.samples,
        )

        for k in measure_names:
            if k not in history:
                history[k] = []

            history[k].append(measures[k])
            tf.summary.scalar(k, data=measures[k], step=step)


class TensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A variant of the tensorboard callback which only logs data during validation.
    This is to prevent the storing of unnecessary amounts of data, which slows TB down.
    """

    def __init__(self, log_dir, validation_freq):
        # Warning: update_freq is in samples in TF 2.0.0, but in batches in TF 2.1
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


class Tqdm(TqdmCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


#         self.on_train_batch_begin = self.on_batch_begin
#         self.on_train_batch_end = self.on_batch_end
#         setattr(self, 'on_test_begin', lambda x: None)
#         setattr(self, 'on_test_end', lambda x: None)
#         setattr(self, 'on_test_batch_begin', lambda x, y: None)
#         setattr(self, 'on_test_batch_end', lambda x, y: None)


class WeightsHistoryCallback(tf.keras.callbacks.Callback):
    """
    Periodically save the model's weights. Dynamically alters the saving frequency
    so that the number of snapshots is in [min_snapshots, 2*min_snapshots).
    """

    def __init__(self, min_snapshots=50):
        self.weights_history = {}
        self.epochs_per_save = 1
        self.min_snapshots = min_snapshots

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epochs_per_save == 0:
            self.weights_history[epoch] = copy.deepcopy(self.model.get_weights())

        if len(self.weights_history) >= 2 * self.min_snapshots:
            self.epochs_per_save *= 2
            self.weights_history = {
                epoch: weights
                for epoch, weights in self.weights_history.items()
                if epoch % self.epochs_per_save == 0
            }
            assert len(self.weights_history) == self.min_snapshots
