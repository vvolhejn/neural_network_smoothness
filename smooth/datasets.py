import numpy as np
import tensorflow as tf


class ClassificationDataset:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        assert np.min(y_train) >= 0
        # assert y_train.dtype == int
        self.n_classes = np.max(y_train) + 1

    def subset(self, n_samples: int):
        """
        Selects the first n_samples of the training set, and an appropriately scaled number of samples
        of the test set.
        The data is not copied, but returned as a view.
        """
        assert n_samples <= len(self.x_train)
        n_samples_test = int(n_samples / len(self.x_train) * len(self.x_test))
        return ClassificationDataset(
            x_train=self.x_train[:n_samples],
            y_train=self.y_train[:n_samples],
            x_test=self.x_test[:n_samples_test],
            y_test=self.y_test[:n_samples_test],
        )

    def add_label_noise(self, p: float):
        """
        With probability p for each label, the label is changed to a random incorrect label.
        """

        def random_wrong(correct):
            """ Random ints in [0, n_classes) of shape correct.shape, where the original values are avoided """
            res = np.random.randint(self.n_classes - 1, size=correct.shape)
            res = res + (res >= correct).astype(int)
            assert not np.any(res == correct)
            return res

        def add_label_noise_to_one(y):
            # Could have been implemented in a simpler way by adjusting p to account for the possibility
            # that we generate the correct label "accidentally"
            y_wrong = random_wrong(y)
            mask = np.random.random(size=y.shape) < p
            return np.where(mask, y_wrong, y)

        return ClassificationDataset(
            x_train=self.x_train,
            y_train=add_label_noise_to_one(self.y_train),
            x_test=self.x_test,
            y_test=self.y_test,
        )


def _get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return ClassificationDataset(x_train, y_train, x_test, y_test)


mnist = _get_mnist()


def get_mnist_variant(n_samples, label_noise):
    rng_state = np.random.get_state()
    np.random.seed(8212)
    mnist2 = mnist.subset(n_samples).add_label_noise(label_noise)
    np.random.set_state(rng_state)
    return mnist2
