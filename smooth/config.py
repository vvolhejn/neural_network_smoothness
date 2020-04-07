import itertools
import re
from typing import Dict, Any, List
from collections import defaultdict

import yaml
import numpy as np

import smooth.util


class Config:
    KEYS = [
        "name",
        "cpus",
        "gpus",
        "max_time",
        "memory_gb",
        "mail_type",
        "debug",
        "confirm",
    ]
    KEYS_SPECIAL = ["hyperparams_grid"]

    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
            self.raw_config = data

            self.hyperparams_grid = HyperparamsGrid(data["hyperparams_grid"])

            self.name = data["name"]
            self.cpus = data["cpus"]
            self.gpus = data.get("gpus", 0)
            self.max_time = data["max_time"]
            self.memory_gb = data["memory_gb"]
            self.mail_type = data["mail_type"]
            self.debug = data["debug"]
            self.confirm = data.get("confirm", True)

            for k in data:
                if k not in Config.KEYS + Config.KEYS_SPECIAL:
                    raise ValueError("Unknown key in config file: {}".format(k))

    def __repr__(self):
        return yaml.safe_dump(self.raw_config) + "\nHyperparam combinations: {}".format(
            self.hyperparams_grid.grid_size()
        )


class HyperparamsGrid:
    """
    Represents a cartesian product of dictionaries.
    """

    def __init__(self, axes):
        self.axes = []
        for axis in axes:
            self.add_axis(axis)

    def add_axis(self, axis):
        """
        Takes an axis in the format {"a": [1,2], "b": [4,5]}, makes sure
        the lists are of equal length and converts scalars/strings to lists if needed.
        """
        axis_length = None
        for key, values in axis.items():
            # Make sure the axis is a list-like (even if it's a single element).
            if isinstance(values, (list, tuple, np.ndarray)):
                values = list(values)
            else:
                values = [values]

            if axis_length is None:
                axis_length = len(values)
            elif axis_length != len(values):
                raise ValueError(
                    "Mismatch of lengths ({} vs {}) along an axis: {}".format(
                        axis_length, len(values), axis
                    )
                )
            axis[key] = values

        self.axes.append(axis)

    def get_constant_keys(self):
        res = []
        for axis in self.axes:
            for key in axis:
                if len(axis[key]) == 1:
                    res.append(key)

        return res

    def iterator(self):
        constant_keys = self.get_constant_keys()

        def with_id(d):
            nonconstant_d = {k: v for (k, v) in d.items() if k not in constant_keys}
            combination_id = smooth.util.dict_to_short_string(nonconstant_d)

            return d, combination_id

        return (
            with_id(merge_dicts(d))
            for d in itertools.product(*(zip_dicts(axis) for axis in self.axes))
        )

    def grid_size(self):
        """ The product of axis lengths. """
        return np.product([len(list(axis.values())[0]) for axis in self.axes])


def hyperparams_by_prefix(d: Dict[str, Any], prefix: str):
    """
    >>> hyperparams_by_prefix({"a.b": 1, "c": 2}, "a.")
    {'b': 1}
    """
    res = {}
    for k, v in d.items():
        if k.startswith(prefix):
            res[k[len(prefix) :]] = v

    return res


def shorten_hyperparams(d: Dict[str, Any]):
    """
    >>> sorted(shorten_hyperparams({"a.b": 1, "c": 2, "d.e": 3, "f.e": 4}).items())
    [('b', 1), ('c', 2), ('d.e', 3), ('f.e', 4)]
    """
    key_map = defaultdict(list)
    for k in d:
        k_short = k.split(".")[-1]
        key_map[k_short].append(k)

    res = {}
    for k_short, old_keys in key_map.items():
        if len(old_keys) == 1:
            res[k_short] = d[old_keys[0]]
        else:
            for k in old_keys:
                res[k] = d[k]

    return res


def zip_dicts(dicts):
    """
    >>> [sorted(x.items()) for x in zip_dicts({"a": [1, 2, 3], "b": [4, 5, 6]})]
    [[('a', 1), ('b', 4)], [('a', 2), ('b', 5)], [('a', 3), ('b', 6)]]
    """
    return [dict(zip(dicts.keys(), t)) for t in zip(*dicts.values())]


def merge_dicts(dicts):
    """
    Merges a list of dictionaries into one. If a key is present in multiple dicts,
    the value from the dict later in the list wins.
    >>> sorted(merge_dicts([{'a': 1}, {'b': 2, 'c': 3}, {'d': 4}]).items())
    [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    """
    return dict(itertools.chain(*(list(d.items()) for d in dicts)))


if __name__ == "__main__":
    cfg = Config("run_config.yaml")
    print(cfg.hyperparams_grid.axes)
    for x in cfg.hyperparams_grid.iterator():
        print(x)
    print()
    print(cfg.hyperparams_grid.axes[1])
    print([zip_dicts(axis) for axis in cfg.hyperparams_grid.axes])
