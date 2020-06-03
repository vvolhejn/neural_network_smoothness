# Smoothness of Functions Learned by Neural Networks

Code of my internship project at IST Austria,
which was then used as my Bachelor thesis at Charles University.
The internship took place between February and April 2020.

## Thesis abstract

Modern neural networks can easily fit their training set perfectly.
Surprisingly, they generalize well despite being "overfit" in this way,
defying the bias--variance trade-off. A prevalent explanation is that
stochastic gradient descent has an implicit bias which leads it to learn
functions which are simple, and these simple functions generalize well.
However, the specifics of this implicit bias are not well understood. In this
work, we explore the hypothesis that SGD is implicitly biased towards learning
functions which are smooth. We propose several measures to formalize the
intuitive notion of smoothness, and conduct experiments to determine whether
they are implicitly being optimized for. We exclude the possibility that
smoothness measures based on the first derivative (the gradient) are being
implicitly optimized for. Measures based on the second derivative (the
Hessian), on the other hand, show promising results.

(Thesis will be made available soon)

## Organization

The repo is organized into the following directories:

- `smooth`: Python package for running experiments
- `thesis_data`: data from experiments which appear
    in the thesis, along with configuration YAML files which can be used
    to reproduce the experiments.
- `thesis_notebooks`: simple Jupyter notebooks which calculate the reported
    results from data in `thesis_data`

The remaining directories were used during the exploratory phase of the work and have not been cleaned up. Thus, they are not meant for external use.

- `notebooks`: "dirty" Jupyter notebooks used for analyzing experimental
    results. These have not been cleaned up and it is possible that the older notebooks will not work with the code from `smooth` which has changed since the notebook's creation.
- `configs`: configuration YAML files used to run experiments
- `logs`: not included in the repository due to its size; contains all data which was collected during the experiments

## Running experiments

After installing using `pip3 install smooth`, an experiment can be run
using the `train_models_general` module:

```
SMOOTH_CONFIG=my_config.yaml python3 -m smooth.train_models_general
```

where `my_config.yaml` is the configuration file to use; see `configs` for examples.
The experiment is saved using [Sacred](https://sacred.readthedocs.io/en/stable/)
into a MongoDB database. For this reason, the environment variables `SMOOTH_DB_URL`
and `SMOOTH_DB_NAME` must be set accordingly.

## Naming

Some of the terms used in the code do not correspond to the ones used in the thesis.
The reason is that the code predates the thesis text and the names were kept
to retain backwards compatibility with the experiments.
The most significant difference is that in the code,
_function path length_ and _gradient path length_ are referred to as
`path_length_f` and `path_length_d` respectively.
