# Smoothness of Functions Learned by Neural Networks

Code of my internship project at IST Austria,
which was then used as my Bachelor thesis.
The internship took place between February and April 2020.

## Abstract

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

## Organization

The repo is organized into the following directories:

- `smooth`: code for running experiments
- `thesis_data`: data from experiments which ended up being used
    in the thesis, along with configuration YAML files which can be used
    to reproduce the experiments.
- `thesis_notebooks`: simple Jupyter notebooks which calculate the reported
    results from data in `thesis_data`

The remaining directories are not meant for external use:
- `configs`: "dirty" configuration YAML files used to run experiments
- `notebooks`: "dirty" Jupyter notebooks used for analyzing experimental
    results. These have not been cleaned up and the old ones likely won't
    work with the new codebase at all.
- `logs`: not in the repo due to its size; contains all data collected
    during the experiments

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
