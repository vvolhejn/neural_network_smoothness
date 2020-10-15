# Smoothness of Functions Learned by Neural Networks

Code of my internship project conducted at IST Austria's
[Machine Learning and Computer Vision Group](http://pub.ist.ac.at/~chl/),
lead by Christoph Lampert. The internship took place between
February and April 2020.

The results of the project were published at the
[German Conference in Pattern Recognition 2020](https://www.gcpr-vmv-vcbm-2020.uni-tuebingen.de/) as _Does SGD Implicitly Optimize for Smoothness_
and also as my [Bachelor thesis](https://dspace.cuni.cz/handle/20.500.11956/119446)
at Charles University in Prague.

## GCPR abstract

Modern neural networks can easily fit their training set perfectly. Surprisingly, despite being "overfit" in this way, they tend to generalize well to future data, thereby defying the classic bias--variance trade-off of machine learning theory.
Of the many possible explanations, a prevalent one is that training by stochastic gradient descent (SGD) imposes an implicit bias that leads it to learn simple functions, and these simple functions generalize well.
However, the specifics of this implicit bias are not well understood.

In this work, we explore the _smoothness conjecture_ which states that SGD is implicitly biased towards learning functions that are smooth.
We propose several measures to formalize the intuitive notion of smoothness, and we conduct experiments to determine whether SGD indeed implicitly optimizes for these measures.
Our findings rule out the possibility that smoothness measures based on first-order derivatives are being implicitly enforced.
They are supportive, though, of the smoothness conjecture for measures based on second-order derivatives.

## Organization

The repository is organized into the following directories:

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

Some of the terms used in the code do not correspond to the ones used in the
published work.
The reason is that the code predates the published text and the names were kept
to retain backwards compatibility with the experiments.
The most significant difference is that in the code,
_function path length_ and _gradient path length_ are referred to as
`path_length_f` and `path_length_d` respectively.
