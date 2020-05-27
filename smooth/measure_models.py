"""
Obsolete. Used to re-measure previously trained models.
"""

import multiprocessing
import os
import sys
import itertools

import pandas as pd
import tqdm
import click

import smooth.util


def measure_saved_model(args):
    import tensorflow as tf
    import smooth.datasets
    import smooth.model
    import smooth.measures

    model_path, gpu_indices = args

    # Hacky - we're relying on an undocumented internal variable
    process_id = multiprocessing.current_process()._identity[0]
    smooth.util.tensorflow_init([gpu_indices[process_id % len(gpu_indices)]])

    # TODO: un-hardcode
    mnist = smooth.datasets.get_mnist()
    model = tf.keras.models.load_model(model_path)
    measures = smooth.measures.get_measures(
        model,
        x=mnist.x_test,
        y=mnist.y_test,
        include_training_measures=False,
        is_classification=True,
    )
    measures["model_path"] = model_path
    return measures


def find_models(log_dir):
    model_filename = "model.h5"

    res = []
    for entry in os.scandir(log_dir):
        if entry.is_dir():
            model_path = os.path.join(entry.path, model_filename)
            if os.path.isfile(model_path):
                res.append(model_path)

    return res


@click.command()
@click.argument("log-dir")
@click.argument("output-file")
@click.option(
    "--processes", type=int, help="How many processes to run in parallel", default=1
)
@click.option(
    "--gpu",
    "-g",
    "gpus",
    type=int,
    help="Use the GPU with this index (use multiple times to use multiple GPUs)",
    multiple=True,
)
def measure_models(log_dir, output_file, processes, gpus):
    models = find_models(log_dir)
    if not models:
        sys.exit("No models found under {}".format(log_dir))

    if os.path.isfile(output_file):
        sys.exit("Output file already exists: {}".format(output_file))

    if not output_file.endswith(".feather"):
        sys.exit("The output file should be a .feather")

    # models = models[:3]

    args = zip(models, itertools.repeat(gpus))

    results = []
    with multiprocessing.Pool(processes=processes) as pool:
        for res in tqdm.tqdm(
            pool.imap_unordered(measure_saved_model, args), total=len(models)
        ):
            results.append(res)
            # print(res)

            df = pd.DataFrame(results)
            df.to_feather(output_file)


if __name__ == "__main__":
    measure_models()
