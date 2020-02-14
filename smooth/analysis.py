from typing import List

import pandas as pd
import scipy.stats


def get_kendall_coefs(
    df: pd.DataFrame,
    hyperparam_cols: List[str],
    result_col: str,
    measure_cols: List[str],
):
    """
    Given a dataframe containing the hyperparameters and measures of trained models,
    computes Kendall's tau coefficients between a measure to be predicted (result_col)
    and the measures used to predict it (measure_cols).

    Following "Fantastic Generalization Measures and Where to Find Them",
    we also compute the granulated Kendall's coefficient, meaning we split the dataset up
    into groups where only one hyperparameter changes, compute taus and then average them.
    We then define psi as the average of these coefficients.
    """

    res = pd.DataFrame(
        index=pd.Index(measure_cols, name="measures"),
        columns=pd.Index(hyperparam_cols + ["overall_tau", "psi"], name="Kendall"),
    )

    def get_kendall(df, measure):
        return scipy.stats.kendalltau(df[result_col], df[measure]).correlation

    for measure in measure_cols:
        res["overall_tau"][measure] = get_kendall(df, measure)

        psi_total = 0
        for hyperparam in hyperparam_cols:
            other_columns = hyperparam_cols.copy()
            other_columns.remove(hyperparam)
            groups = df.groupby(other_columns)
            groups = groups.apply(lambda group_df: get_kendall(group_df, measure))
            res[hyperparam][measure] = groups.mean()
            psi_total += res[hyperparam][measure]

        res["psi"][measure] = psi_total / len(hyperparam_cols)

    return res
