"""Assignment 5 module.

Alan Szmyt
Class: CS 677
Date: April 13th, 2023
Assignment #5
Description:
This module explores and compares Naive Bayesian and Decision Tree classification for
identifying normal vs. non-normal fetus status based upon fetal cardiograms.
"""
from __future__ import annotations

from enum import IntEnum
from logging import DEBUG, Logger, StreamHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import CategoricalDtype, DataFrame

from assignment5.classifier_analytics import (
    ClassifierAnalyticsCollection,
    DecisionTreeAnalytics,
    NaiveBayesianAnalytics,
    RandomForestAnalytics,
    RandomForestAnalyticsCollection,
)
from assignment5.constants import CTG_DATASET
from assignment5.utils import DATAFRAME_NAME, data_directory, save_df_to_checkpoints

# Configure a logger to log statements.
logger: Logger = getLogger(__file__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


class CtgColumns:
    """Class container for the cardiotocography dataset columns."""
    # Column names.
    COL_LB: str = "LB"
    COL_MLTV: str = "MLTV"
    COL_WIDTH: str = "Width"
    COL_VARIANCE: str = "Variance"
    COL_NSP: str = "NSP"

    # Feature columns.
    FEATURE_COLS: list[str] = [
        COL_LB,
        COL_MLTV,
        COL_WIDTH,
        COL_VARIANCE,
    ]

    # Initial columns.
    INITIAL_COLS: list[str] = FEATURE_COLS + [COL_NSP]

    # dtype mapping to use for the Excel file.
    dtypes: dict = {
        COL_LB: np.int64,
        COL_MLTV: np.float64,
        COL_WIDTH: np.float64,
        COL_VARIANCE: np.float64,
    }


class FetusStatus(IntEnum):
    """Enum class of fetus status based on fetal cardiograms."""

    ABNORMAL: int = 0
    NORMAL: int = 1

    @staticmethod
    def dtype() -> CategoricalDtype:
        """Get the categorical dtype associated with the Fetus Status."""
        return CategoricalDtype(
            categories=[FetusStatus.ABNORMAL.value, FetusStatus.NORMAL.value]
        )


def remap_nsp(original: int) -> int:
    """Remap NSP column values to Abnormal (A) and Normal (N).

    Normal values are already 1. Everything else is remapped to 0 for Abnormal.

    Args:
        original (int): The original cell value.

    Returns:
        int: The remapped cell value.
    """
    return 0 if original != 1 else original


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

    # region Initial Setup.
    # Global Seaborn options.
    sns.set_theme(font_scale=1.5, rc={"text.usetex": True})

    # Cardiotocography dataset file from UCI.
    dataset_xls: str = "CTG.xls"
    ctg_dataset_file: Path = data_directory.joinpath(dataset_xls)
    # endregion

    # region Question 1
    # Load the cardiotocography raw data into a dataframe.
    # Group 4: LB, MLTV, Width, Variance

    # noinspection PyTypeChecker
    ctg_dataset: DataFrame = pd.read_excel(
        io=ctg_dataset_file,
        sheet_name="Raw Data",  # The actual data is in the 'Raw Data' sheet.
        usecols=CtgColumns.INITIAL_COLS,  # Only get the initial columns from the file.
        skiprows=[1],  # Skip the empty row under the header columns.
        skipfooter=3,  # Skip bottom rows that are output of Excel functions.
        na_values=["NaT"],  # Tell pandas that 'NaT' is NaN.
        engine="xlrd",  # xlrd is used to load '.xls' files.
        dtype=CtgColumns.dtypes,  # Set the initial dtypes for each column.
        converters={CtgColumns.COL_NSP: remap_nsp},  # Remap 'NSP' column.
    ).astype(
        {CtgColumns.COL_NSP: FetusStatus.dtype()}
    )  # 'NSP' column is categorical type.
    ctg_dataset.attrs[DATAFRAME_NAME] = CTG_DATASET

    save_df_to_checkpoints(ctg_dataset)
    print("Finished Question 1!")
    # endregion

    # region Question 2
    # Train naive bayesian model, make predictions, and gather analytics.
    naive_bayesian_analytics: NaiveBayesianAnalytics = NaiveBayesianAnalytics(
        dataset=ctg_dataset, predictor_col=CtgColumns.COL_NSP, persistence=True
    )
    naive_bayesian_analytics.pretty_print()

    # Compute the accuracy.
    print(f"Naive Bayesian accuracy: {naive_bayesian_analytics.accuracy.score}")

    # Compute the confusion matrix.
    naive_bayesian_analytics.show_confusion_matrix()

    print("Finished Question 2!")
    # endregion

    # region Question 3
    # Train decision tree model, make predictions, and gather analytics.
    decision_tree_analytics: DecisionTreeAnalytics = DecisionTreeAnalytics(
        dataset=ctg_dataset, predictor_col=CtgColumns.COL_NSP, persistence=True
    )
    decision_tree_analytics.pretty_print()
    decision_graph: tuple[Path, Path] = decision_tree_analytics.export()

    # Compute the accuracy.
    print(f"Decision Tree accuracy: {decision_tree_analytics.accuracy.score}")

    # Compute the confusion matrix.
    decision_tree_analytics.show_confusion_matrix()

    print("Finished Question 3!")
    # endregion

    # region Question 4
    # Train random forest model, make predictions, and gather analytics.
    random_forest_analytics: RandomForestAnalyticsCollection = (
        RandomForestAnalyticsCollection(
            analytics=[
                RandomForestAnalytics(
                    dataset=ctg_dataset,
                    predictor_col=CtgColumns.COL_NSP,
                    persistence=True,
                    trees=n,
                    max_depth=d,
                )
                for n in range(1, 11)
                for d in range(1, 6)
            ]
        )
    )
    # Error plot for all random forest classifiers.
    random_forest_analytics.show_error_plot()

    # Get the random forest with the lowest error rate.
    lowest_error_rf: RandomForestAnalytics = random_forest_analytics.lowest_error
    print(
        f"Best combination of N and d is: "
        f"N={lowest_error_rf.trees}, "
        f"d={lowest_error_rf.max_depth}"
    )

    print(
        f"The accuracy for best random forest is:" f" {lowest_error_rf.accuracy_score}"
    )

    # Compute the confusion matrix.
    lowest_error_rf.show_confusion_matrix()

    print("Finished Question 4!")
    # endregion

    # region Question 5
    # Gather analytics for all classifier models.
    classifier_analytics = ClassifierAnalyticsCollection(
        analytics=[naive_bayesian_analytics, decision_tree_analytics, lowest_error_rf]
    )

    # Generate the summary table from the analytics.
    summary_table_df: DataFrame = classifier_analytics.summary_table
    summary_table_df.attrs[DATAFRAME_NAME] = "classifier_summary_table"

    save_df_to_checkpoints(summary_table_df)

    print("Finished Question 5!")
    # endregion
