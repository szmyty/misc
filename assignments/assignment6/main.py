"""Assignment 6 module.

Alan Szmyt
Class: CS 677
Date: April 21st, 2023
Assignment #6
Description:
This module explores implementing k-means clustering to create a multi-label classifier.
"""
from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, StrEnum, unique
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import CategoricalDtype, DataFrame

from assignment6.analytics import (
    AnalyticsCollection,
    ClassifierAnalytics,
    GaussianSVMModel,
    KMeansAnalytics,
    KMeansAnalyticsCollection,
    KMeansModel,
    KNNModel,
    LinearSVMModel,
    ModelAnalyzer,
    PolynomialSVMModel,
    RandomForestModel,
)
from assignment6.constants import KMEANS_CLASSIFIER_K, SEEDS_DATASET, SEEDS_SUBSET
from assignment6.utils import (
    DATAFRAME_NAME,
    data_directory,
    log,
    save_df_to_checkpoints,
    shuffle,
)

if TYPE_CHECKING:
    from pathlib import Path


class WheatClass(IntEnum):
    """Enum class of wheat classifications."""

    KAMA: int = 1
    ROSA: int = 2
    CANADIAN: int = 3

    def to_title(self: WheatClass) -> str:
        """Get the wheat class as a title string.

        Returns:
            str: The wheat class as a title string.
        """
        return self.name.title()

    @staticmethod
    def dtype() -> CategoricalDtype:
        """Get the categorical dtype associated with the wheat class.

        Returns:
            CategoricalDtype: The dtype associated with the wheat class.
        """
        return CategoricalDtype(
            categories=[
                WheatClass.KAMA,
                WheatClass.ROSA,
                WheatClass.CANADIAN,
            ]
        )

    @classmethod
    def from_label(
            cls: type[WheatClass],
            label: int,
            title: bool = False
    ) -> WheatClass | str:
        """Get a WheatClass enum from a label.

        Args:
            label (int): The label's value.
            title (bool): Whether to return the class as a title.

        Returns:
            WheatClass | str: The enum representation of the wheat class label.
        """
        wheat_class: WheatClass = cls(label)
        return wheat_class if not title else wheat_class.to_title()

    @staticmethod
    def label_to_title(label: int) -> str:
        """Convert a label from the dataset to a wheat class title.

        Args:
            label (int): The label int representation of the wheat class.

        Returns:
            str: The wheat class title.
        """
        return WheatClass.from_label(label).to_title()


@dataclass
class LabelClasses:
    """Label class choice based upon BUID."""
    negative: WheatClass
    positive: WheatClass

    @classmethod
    def select_classes(cls: type[LabelClasses], remainder: int) -> LabelClasses:
        """Select the wheat classes based upon the remainder value.

        Args:
            remainder (int): The remainder to use for selecting the class.

        Returns:
            LabelClasses: The label selected wheat classes.
        """
        match remainder:
            case 1:
                return cls(negative=WheatClass.ROSA, positive=WheatClass.CANADIAN)
            case 2:
                return cls(negative=WheatClass.KAMA, positive=WheatClass.CANADIAN)
            case _:
                return cls(negative=WheatClass.KAMA, positive=WheatClass.ROSA)

    @property
    def classes(self: LabelClasses) -> list[WheatClass]:
        """Get the wheat classes as a list.

        Returns:
            list[WheatClass]: The wheat classes as a list.
        """
        return [self.negative, self.positive]


@unique
class SeedsColumn(StrEnum):
    """Enum representation for the seeds dataset columns."""

    AREA: str = "area"
    PERIMETER: str = "perimeter"
    COMPACTNESS: str = "compactness"
    LENGTH: str = "length"
    WIDTH: str = "width"
    ASYMMETRY: str = "asymmetry"
    GROOVE_LENGTH: str = "groove"
    CLASS: str = "class"

    @classmethod
    def columns(cls: type[StrEnum]) -> list[str]:
        """Get a list of the columns for initializing a dataframe.

        Returns:
            list[str]: The list of initial columns.
        """
        return list(cls)

    @staticmethod
    def dtype() -> dict:
        """Get the dtype mapping for the columns.

        Returns:
            dict: The dtype mapping for the columns.
        """
        # noinspection PyTypeChecker
        return defaultdict(np.float64, {SeedsColumn.CLASS: WheatClass.dtype()})

    @staticmethod
    def feature_columns() -> list[SeedsColumn]:
        """Get the list of feature columns in the dataset.

        Returns:
            list[SeedsColumn]: The list of feature columns.
        """
        return [
            SeedsColumn.AREA,
            SeedsColumn.PERIMETER,
            SeedsColumn.COMPACTNESS,
            SeedsColumn.LENGTH,
            SeedsColumn.WIDTH,
            SeedsColumn.ASYMMETRY,
            SeedsColumn.GROOVE_LENGTH,
        ]

    @staticmethod
    def random_columns() -> list[SeedsColumn]:
        """Get feature columns in a random order.

        Returns:
            list[SeedsColumn]: Feature columns in a random order.
        """
        return shuffle(SeedsColumn.feature_columns(), seed=4)


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

    # region Initial Setup.
    # Global Seaborn options.
    sns.set_theme(font_scale=1.5, rc={"text.usetex": True})

    # Choosing the 2 classes based upon BUID.
    buid: str = "U38573068"
    label_classes: LabelClasses = LabelClasses.select_classes(
        remainder=int(int(buid[-1]) % 3)
    )

    # seeds dataset file from UCI.
    dataset_csv: str = "seeds_dataset.csv"
    seeds_dataset_file: Path = data_directory.joinpath(dataset_csv)
    # endregion

    # region Question 1
    # Load the whole dataset from the csv file.
    seeds_dataset: DataFrame = pd.read_csv(
        filepath_or_buffer=seeds_dataset_file,
        delimiter="\t",
        engine="python",
        header=None,
        names=SeedsColumn.columns(),
        dtype=SeedsColumn.dtype(),
    )
    seeds_dataset.attrs[DATAFRAME_NAME] = SEEDS_DATASET
    save_df_to_checkpoints(df=seeds_dataset, filename=SEEDS_DATASET)

    # Create subset of dataset using the 2 selected classes.
    seeds_subset: DataFrame = seeds_dataset[
        seeds_dataset[SeedsColumn.CLASS].isin(label_classes.classes)
    ].copy()
    seeds_subset.attrs[DATAFRAME_NAME] = SEEDS_SUBSET
    save_df_to_checkpoints(df=seeds_subset, filename=SEEDS_SUBSET)

    # Machine learning model analyzer context class.
    analyzer: ModelAnalyzer = ModelAnalyzer(
        dataset=seeds_subset,
        model=LinearSVMModel(
            predictor_col=SeedsColumn.CLASS,
            label_transformer=WheatClass.from_label
        )
    )

    # Train linear svm model, make predictions, and gather analytics.
    linear_svm_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"Linear kernel SVM accuracy: "
        f"{linear_svm_analytics.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    linear_svm_analytics.show_confusion_matrix()

    # Train gaussian svm model, make predictions, and gather analytics.
    analyzer.model = GaussianSVMModel(
        predictor_col=SeedsColumn.CLASS,
        label_transformer=WheatClass.from_label
    )
    gaussian_svm_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"Gaussian kernel SVM accuracy: "
        f"{gaussian_svm_analytics.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    gaussian_svm_analytics.show_confusion_matrix()

    # Train polynomial svm model, make predictions, and gather analytics.
    analyzer.model = PolynomialSVMModel(
        predictor_col=SeedsColumn.CLASS,
        label_transformer=WheatClass.from_label
    )
    polynomial_svm_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"Polynomial kernel SVM accuracy: "
        f"{polynomial_svm_analytics.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    polynomial_svm_analytics.show_confusion_matrix()

    log("Finished Question 1!")
    # endregion

    # region Question 2
    # Train kNN model, make predictions, and gather analytics.
    analyzer.model = KNNModel(
        predictor_col=SeedsColumn.CLASS,
        label_transformer=WheatClass.from_label
    )
    knn_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"kNN classifier accuracy: "
        f"{knn_analytics.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    knn_analytics.show_confusion_matrix()

    # Train random forest model, make predictions, and gather analytics.
    analyzer.model = RandomForestModel(
        predictor_col=SeedsColumn.CLASS,
        label_transformer=WheatClass.from_label
    )
    random_forest_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"Random forest classifier accuracy: "
        f"{random_forest_analytics.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    random_forest_analytics.show_confusion_matrix()

    # Gather analytics for all classifier models.
    classifier_analytics: AnalyticsCollection = AnalyticsCollection(
        [
            linear_svm_analytics,
            gaussian_svm_analytics,
            polynomial_svm_analytics,
            knn_analytics,
            random_forest_analytics,
        ]
    )

    # Generate the summary table from the analytics.
    summary_table_df: DataFrame = classifier_analytics.summary_table
    summary_table_df.attrs[DATAFRAME_NAME] = "classifier_summary_table"

    save_df_to_checkpoints(summary_table_df)

    log("Finished Question 2!")
    # endregion

    # region Question 3
    # Train k-means clustering model, make predictions, and gather analytics.
    analyzer.dataset = seeds_dataset
    k_means_list: list[KMeansAnalytics] = []
    for k in range(1, 8):
        analyzer.model = KMeansModel(
            predictor_col=SeedsColumn.CLASS,
            n_clusters=k,
            features=SeedsColumn.feature_columns(),
        )
        k_means_list.append(analyzer.analyze())

    k_means_analytics: KMeansAnalyticsCollection = KMeansAnalyticsCollection(
        k_means_list
    )

    # Use the 'knee' method to find the best k.
    optimal_k: int = k_means_analytics.show_knee_plot()

    # The optimal k value found by the kneedle algorithm.
    log(f"Optimal k value: {optimal_k}")

    # Get two random feature columns, f_i and f_j.
    random_features: list[str] = SeedsColumn.random_columns()[:2]

    log(f"Random features: {', '.join(random_features)}")

    # Use optimal k value for k-means clustering.
    analyzer.model = KMeansModel(
        predictor_col=SeedsColumn.CLASS,
        n_clusters=optimal_k,
        features=random_features,
        label_transformer=WheatClass.from_label
    )
    optimal_k_means_analytics: KMeansAnalytics = analyzer.analyze()

    # Plot the random features.
    optimal_k_means_analytics.plot(random_features, WheatClass.label_to_title)

    # Use all features again.
    analyzer.model = KMeansModel(
        predictor_col=SeedsColumn.CLASS,
        n_clusters=optimal_k,
        features=SeedsColumn.feature_columns(),
        label_transformer=WheatClass.from_label
    )

    # Print cluster centroid and assigned label.
    for i, cluster in enumerate(optimal_k_means_analytics.clusters):
        log(
            f"Cluster {i+1}: "
            f"Centroid={cluster.centroid}, "
            f"Label={WheatClass.from_label(cluster.majority).to_title()}"
        )

    if optimal_k <= KMEANS_CLASSIFIER_K:
        # Rerun with k=3.
        analyzer.model = KMeansModel(
            predictor_col=SeedsColumn.CLASS,
            n_clusters=3,
            features=SeedsColumn.feature_columns(),
            label_transformer=WheatClass.from_label,
        )
    elif optimal_k > KMEANS_CLASSIFIER_K:
        # Find the largest 3 clusters.
        clusters = [
            cluster.dataset
            for cluster in sorted(
                optimal_k_means_analytics.clusters,
                key=operator.attrgetter("size"),
                reverse=True,
            )[:3]
        ]

        # Run k-means with largest 3 clusters.
        analyzer.dataset = pd.concat(clusters)

    k_means_classifier_analytics: KMeansAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"K-means classifier accuracy: "
        f"{k_means_classifier_analytics.confusion_matrix.accuracy.score}"
    )

    # Use the subset dataframe again.
    analyzer.dataset = seeds_subset
    k_means_classifier_subset_analytics: KMeansAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"K-means classifier on subset accuracy: "
        f"{k_means_classifier_subset_analytics.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    k_means_classifier_subset_analytics.show_confusion_matrix()

    # Generate the summary table from the analytics.
    classifier_analytics.append(k_means_classifier_subset_analytics)
    final_summary_df: DataFrame = classifier_analytics.summary_table
    final_summary_df.attrs[DATAFRAME_NAME] = "final_classifier_summary_table"

    save_df_to_checkpoints(final_summary_df)

    log("Finished Question 3!")
    # endregion
