from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from operator import attrgetter
from pathlib import Path
from pprint import PrettyPrinter
from typing import Generic, TypeVar, Tuple
from uuid import uuid4

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from marshmallow import fields
from numpy.random import MT19937, SeedSequence
from numpy.random.mtrand import RandomState
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from assignment5.utils import (
    DATAFRAME_NAME,
    ConfusionMatrix,
    ConfusionMatrixColumns,
    MetricScore,
    Model,
    artifacts_directory,
    save_dfs_to_checkpoints,
    save_figures,
    shuffle,
    to_title, save_tree,
)

T = TypeVar("T", bound=ClassifierMixin)


class ClassifierAnalytics(Model, metaclass=ABCMeta):
    """Abstract base classifier analytics class."""

    colors: list[str] = shuffle(list(mcolors.cnames.keys()), seed=19)

    def __init__(
        self: ClassifierAnalytics,
        dataset: DataFrame,
        predictor_col: str,
        persistence: bool = False,
    ) -> None:
        """Initialize the classifier analyzer with a dataset.

        Args:
            dataset (DataFrame): The dataset for training and testing the classifier.
            predictor_col (str): The predictor column's name.
            persistence (bool): Persistence flag for saving state.
        """
        self.dataset: DataFrame = dataset
        self.persistence: bool = persistence

        # Split dataframe into X, y datasets for training/testing.
        self.predictor_col: str = predictor_col
        self.y: Series = self.dataset[self.predictor_col]
        self.x: DataFrame = dataset.drop(self.predictor_col, axis=1)

        # Using a reproducible random state across all classifiers.
        self.seed: RandomState = RandomState(MT19937(SeedSequence(42)))

        # Split the dataset into training/testing datasets split 50/50.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.5, random_state=self.seed, stratify=self.y
        )

        # Set the dataframe name for each dataset.
        self._set_name_attr()

        # Flag to check if the model has been trained.
        self.__trained__: bool = False

    @staticmethod
    def get_color(index: int) -> str:
        """Get a color using the provided index.

        Args:
            index (int): The index to use to get the color name.

        Returns:
            str: The color's name.
        """
        return ClassifierAnalytics.colors[index % len(ClassifierAnalytics.colors)]

    @property
    def schema_fields(self) -> dict:
        """Get the schema fields for JSON serialization.

        Returns:
            dict: The schema fields for serialization.
        """
        return {
            "dataset_name": fields.Str(),
            "predictor_col": fields.Str(),
            "persistence": fields.Boolean(),
            "dataset_title": fields.Str(),
            "classifier_title": fields.Str(),
        }

    @cached_property
    def dataset_name(self: ClassifierAnalytics) -> str:
        """Get the main dataset's name.

        Returns:
            str: The dataset's name.
        """
        return self.dataset.attrs.get(DATAFRAME_NAME)

    @cached_property
    def dataset_title(self: ClassifierAnalytics) -> str:
        """Get the main dataset's title.

        Returns:
            str: The dataset's title.
        """
        return to_title(self.dataset_name)

    @cached_property
    def classifier_title(self: ClassifierAnalytics) -> str:
        """Get the classifier model's title.

        Returns:
            str: The classifier's title.
        """
        return to_title(self.classifier_name)

    @cached_property
    def datasets(self: ClassifierAnalytics) -> list:
        """Get the testing and training datasets as a list.

        Returns:
            list: The testing and training datasets as a list.
        """
        return [
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ]

    def _set_name_attr(self: ClassifierAnalytics) -> None:
        """Set the name attribute for testing and training datasets."""
        # Set the dataframe's name if it's not set yet.
        if not self.dataset.attrs.get(DATAFRAME_NAME):
            self.dataset.attrs[DATAFRAME_NAME] = str(uuid4())

        # Naming each dataframe.
        self.x_train.attrs[DATAFRAME_NAME] = f"{self.dataset_name}_x_train"
        self.x_test.attrs[DATAFRAME_NAME] = f"{self.dataset_name}_x_test"
        self.y_train.attrs[DATAFRAME_NAME] = f"{self.dataset_name}_y_train"
        self.y_test.attrs[DATAFRAME_NAME] = f"{self.dataset_name}_y_test"

        if self.persistence:
            save_dfs_to_checkpoints(self.datasets)

    @cached_property
    def predictions(self: ClassifierAnalytics) -> np.array:
        """Get the classifier's predictions.

        Returns:
            np.array: The classifier's predictions.
        """
        return self.predict()

    @cached_property
    def prediction_probabilities(self: ClassifierAnalytics) -> np.array:
        """Get the classifier's prediction probabilities.

        Returns:
            np.array: The classifier's prediction probabilities.
        """
        return self.predict_probabilities()

    @cached_property
    def confusion_matrix(self: ClassifierAnalytics) -> ConfusionMatrix:
        """Get a confusion matrix from the main dataset.

        Returns:
            ConfusionMatrix: The confusion matrix from the main dataset.
        """
        confusion_matrix = ConfusionMatrix.from_dataset(self.y_test, self.predictions)

        if self.persistence:
            confusion_matrix.save(filename=f"{self.dataset_name}_confusion_matrix")

        return confusion_matrix

    @cached_property
    def confusion_matrix_plot(self: ClassifierAnalytics) -> ConfusionMatrixDisplay:
        """Get the confusion matrix display plot instance.

        Returns:
            ConfusionMatrixDisplay: The confusion matrix display plot instance.
        """
        return ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix.matrix)

    @cached_property
    def accuracy(self: ClassifierAnalytics) -> MetricScore:
        """Get the accuracy metric score of the predictions.

        Returns:
            MetricScore: The accuracy metric score of the predictions.
        """
        return self.confusion_matrix.accuracy

    @cached_property
    def accuracy_score(self: ClassifierAnalytics) -> str:
        """Get the accuracy percentage score of the predictions.

        Returns:
            str: The accuracy percentage score of the predictions.
        """
        return self.accuracy.score

    @cached_property
    def true_positives(self: ClassifierAnalytics) -> int:
        """Get the true positives count of the predictions.

        Returns:
            int: The true positives count of the predictions.
        """
        return self.confusion_matrix.true_positives

    @cached_property
    def true_negatives(self: ClassifierAnalytics) -> int:
        """Get the true negatives count of the predictions.

        Returns:
            int: The true negatives count of the predictions.
        """
        return self.confusion_matrix.true_negatives

    @cached_property
    def false_positives(self: ClassifierAnalytics) -> int:
        """Get the false positives count of the predictions.

        Returns:
            int: The false positives count of the predictions.
        """
        return self.confusion_matrix.false_positives

    @cached_property
    def false_negatives(self: ClassifierAnalytics) -> int:
        """Get the false negatives count of the predictions.

        Returns:
            int: The false negatives count of the predictions.
        """
        return self.confusion_matrix.false_negatives

    @cached_property
    def specificity(self: ClassifierAnalytics) -> MetricScore:
        """Get the specificity metric score of the predictions.

        Returns:
            MetricScore: The specificity metric score of the predictions.
        """
        return self.confusion_matrix.specificity

    @cached_property
    def recall(self: ClassifierAnalytics) -> MetricScore:
        """Get the recall metric score of the predictions.

        Returns:
            MetricScore: The recall metric score of the predictions.
        """
        return self.confusion_matrix.recall

    @cached_property
    def f1_score(self: ClassifierAnalytics) -> MetricScore:
        """Get the f1 metric score of the predictions.

        Returns:
            MetricScore: The f1 metric score of the predictions.
        """
        return self.confusion_matrix.f1_score

    @cached_property
    def error_rate(self: ClassifierAnalytics) -> MetricScore:
        """Get the error rate metric score of the predictions.

        Returns:
            MetricScore: The error rate metric score of the predictions.
        """
        return self.confusion_matrix.error

    def show_confusion_matrix(self: ClassifierAnalytics) -> None:
        """Display the confusion matrix."""
        self.confusion_matrix_plot.plot()
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.title(f"{self.dataset_title} {self.classifier_title} Confusion Matrix")

        if self.persistence:
            save_figures(filename=f"{self.dataset_name}_confusion_matrix")

        plt.show()

    @classmethod
    def __subclasshook__(cls: type[ClassifierAnalytics], subclass) -> bool:
        """Enforce that the abstract methods are implemented in derived classes."""
        return (
            hasattr(subclass, "__pretty__")
            and callable(subclass.__pretty__)
            or hasattr(subclass, "classifier")
            and callable(subclass.classifier)
            or hasattr(subclass, "classifier_name")
            and callable(subclass.classifier_name)
            or NotImplemented
        )

    @cache
    def train(self: ClassifierAnalytics) -> T:
        """Train the classifier on the training datasets.

        Returns:
            T: The trained classifier.
        """
        return self.classifier.fit(self.x_train, self.y_train)

    @cache
    def predict(self: ClassifierAnalytics) -> np.array:
        """Use the classifier instance to make predictions.

        Returns:
            np.array: The classifier's predictions.
        """
        self.__trained__ = bool(self.train())
        return self.classifier.predict(self.x_test)

    @cache
    def predict_probabilities(self: ClassifierAnalytics) -> np.array:
        """Use the classifier instance to gather prediction probabilities.

        Returns:
            np.array: The classifier's prediction probabilities.
        """
        return self.classifier.predict_proba(self.x_test)

    @property
    @abstractmethod
    def classifier(self: ClassifierAnalytics) -> T:
        """Get the classifier instance.

        Returns:
            T: The classifier associated with the analytics class.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def classifier_name(self: ClassifierAnalytics) -> str:
        """Get the classifier model's name.

        Returns:
            str: The classifier's name.
        """
        raise NotImplementedError

    def __repr__(self: ClassifierAnalytics) -> str:
        """Representation of the class instance as a string.

        Returns:
            str: The class instance's string representation.
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"\t\tclassifier={self.classifier},\n"
            f"\t\tclassifier_name={self.classifier_name},\n"
            f"\t\tpersistence={self.persistence},\n"
            f")"
        )

    @property
    def __pretty__(self: ClassifierAnalytics) -> str:
        """Pretty string of model representation.

        Returns:
            str: A pretty formatted string representation of the model class instance.
        """
        return PrettyPrinter(indent=4, depth=1).pformat(self)


C = TypeVar("C", bound=ClassifierAnalytics)


class NaiveBayesianAnalytics(ClassifierAnalytics):
    """Analytics class for Naive Bayesian classifier."""

    @cached_property
    def classifier(self: NaiveBayesianAnalytics) -> MultinomialNB:
        """Get the naive bayesian classifier instance.

        Returns:
            MultinomialNB: The MultinomialNB instance.
        """
        return MultinomialNB()

    @cached_property
    def classifier_name(self: NaiveBayesianAnalytics) -> str:
        """Get the classifier model's name.

        Returns:
            str: The classifier's name.
        """
        return "Naive Bayesian"


class DecisionTreeAnalytics(ClassifierAnalytics):
    """Analytics class for Decision Tree classifier."""

    @cached_property
    def classifier(self: DecisionTreeAnalytics) -> DecisionTreeClassifier:
        """Get the decision tree classifier instance.

        Returns:
            DecisionTreeClassifier: The DecisionTreeClassifier instance.
        """
        return DecisionTreeClassifier(random_state=self.seed)

    @cached_property
    def classifier_name(self: DecisionTreeAnalytics) -> str:
        """Get the classifier model's name.

        Returns:
            str: The classifier's name.
        """
        return "Decision Tree"

    def export(
        self: DecisionTreeAnalytics,
        output_directory: Path = artifacts_directory,
    ) -> tuple[Path, Path]:
        if not self.__trained__:
            self.predict()
        return save_tree(
            classifier=self.classifier,
            filename=f"{self.dataset_name.lower()}_{self.classifier_name.lower()}",
            output_directory=output_directory
        )


class RandomForestAnalytics(ClassifierAnalytics):
    """Analytics class for Random Forest classifier."""

    def __init__(
        self: RandomForestAnalytics,
        dataset: DataFrame,
        predictor_col: str,
        persistence: bool = False,
        trees: int = 100,
        max_depth: int | None = None,
        criterion: str = "entropy",
    ) -> None:
        """Initialize the random forest classifier analyzer with a dataset.

        Args:
            dataset (DataFrame): The dataset for training and testing the classifier.
            predictor_col (str): The predictor column's name.
            persistence (bool): Persistence flag for saving state.
            trees (int): The number of trees for the classifier to use.
            max_depth (int | None): The max depth of the random forest recursion.
            criterion (str): The criterion for the random forest classifier to use.
        """
        super().__init__(dataset, predictor_col, persistence)
        self.trees: int = trees
        self.max_depth: int | None = max_depth
        self.criterion: str = criterion

    @cached_property
    def classifier(self: RandomForestAnalytics) -> RandomForestClassifier:
        """Get the random forest classifier instance.

        Returns:
            RandomForestClassifier: The RandomForestClassifier instance.
        """
        return RandomForestClassifier(
            n_estimators=self.trees,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.seed,
        )

    @cached_property
    def classifier_name(self: RandomForestAnalytics) -> str:
        """Get the classifier model's name.

        Returns:
            str: The classifier's name.
        """
        return "Random Forest"


@dataclass
class ClassifierAnalyticsCollection(Generic[C]):
    """Classifier Analytics Collection Container class."""

    analytics: list[C]

    @property
    def lowest_error(self: ClassifierAnalyticsCollection) -> C:
        """Get the classifier with the lowest error rate.

        Returns:
            C: The classifier with the lowest error rate.
        """
        return min(self.analytics, key=attrgetter("error_rate"))

    @property
    def classifiers(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the classifier names.

        Returns:
            Series: The series of classifier names.
        """
        return Series([analytic.classifier_name for analytic in self.analytics])

    @property
    def true_positives(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the true positive counts for the classifiers.

        Returns:
            Series: The series of true positive counts.
        """
        return Series([analytic.true_positives for analytic in self.analytics])

    @property
    def true_negatives(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the true negative counts for the classifiers.

        Returns:
            Series: The series of true negative counts.
        """
        return Series([analytic.true_negatives for analytic in self.analytics])

    @property
    def false_positives(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the false positive counts for the classifiers.

        Returns:
            Series: The series of false positive counts.
        """
        return Series([analytic.false_positives for analytic in self.analytics])

    @property
    def false_negatives(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the false negative counts for the classifiers.

        Returns:
            Series: The series of false negative counts.
        """
        return Series([analytic.false_negatives for analytic in self.analytics])

    @property
    def accuracy(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the accuracy scores for the classifiers.

        Returns:
            Series: The series of accuracy scores.
        """
        return Series([analytic.accuracy for analytic in self.analytics])

    @property
    def specificity(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the specificity scores for the classifiers.

        Returns:
            Series: The series of specificity scores.
        """
        return Series([analytic.specificity for analytic in self.analytics])

    @property
    def recall(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the recall scores for the classifiers.

        Returns:
            Series: The series of recall scores.
        """
        return Series([analytic.recall for analytic in self.analytics])

    @property
    def f1_score(self: ClassifierAnalyticsCollection) -> Series:
        """Get a Series of the f1 scores for the classifiers.

        Returns:
            Series: The series of f1 scores.
        """
        return Series([analytic.f1_score for analytic in self.analytics])

    @property
    def summary_table(self: ClassifierAnalyticsCollection) -> DataFrame:
        """Get a summary table of the statistics for all classifiers.

        Returns:
            DataFrame: The summary table of statistics.
        """
        return DataFrame(
            {
                ConfusionMatrixColumns.MODEL.value: self.classifiers,
                ConfusionMatrixColumns.TRUE_POSITIVE.value: self.true_positives,
                ConfusionMatrixColumns.FALSE_POSITIVE.value: self.false_positives,
                ConfusionMatrixColumns.TRUE_NEGATIVE.value: self.true_negatives,
                ConfusionMatrixColumns.FALSE_NEGATIVE.value: self.false_negatives,
                ConfusionMatrixColumns.ACCURACY.value: self.accuracy,
                ConfusionMatrixColumns.TRUE_POSITIVE_RATE.value: self.recall,
                ConfusionMatrixColumns.TRUE_NEGATIVE_RATE.value: self.specificity,
            }
        )


@dataclass
class RandomForestAnalyticsCollection(
    ClassifierAnalyticsCollection[RandomForestAnalytics]
):
    """Collection class that contains RandomForestAnalytics instances."""

    @property
    def tree_counts(self: RandomForestAnalyticsCollection) -> set[int]:
        """The tree count values that are in the collection.

        Returns:
            set[int]: The tree count values in the collection.
        """
        return set(random_forest.trees for random_forest in self.analytics)

    @property
    def depths(self: RandomForestAnalyticsCollection) -> set[int]:
        """The max_depth values that are in the collection.

        Returns:
            set[int]: The max_depth values in the collection.
        """
        return set(random_forest.max_depth for random_forest in self.analytics)

    def analytics_at_depth(self: RandomForestAnalyticsCollection, depth: int) -> list:
        """Get the classifier analytics at a specific max depth value.

        Args:
            depth (int): The max_depth value of the random forest.

        Returns:
            list: The list of analytics that have the max_depth provided.
        """
        return list(filter(lambda a: a.max_depth == depth, self.analytics))

    def show_error_plot(self: RandomForestAnalyticsCollection) -> None:
        """Show an error rate plot for all random forests."""
        for depth in self.depths:
            plt.plot(
                list(self.tree_counts),
                [rf.error_rate for rf in self.analytics_at_depth(depth)],
                color=ClassifierAnalytics.get_color(depth),
                marker="o",
                markersize=9,
                label=f"d={depth}",
            )
        plt.legend()
        plt.xlabel("Number of trees")
        plt.ylabel("Error rate of Predictions")
        plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
        plt.title("Error rate for Random Forest Classifiers")
        plt.show()
