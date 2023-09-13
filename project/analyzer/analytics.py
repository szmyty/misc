"""Analytics module.

Alan Szmyt
Class: CS 677
Date: April 21st, 2023
Assignment 6
Description:
Machine learning model analytics for training and testing on datasets and
gathering analytics from the results.
"""
from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, unique
from operator import attrgetter
from pprint import PrettyPrinter
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import uuid4

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seaborn.axisgrid import PairGrid
from kneed import KneeLocator
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as DrawPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.legend import Legend
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from numpy.random import MT19937, SeedSequence
from numpy.random.mtrand import RandomState
from pandas import DataFrame, Series
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    euclidean_distances,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from project.utils import (
    DATAFRAME_NAME,
    DataModel,
    artifacts_directory,
    float_to_percentage,
    plot_correlation_matrix,
    save_array,
    save_dfs_to_checkpoints,
    save_figures,
    save_tree,
    shuffle,
    to_title,
    wrap_labels,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def radar_factory(num_vars: int, frame: str = "circle") -> np.array:
    """Factory method to create a radar plot.

    Args:
        num_vars (int): The number of variables being plotted.
        frame (str): The type of frame to have for the plot.

    References:
        - https://www.python-graph-gallery.com/390-basic-radar-chart
        - https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    Returns:
        np.array: The theta angles for the radar plot.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self: RadarTransform, path: DrawPath):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            # noinspection PyProtectedMember,Ruff
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return DrawPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self: RadarAxes, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self: RadarAxes, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self: RadarAxes, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self: RadarAxes, line):
            x, y = line.get_data()

            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self: RadarAxes, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self: RadarAxes):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self: RadarAxes):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=DrawPath.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


@dataclass
class LinearPlotData:
    """Data class with data for plotting the linear model results."""

    x_label: str
    y_label: str
    x_scatter: object
    y_scatter: object
    x_regression: object
    y_regression: object


@dataclass
class Point2D:
    """Point coordinates data model."""

    x: float
    y: float


@dataclass
class Cluster:
    """Cluster data model for a K-means cluster."""

    label: int | str
    dataset: DataFrame
    init_centroid: np.array
    category_col: str
    centroid: np.array = field(init=False)
    size: int = field(init=False)
    majority: Any = field(init=False)

    def __post_init__(self: Cluster) -> None:
        """Initialize the cluster size and majority after init."""
        self.size = len(self.dataset.index)
        self.majority = self.dataset[self.category_col].mode().iloc[0].item()
        self.centroid = self.dataset.drop(self.category_col, axis=1).mean().to_numpy()


@dataclass
class SplitDataset:
    """Data model representation of test and train split datasets."""

    x_train: DataFrame
    x_test: DataFrame
    y_train: DataFrame
    y_test: DataFrame


@dataclass(unsafe_hash=True)
class MetricScore(float):
    """Data class container for a metric score."""

    value: float

    @property
    def score(self: MetricScore) -> str:
        """Get metric score as a percentage.

        Returns:
            str: The metric score as a percentage.
        """
        return float_to_percentage(self.value)


@unique
class ConfusionMatrixColumns(StrEnum):
    """Enum for confusion matrix columns."""

    MODEL: str = "Model"
    TRUE_POSITIVE: str = "TP"
    FALSE_POSITIVE: str = "FP"
    TRUE_NEGATIVE: str = "TN"
    FALSE_NEGATIVE: str = "FN"
    TRUE_POSITIVE_RATE: str = "TPR"
    TRUE_NEGATIVE_RATE: str = "TNR"
    ACCURACY: str = "Accuracy"
    F1_SCORE: str = "F1 Score"

    @staticmethod
    def scores() -> list[ConfusionMatrixColumns]:
        """Get columns that are associated with score metrics.

        Returns:
            list[ConfusionMatrixColumns]: The columns that are associated with score
                metrics.
        """
        return [
            ConfusionMatrixColumns.ACCURACY,
            ConfusionMatrixColumns.F1_SCORE,
        ]


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion matrix data class."""

    matrix: np.array

    @property
    def raw(self: ConfusionMatrix) -> np.array:
        """Get the raw numpy array confusion matrix data.

        Returns:
            np.array: The raw confusion matrix data.
        """
        return self.matrix.ravel()

    @property
    def true_negatives(self: ConfusionMatrix) -> int:
        """Get the true negatives from the confusion matrix.

        Returns:
            int: The true negatives.
        """
        return self.raw[0]

    @property
    def false_positives(self: ConfusionMatrix) -> int:
        """Get the false positives from the confusion matrix.

        Returns:
            int: The false positives.
        """
        return self.raw[1]

    @property
    def false_negatives(self: ConfusionMatrix) -> int:
        """Get the false negatives from the confusion matrix.

        Returns:
            int: The false negatives.
        """
        return self.raw[2]

    @property
    def true_positives(self: ConfusionMatrix) -> int:
        """Get the true positives from the confusion matrix.

        Returns:
            int: The true positives.
        """
        return self.raw[3]

    @property
    def positives(self: ConfusionMatrix) -> int:
        """Get all positive values.

        Returns:
            int: The positive values.
        """
        return self.true_positives + self.false_negatives

    @property
    def negatives(self: ConfusionMatrix) -> int:
        """Get all negative values.

        Returns:
            int: The negative values.
        """
        return self.true_negatives + self.false_positives

    @property
    def total(self: ConfusionMatrix) -> int:
        """Get the total count of values from the confusion matrix.

        Returns:
            int: The total count of values.
        """
        return self.positives + self.negatives

    @property
    def accurate(self: ConfusionMatrix) -> int:
        """Get the count of accurate values from the confusion matrix.

        Returns:
            int: The count of accurate values.
        """
        return self.true_positives + self.true_negatives

    @property
    def inaccurate(self: ConfusionMatrix) -> int:
        """Get the count of inaccurate values from the confusion matrix.

        Returns:
            int: The count of inaccurate values.
        """
        return self.false_positives + self.false_negatives

    @property
    def recall(self: ConfusionMatrix) -> MetricScore:
        """Get recall score from the confusion matrix.

        Also known as sensitivity or true positive rate (TPR).

        Returns:
            MetricScore: The recall score.
        """
        return MetricScore(self.true_positives / self.positives)

    @property
    def specificity(self: ConfusionMatrix) -> MetricScore:
        """Get specificity score from the confusion matrix.

        Also known as true negative rate (TNR).

        Returns:
            MetricScore: The specificity score.
        """
        return MetricScore(self.true_negatives / self.negatives)

    @property
    def precision(self: ConfusionMatrix) -> MetricScore:
        """Get the precision from the confusion matrix.

        References:
            - https://tinyurl.com/5f5pa39j

        Returns:
            MetricScore: The precision score.
        """
        return MetricScore(
            self.true_positives / (self.true_positives + self.false_positives)
        )

    @property
    def false_positive_rate(self: ConfusionMatrix) -> MetricScore:
        """Get the false positive rate from the confusion matrix.

        Returns:
            MetricScore: The false positive rate.
        """
        return MetricScore(self.false_positives / (1 - self.specificity))

    @property
    def accuracy(self: ConfusionMatrix) -> MetricScore:
        """Get the accuracy from the confusion matrix.

        Returns:
            MetricScore: The accuracy.
        """
        return MetricScore(self.accurate / self.total)

    @property
    def error_rate(self: ConfusionMatrix) -> MetricScore:
        """Get the error rate from the confusion matrix.

        Returns:
            MetricScore: The error rate.
        """
        return MetricScore(self.inaccurate / self.total)

    @property
    def f1_score(self: ConfusionMatrix) -> MetricScore:
        """Get the F1 score from the confusion matrix.

        Returns:
            MetricScore: The F1 score.
        """
        return MetricScore(
            2 * ((self.precision * self.recall) / (self.precision + self.recall))
        )

    @property
    def mcc(self: ConfusionMatrix) -> MetricScore:
        """Get Matthew's correlation coefficient from confusion matrix.

        Returns:
            MetricScore: Matthew's correlation coefficient.
        """
        return MetricScore(
            (
                (self.true_positives * self.true_negatives)
                - (self.false_positives * self.false_negatives)
            )
            / (
                math.sqrt(
                    (self.true_positives + self.false_positives)
                    * (self.true_positives + self.false_negatives)
                    * (self.true_negatives + self.false_positives)
                    * (self.true_negatives + self.false_negatives)
                )
            )
        )

    def save(self: ConfusionMatrix, filename: str | None = None) -> Path:
        """Save the confusion matrix to a file.

        Args:
            filename (str | None): The filename to save the confusion matrix to.

        Returns:
            Path: The file path of the saved confusion matrix.
        """
        # If the filename is not provided, save using a UUID.
        if not filename:
            filename = str(uuid4())

        return save_array(arr=self.matrix, filename=filename)

    @classmethod
    def from_dataset(
        cls: type[ConfusionMatrix], y_true: np.array, y_pred: np.array
    ) -> ConfusionMatrix:
        """Create a confusion matrix from a dataset.

        Args:
            y_true (np.array): The true values.
            y_pred (np.array): The prediction values.

        Returns:
            ConfusionMatrix: The confusion matrix of the dataset.
        """
        return cls(metrics.confusion_matrix(y_true, y_pred))


@dataclass(kw_only=True)
class Analytics:
    """Base analytics data model."""

    classes: Any
    model_name: str
    dataset_name: str
    dataset: DataFrame
    predictions: np.array
    truth_values: np.array
    persistence: bool = field(default=False)


@dataclass(kw_only=True)
class RegressionAnalytics(Analytics):
    """Analytics data model for regressions."""

    mse: float = field(init=False)
    rmse: float = field(init=False)
    sse: float = field(init=False)
    mae: float = field(init=False)
    r2_metric: float = field(init=False)

    def __post_init__(self: ClassifierAnalytics) -> None:
        """Compute regression metrics."""
        self.mse = mean_squared_error(self.truth_values, self.predictions)
        self.rmse = mean_squared_error(
            self.truth_values, self.predictions, squared=False
        )
        self.mae = mean_absolute_error(self.truth_values, self.predictions)
        self.sse = np.sum(np.square(self.predictions - self.truth_values))
        self.r2_metric = r2_score(self.truth_values, self.predictions)


@dataclass(kw_only=True)
class ClassifierAnalytics(Analytics):
    """Analytics data model for classifiers."""

    probabilities: np.array | None = field(default=None)
    confusion_matrix: ConfusionMatrix = field(init=False)
    confusion_matrix_plot: ConfusionMatrixDisplay = field(init=False)

    def __post_init__(self: ClassifierAnalytics) -> None:
        """Initialize confusion matrix based upon other analytics."""
        # The computed confusion matrix based upon the predictions.
        self.confusion_matrix: ConfusionMatrix = ConfusionMatrix.from_dataset(
            self.truth_values, self.predictions
        )

        # The confusion matrix display plot instance.
        self.confusion_matrix_plot: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
            confusion_matrix=self.confusion_matrix.matrix,
            display_labels=self.classes,
        )

    @property
    def error_rate(self: ClassifierAnalytics) -> float:
        """Get the error rate from the confusion matrix.

        Returns:
            float: The error rate.
        """
        return self.confusion_matrix.error_rate

    def show_confusion_matrix(self: ClassifierAnalytics) -> None:
        """Display the confusion matrix."""
        self.confusion_matrix_plot.plot(cmap="plasma")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.title(
            f"{to_title(self.dataset_name)} "
            f"{to_title(self.model_name)} Confusion Matrix"
        )

        if self.persistence:
            save_figures(filename=f"{self.dataset_name}_confusion_matrix")

        plt.show()

    @staticmethod
    def get_color(index: int) -> str:
        """Get a color using the provided index.

        Args:
            index (int): The index to use to get the color name.

        Returns:
            str: The color's name.
        """
        return ModelAnalyzer.colors[index % len(ModelAnalyzer.colors)]


# noinspection PyDataclass
@dataclass(kw_only=True)
class RandomForestAnalytics(ClassifierAnalytics):
    """Analytics data model for Random Forest Model."""

    trees: int
    max_depth: int | None


# noinspection PyDataclass
@dataclass(kw_only=True)
class LinearModelAnalytics(RegressionAnalytics):
    """Analytics data model for linear models."""

    coefficients: np.array
    intercept: np.array
    residuals: np.array
    plot_data: LinearPlotData

    def plot(self: LinearModelAnalytics, title: str) -> None:
        """Plot the linear model predicted values and actual values."""
        # Plot the x_test data in scatter plot.
        plt.scatter(
            self.plot_data.x_scatter, self.plot_data.y_scatter, color="dodgerblue"
        )

        # Plot the regression model predictions.
        plt.plot(self.plot_data.x_regression, self.plot_data.y_regression, color="red")
        plt.xlabel(self.plot_data.x_label)
        plt.ylabel(self.plot_data.y_label)
        plt.gcf().subplots_adjust(bottom=0.19, left=0.22)

        plt.title(title)

        # Save the plot to a file.
        save_figures("_".join(title.lower().split()))

        plt.show()


# noinspection PyDataclass
@dataclass(kw_only=True)
class KMeansAnalytics(ClassifierAnalytics):
    """Analytics data model for K-Means Model."""

    category_col: str
    labels: np.array
    clusters: list[Cluster]
    centroids: np.array
    inertia: MetricScore
    n_clusters: int

    def plot(
        self: KMeansAnalytics, features: list[str], label_transform: Callable
    ) -> None:
        """Plot the K-means clusters.

        Args:
            features (list[str]): The features to use for x and y axes.
            label_transform (Callable): Transformer method for the label.
        """
        # The two features to plot on x and y.
        feature_1: str = features[0]
        feature_2: str = features[1]

        # Set axis labels.
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)

        for cluster in self.clusters:
            plt.scatter(
                x=cluster.dataset[feature_1],
                y=cluster.dataset[feature_2],
                label=label_transform(cluster.majority),
            )

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker="x", color="r")
        plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
        plt.legend()
        plt.show()


ModelAnalytics = TypeVar("ModelAnalytics", bound=ClassifierAnalytics)


class ModelAnalyzer:
    """Analyze ML model performance."""

    colors: list[str] = shuffle(list(mcolors.cnames.keys()), seed=19)

    def __init__(
        self: ModelAnalyzer,
        dataset: DataFrame,
        model: MachineLearningModel,
        category_col: str,
    ) -> None:
        """Initialize the analyzer with a dataset and ML model.

        Args:
            dataset (DataFrame): The dataset to use for the analysis.
            model (MachineLearningModel): The ML model to analyze.
            category_col (str): The category column in the dataset.
        """
        self._dataset = dataset
        self._model = model
        self._category_col = category_col

    @property
    def dataset(self: ModelAnalyzer) -> DataFrame:
        """Get the dataset to analyze with the ML model.

        Returns:
            DataFrame: The dataset to analyze with the ML model.
        """
        return self._dataset

    @dataset.setter
    def dataset(self: ModelAnalyzer, dataset: DataFrame) -> None:
        self._dataset = dataset

    @property
    def model(self: ModelAnalyzer) -> MachineLearningModel:
        """Get the ML model to analyze.

        Returns:
            MachineLearningModel: The ML model to analyze.
        """
        return self._model

    @model.setter
    def model(self: ModelAnalyzer, model: MachineLearningModel) -> None:
        self._model = model

    @property
    def dataset_name(self: ModelAnalyzer) -> str:
        """Get the dataset's name.

        Returns:
            str: The dataset's name.
        """
        return self.dataset.attrs.get(DATAFRAME_NAME, str(uuid4()))

    def analyze(self: ModelAnalyzer) -> ModelAnalytics:
        """Analyze the performance of the ML model.

        Returns:
            ModelAnalytics: The analytics result.
        """
        return self._model.analyze_model(self.dataset)

    @staticmethod
    def create_pairwise_plot(
            dataset: DataFrame,
            category_col: str,
            title: str,
            label_transformer: Callable[[Any, bool], Any] | None = None,
    ) -> None:
        if label_transformer:
            dataset = dataset.copy()
            dataset[category_col] = dataset[category_col].map(
                label_transformer
            )

        # Create, save, and plot a pairwise plot of the all bills.
        dataset_pairwise: PairGrid = sns.pairplot(
            dataset,
            hue=str(category_col)
        )
        dataset_pairwise.fig.suptitle(title)

        plt.gcf().subplots_adjust(top=0.85)

        # Save the plots to a file.
        save_figures("_".join(title.lower().split()))

        plt.show()

    def pairwise_plot(self: ModelAnalyzer) -> list[Path]:
        """Plot and save a pairwise plot of the dataset.

        Returns:
            list[Path]: The list of file paths for the saved pairwise plot.
        """
        # Create, save, and plot a pairwise plot of the all bills.
        dataset_pairwise = sns.pairplot(
            self.dataset,
            hue=str(self._category_col),
        )
        dataset_pairwise.fig.suptitle(f"{to_title(self.dataset_name)} Pairwise Plot")

        paths: list[Path] = save_figures(f"{self.dataset_name}_pairwise_plot")
        plt.show()
        return paths

    @staticmethod
    def create_correlation_matrix(dataset: DataFrame, title: str) -> DataFrame:
        """Create a correlation matrix.

        Args:
            dataset (DataFrame): The dataset to create a correlation matrix from.
            title (str): The title for the correlation matrix plot.

        Returns:
            DataFrame: The correlation matrix.
        """
        return plot_correlation_matrix(
            df=dataset,
            title=title,
        )

    def correlation_matrix(self: ModelAnalyzer, drop: bool = False) -> DataFrame:
        """Get and plot the correlation matrix for the dataset.

        Args:
            drop (bool): Whether to drop category column for the graph.

        Returns:
            DataFrame: The correlation matrix dataframe.
        """
        return plot_correlation_matrix(
            df=self.dataset if drop else self.dataset.drop(self._category_col, axis=1),
            title=f"{to_title(self.dataset_name)} Correlation Matrix",
        )

    def radar_plot(
        self: ModelAnalyzer,
        title: str,
        label_transformer: Callable[[Any, bool], Any] | None = None,
    ) -> list[Path] | None:
        """Plot and save a radar plot of the dataset.

        Args:
            title (str): The title of the radar plot.
            label_transformer (Callable[[Any, bool], Any] | None): The label transform
                method to use.

        Returns:
            list[Path] | None: The list of file paths for the saved radar plot.
        """
        # Scale the dataset to be values in between 0 and 1.
        scaled_values: np.array = MinMaxScaler().fit_transform(self.dataset)
        scaled_dataset: DataFrame = self.dataset.copy()
        scaled_dataset.loc[:, :] = scaled_values

        # Get a list of the features in the dataset.
        features: np.array = scaled_dataset.drop(
            self._category_col, axis=1
        ).columns.to_numpy()
        feature_count: int = len(features)

        # Get the theta angle value from the radar factory.
        theta = radar_factory(feature_count, frame="polygon")

        fig, axs = plt.subplots(
            figsize=(9, 9), nrows=1, ncols=2, subplot_kw={"projection": "radar"}
        )
        fig.subplots_adjust(wspace=0.50, hspace=0.20, top=0.95, bottom=0.05)

        for ax, (category, df) in zip(
            axs, scaled_dataset.groupby(self._category_col), strict=True
        ):
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

            ax.set_title(
                label_transformer(category, True),
                weight="bold",
                size="medium",
                position=(0.5, 1.1),
                horizontalalignment="center",
                verticalalignment="center",
            )

            values = df.drop(self._category_col, axis=1).mean(axis=0)

            ax.plot(theta, values, color="b")
            ax.fill(theta, values, facecolor="b", alpha=0.25, label="_nolegend_")

            ax.set_varlabels(features)

        fig.text(
            0.5,
            0.965,
            title,
            horizontalalignment="center",
            color="black",
            weight="bold",
            size="large",
        )
        plt.tight_layout()
        paths: list[Path] = save_figures(f"{self.dataset_name}_radar_plot")
        plt.show()
        return


class MachineLearningModel(DataModel, metaclass=ABCMeta):
    """Machine learning model class for interfacing with ML models in scikit."""

    # Reproducible random state across all classifiers.
    seed: RandomState = RandomState(MT19937(SeedSequence(42)))

    def __init__(
        self: MachineLearningModel,
        category_col: str,
        estimator: Pipeline,
        persistence: bool,
        label_transformer: Callable[[Any], Any] | None = None,
        x_transform_func: Callable[Series | DataFrame] | None = None,
        y_transform_func: Callable[Series | DataFrame] | None = None,
    ) -> None:
        """Initialize the ML model.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
            x_transform_func (Callable[Series | DataFrame] | None): The function to
                transform x column.
            y_transform_func (Callable[Series | DataFrame] | None): The function to
                transform y column.
        """
        self.category_col: str = category_col
        self.estimator: Pipeline = estimator
        self.persistence: bool = persistence
        self.label_transformer: Callable[[Any, Any], Any] | None = label_transformer
        self.x_transform_func = x_transform_func
        self.y_transform_func = y_transform_func

    @abstractmethod
    def analyze_model(
        self: MachineLearningModel, dataset: DataFrame
    ) -> ClassifierAnalytics:
        """Analyze the performance of the ML model.

        Returns:
            ClassifierAnalytics: The analytics result.
        """

    @property
    def classes(self: MachineLearningModel) -> np.array | None:
        """Get the ML model's class labels.

        Returns:
            Any: The ML model's class labels.
        """
        try:
            return (
                self.estimator.classes_
                if self.label_transformer is None
                else [
                    self.label_transformer(label, True)
                    for label in self.estimator.classes_
                ]
            )
        except AttributeError:
            return None

    @property
    @abstractmethod
    def name(self: MachineLearningModel) -> str:
        """Get the machine learning model's name.

        Returns:
            str: The machine learning model's name.
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(
        cls: type[MachineLearningModel], subclass: type[MachineLearningModel]
    ) -> bool:
        """Enforce that the abstract methods are implemented in derived classes."""
        return (
            hasattr(subclass, "model")
            and callable(subclass.model)
            or hasattr(subclass, "name")
            and callable(subclass.name)
            or NotImplemented
        )

    @property
    def schema_fields(self: MachineLearningModel) -> dict:
        """Get the schema fields for JSON serialization.

        Returns:
            dict: The schema fields for serialization.
        """
        return {}

    @property
    def __pretty__(self: MachineLearningModel) -> str:
        """Pretty string of model representation.

        Returns:
            str: A pretty formatted string representation of the model class instance.
        """
        return PrettyPrinter(indent=4, depth=1).pformat(self)


class ClassifierModel(MachineLearningModel, metaclass=ABCMeta):
    """Machine learning model that is used for classification."""

    def analyze_model(
        self: ClassifierModel,
        dataset: DataFrame,
    ) -> ClassifierAnalytics:
        """Analyze the machine learning model against a dataset.

        Args:
            dataset (DataFrame): The dataset to use for the analysis.

        Returns:
            ClassifierAnalytics: The analytics results after training and testing.
        """
        # Transformer function to apply to the 'X' dataset.
        x_transformer: FunctionTransformer = FunctionTransformer(self.x_transform_func)

        # Transformer function to apply to the 'y' dataset.
        y_transformer: FunctionTransformer = FunctionTransformer(self.y_transform_func)

        # Split dataframe into X, y datasets for training/testing.
        y: Series = y_transformer.transform(dataset[self.category_col])
        x: DataFrame = x_transformer.transform(dataset.drop(self.category_col, axis=1))

        # Split the dataset into training/testing datasets split 50/50.
        split_dataset: SplitDataset = SplitDataset(
            *train_test_split(x, y, test_size=0.5, random_state=self.seed, stratify=y)
        )

        # Train the model with the training datasets.
        self.estimator.fit(split_dataset.x_train, split_dataset.y_train)

        dataset_name: str = dataset.attrs.get(DATAFRAME_NAME, str(uuid4()))

        # Naming each dataframe.
        split_dataset.x_train.attrs[DATAFRAME_NAME] = f"{dataset_name}_x_train"
        split_dataset.x_test.attrs[DATAFRAME_NAME] = f"{dataset_name}_x_test"
        split_dataset.y_train.attrs[DATAFRAME_NAME] = f"{dataset_name}_y_train"
        split_dataset.y_test.attrs[DATAFRAME_NAME] = f"{dataset_name}_y_test"

        if self.persistence:
            save_dfs_to_checkpoints(
                [
                    split_dataset.x_train,
                    split_dataset.x_test,
                    split_dataset.y_train,
                    split_dataset.y_test,
                ]
            )

        return ClassifierAnalytics(
            classes=self.classes,
            model_name=self.name,
            dataset_name=dataset.attrs.get(DATAFRAME_NAME, str(uuid4())),
            dataset=dataset,
            predictions=self.estimator.predict(split_dataset.x_test),
            truth_values=split_dataset.y_test,
            persistence=self.persistence,
        )


Classifier = TypeVar("Classifier", bound=ClassifierAnalytics)


class LinearRegressionModel(ClassifierModel):
    """Linear Regression Model."""

    def __init__(
        self: LinearRegressionModel,
        category_col: str,
        predictor_col: str,
        response_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        degree: int = 1,
        label_transformer: Callable[[Any], Any] | None = None,
        x_transform_func: Callable[Series | DataFrame] | None = None,
        y_transform_func: Callable[Series | DataFrame] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            predictor_col (str): The predictor column in the dataset.
            response_col (str): The response variable column of the dataframe.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            degree (int): The polynomial degree for the polynomial features transform.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
            x_transform_func (Callable[Series | DataFrame] | None): The function to
                transform x column.
            y_transform_func (Callable[Series | DataFrame] | None): The function to
                transform y column.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    (
                        "regression",
                        LinearRegression(),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
            x_transform_func=x_transform_func,
            y_transform_func=y_transform_func,
        )
        self.degree = degree
        self.predictor_col = predictor_col
        self.response_col = response_col

    @property
    def name(self: LinearRegressionModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Linear Regression"

    @property
    def regression(self: LinearRegressionModel) -> LinearRegression:
        """Get regression instance from the pipeline.

        Returns:
            str: The regression instance from the pipeline.
        """
        return self.estimator.named_steps.get("regression")

    def analyze_model(
        self: LinearRegressionModel, dataset: DataFrame
    ) -> LinearModelAnalytics:
        """Analyze the machine learning model against a dataset.

        Args:
            dataset (DataFrame): The dataset to use for the analysis.

        Returns:
            LinearModelAnalytics: The analytics results after training and testing.
        """
        # Transformer function to apply to the 'X' dataset.
        x_transformer: FunctionTransformer = FunctionTransformer(self.x_transform_func)

        # Transformer function to apply to the 'y' dataset.
        y_transformer: FunctionTransformer = FunctionTransformer(self.y_transform_func)

        # Split dataframe into X, y datasets for training/testing.
        y: Series = y_transformer.transform(dataset[self.response_col])
        x: DataFrame = x_transformer.transform(dataset[[self.predictor_col]])

        # Split the dataset into training/testing datasets split 50/50.
        split_dataset: SplitDataset = SplitDataset(
            *train_test_split(x, y, test_size=0.5, random_state=self.seed)
        )

        dataset_name: str = dataset.attrs.get(DATAFRAME_NAME, str(uuid4()))

        # Naming each dataframe.
        split_dataset.x_train.attrs[DATAFRAME_NAME] = f"{dataset_name}_x_train"
        split_dataset.x_test.attrs[DATAFRAME_NAME] = f"{dataset_name}_x_test"
        split_dataset.y_train.attrs[DATAFRAME_NAME] = f"{dataset_name}_y_train"
        split_dataset.y_test.attrs[DATAFRAME_NAME] = f"{dataset_name}_y_test"

        if self.persistence:
            save_dfs_to_checkpoints(
                [
                    split_dataset.x_train,
                    split_dataset.x_test,
                    split_dataset.y_train,
                    split_dataset.y_test,
                ]
            )

        # Create polynomial features object from the degree.
        poly: PolynomialFeatures = PolynomialFeatures(self.degree, include_bias=False)

        # Transform x_train using the polynomial features.
        x_p_train: np.array = poly.fit_transform(split_dataset.x_train)

        # Fit the model on x_train.
        # Train the model with the training datasets.
        self.estimator.fit(x_p_train, split_dataset.y_train)

        # Make predictions using the model.
        x_p_test: np.array = poly.fit_transform(split_dataset.x_test)
        y_predict: np.array = self.estimator.predict(x_p_test)

        x_s, y_s = zip(
            *sorted(
                zip(split_dataset.x_test[self.predictor_col], y_predict, strict=True)
            ),
            strict=True,
        )
        x_s = x_transformer.inverse_transform(x_s)
        y_s = y_transformer.inverse_transform(y_s)

        return LinearModelAnalytics(
            classes=self.classes,
            model_name=self.name,
            dataset=dataset,
            dataset_name=dataset.attrs.get(DATAFRAME_NAME, str(uuid4())),
            residuals=(split_dataset.y_test - y_predict),
            coefficients=self.regression.coef_,
            intercept=self.regression.intercept_,
            truth_values=split_dataset.y_test,
            predictions=y_predict,
            persistence=self.persistence,
            plot_data=LinearPlotData(
                x_label=self.predictor_col,
                y_label=self.response_col,
                x_scatter=split_dataset.x_test,
                y_scatter=split_dataset.y_test,
                x_regression=x_s,
                y_regression=y_s,
            ),
        )


class LogisticRegressionModel(ClassifierModel):
    """Logistic Regression Model."""

    def __init__(
        self: LogisticRegressionModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    (
                        "logistic_regression",
                        LogisticRegression(
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: LogisticRegressionModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Logistic Regression"


class LinearSVMModel(ClassifierModel):
    """Linear Kernel SVM Model."""

    def __init__(
        self: LinearSVMModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "linearsvc",
                        LinearSVC(
                            random_state=self.seed,
                            max_iter=10000,
                            loss="hinge",
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: LinearSVMModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Linear Kernel SVM"


class GaussianSVMModel(ClassifierModel):
    """Gaussian Kernel SVM Model."""

    def __init__(
        self: GaussianSVMModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "gaussiansvc",
                        SVC(
                            kernel="rbf",
                            gamma="scale",
                            C=1,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: GaussianSVMModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Gaussian Kernel SVM"


class PolynomialSVMModel(ClassifierModel):
    """Polynomial Kernel SVM Model."""

    def __init__(
        self: PolynomialSVMModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "polysvc",
                        SVC(
                            kernel="poly",
                            degree=3,
                            coef0=1,
                            C=10,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: PolynomialSVMModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Polynomial Kernel SVM"


class KNNModel(ClassifierModel):
    """k-Nearest Neighbor Model."""

    def __init__(
        self: KNNModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "knn",
                        KNeighborsClassifier(n_neighbors=8, p=2, metric="euclidean"),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: KNNModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "kNN Classifier"


class DecisionTreeModel(ClassifierModel):
    """Decision Tree Model."""

    def __init__(
        self: DecisionTreeModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "decision_tree",
                        DecisionTreeClassifier(
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: DecisionTreeModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Decision Tree"

    @property
    def decision_tree(self: DecisionTreeModel) -> DecisionTreeClassifier:
        """Get decision tree classifier instance from the pipeline.

        Returns:
            str: The decision tree classifier instance from the pipeline.
        """
        return self.estimator.named_steps.get("decision_tree")

    def export(
        self: DecisionTreeModel,
        dataset_name: str,
        output_directory: Path = artifacts_directory,
    ) -> tuple[Path, Path]:
        """Export the decision tree as a png and dot file.

        Args:
            dataset_name (str): The name of the dataset.
            output_directory (Path): The directory to export to.

        Returns:
            tuple[Path, Path]: The file paths to the dot and png file.
        """
        return save_tree(
            classifier=self.decision_tree,
            filename=f"{dataset_name.lower()}_{self.name.lower()}",
            output_directory=output_directory,
        )


class RandomForestModel(ClassifierModel):
    """Random Forest Model."""

    def __init__(
        self: RandomForestModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        trees: int = 100,
        max_depth: int | None = None,
        criterion: str = "entropy",
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            trees (int): The count of subtrees in the forest.
            max_depth (int | None): The max depth to use.
            criterion (str): The criterion to use.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "random_forest",
                        RandomForestClassifier(
                            n_estimators=trees,
                            max_depth=max_depth,
                            criterion=criterion,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )
        self.trees = trees
        self.max_depth = max_depth

    @property
    def name(self: RandomForestModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Random Forest"

    def analyze_model(
        self: RandomForestModel, dataset: DataFrame
    ) -> RandomForestAnalytics:
        """Analyze the random forest model against a dataset.

        Args:
            dataset (DataFrame): The dataset to analyze.

        Returns:
            RandomForestAnalytics: The analytics results.
        """
        analytics: ClassifierAnalytics = super().analyze_model(dataset)
        return RandomForestAnalytics(
            trees=self.trees,
            max_depth=self.max_depth,
            classes=analytics.classes,
            model_name=analytics.model_name,
            dataset_name=analytics.dataset_name,
            dataset=analytics.dataset,
            truth_values=analytics.truth_values,
            persistence=analytics.persistence,
            predictions=analytics.predictions,
        )


class NaiveBayesianModel(ClassifierModel):
    """Naive Bayesian Model."""

    def __init__(
        self: NaiveBayesianModel,
        category_col: str,
        estimator: Pipeline | None = None,
        persistence: bool = False,
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "naive_bayes",
                        MultinomialNB(),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )

    @property
    def name(self: NaiveBayesianModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "Naive Bayesian"


class KMeansModel(ClassifierModel):
    """K-Means Clustering Model."""

    def __init__(
        self: KMeansModel,
        category_col: str,
        features: list[str],
        estimator: Pipeline | None = None,
        persistence: bool = False,
        n_clusters: int = 8,
        n_init: int | str = "auto",
        label_transformer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize ML model with an estimator pipeline and attributes.

        Args:
            category_col (str): The category column in the dataset.
            features (list[str]): The features in the dataset to focus on.
            estimator (Pipeline | None): The estimator pipeline to use.
            persistence (bool): Flag to persist data and plots to files.
            n_clusters (bool): The number of clusters to use.
            n_init (int | str): The n_init to use.
            label_transformer (Callable[[Any], Any] | None): The label transform method
                to use.
        """
        super().__init__(
            category_col=category_col,
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "kmeans",
                        KMeans(
                            init="random",
                            n_clusters=n_clusters,
                            n_init=n_init,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            if estimator is None
            else estimator,
            persistence=persistence,
            label_transformer=label_transformer,
        )
        self.features: list[str] = features
        self.columns: list[str] = [*features, category_col]

    @property
    def name(self: KMeansModel) -> str:
        """Get the ML model's name.

        Returns:
            str: The ML model's name.
        """
        return "K-Means Clustering Classifier"

    @property
    def kmeans(self: KMeansModel) -> KMeans:
        """Get kmeans instance from the pipeline.

        Returns:
            str: The kmeans instance from the pipeline.
        """
        return self.estimator.named_steps.get("kmeans")

    @staticmethod
    def get_distance(row: Series, clusters: list[Cluster]) -> int:
        """Get nearest cluster to value in row and return label.

        Args:
            row (Series): The dataset row.
            clusters (list[Cluster]): The list of clusters to compute distance for.

        Returns:
            int: The label value.
        """
        distances = {}
        for i, cluster in enumerate(clusters):
            distances.update(
                {
                    i: np.linalg.norm(
                        np.array(row.to_numpy()) - np.array(cluster.centroid)
                    )
                }
            )
        closest = min(distances, key=distances.get)
        return closest + 1

    def analyze_model(self: KMeansModel, dataset: DataFrame) -> KMeansAnalytics:
        """Analyze the K-means clustering model.

        Args:
            dataset (DataFrame): The dataset to analyze.

        Returns:
            KMeansAnalytics: The analytics results.
        """
        # Get dataset of the features.
        dataset = dataset[self.columns]

        # Cluster over the features columns.
        predictions: np.array = self.estimator.fit_predict(dataset[self.features])

        labels: np.array = np.array(predictions)

        # Centroids for each cluster that were calculated from kmeans.
        centers: np.array = self.kmeans.cluster_centers_

        # Cluster data model for each cluster.
        clusters: list[Cluster] = [
            Cluster(
                label=label,
                dataset=dataset[labels == label],
                init_centroid=centers[cluster],
                category_col=self.category_col,
            )
            for cluster, label in enumerate(np.unique(labels))
        ]

        # Centroids from clusters.
        centroids: np.array = np.array([cluster.centroid for cluster in clusters])

        # Distances from each data point to each cluster's centroid.
        distances: DataFrame = DataFrame(
            data=euclidean_distances(
                dataset[self.features],
                centroids,
            )
        )

        # Find the closest distance.
        distances["closest"] = distances.apply(
            lambda row: row.idxmin() + 1
            if self.label_transformer is None
            else self.label_transformer(row.idxmin() + 1, False),
            axis=1,
        )

        return KMeansAnalytics(
            classes=self.classes,
            category_col=self.category_col,
            model_name=self.name,
            dataset_name=dataset.attrs.get(DATAFRAME_NAME, str(uuid4())),
            dataset=dataset,
            predictions=distances["closest"].to_numpy(),
            truth_values=labels,
            labels=labels,
            clusters=clusters,
            persistence=self.persistence,
            centroids=centroids,
            inertia=self.kmeans.inertia_,
            n_clusters=self.kmeans.n_clusters,
        )


class AnalyticsCollection(list[Generic[ModelAnalytics]]):
    """Analytics Collection Container class."""

    @property
    def models(self: AnalyticsCollection) -> Series:
        """Get a Series of the model names.

        Returns:
            Series: The series of model names.
        """
        return Series([analytic.model_name for analytic in self])

    @property
    def lowest_error(self: AnalyticsCollection) -> Classifier:
        """Get the classifier with the lowest error rate.

        Returns:
            Classifier: The classifier with the lowest error rate.
        """
        return min(self, key=attrgetter("error_rate"))

    @property
    def true_positives(self: AnalyticsCollection) -> Series:
        """Get a Series of the true positive counts for the models.

        Returns:
            Series: The series of true positive counts.
        """
        return Series([analytic.confusion_matrix.true_positives for analytic in self])

    @property
    def true_negatives(self: AnalyticsCollection) -> Series:
        """Get a Series of the true negative counts for the models.

        Returns:
            Series: The series of true negative counts.
        """
        return Series([analytic.confusion_matrix.true_negatives for analytic in self])

    @property
    def false_positives(self: AnalyticsCollection) -> Series:
        """Get a Series of the false positive counts for the models.

        Returns:
            Series: The series of false positive counts.
        """
        return Series([analytic.confusion_matrix.false_positives for analytic in self])

    @property
    def false_negatives(self: AnalyticsCollection) -> Series:
        """Get a Series of the false negative counts for the models.

        Returns:
            Series: The series of false negative counts.
        """
        return Series([analytic.confusion_matrix.false_negatives for analytic in self])

    @property
    def accuracy(self: AnalyticsCollection) -> Series:
        """Get a Series of the accuracy scores for the models.

        Returns:
            Series: The series of accuracy scores.
        """
        return Series([analytic.confusion_matrix.accuracy for analytic in self])

    @property
    def specificity(self: AnalyticsCollection) -> Series:
        """Get a Series of the specificity scores for the models.

        Returns:
            Series: The series of specificity scores.
        """
        return Series([analytic.confusion_matrix.specificity for analytic in self])

    @property
    def recall(self: AnalyticsCollection) -> Series:
        """Get a Series of the recall scores for the models.

        Returns:
            Series: The series of recall scores.
        """
        return Series([analytic.confusion_matrix.recall for analytic in self])

    @property
    def f1_score(self: AnalyticsCollection) -> Series:
        """Get a Series of the f1 scores for the models.

        Returns:
            Series: The series of f1 scores.
        """
        return Series([analytic.confusion_matrix.f1_score for analytic in self])

    @property
    def summary_table(self: AnalyticsCollection) -> DataFrame:
        """Get a summary table of the statistics for all models.

        Returns:
            DataFrame: The summary table of statistics.
        """
        return DataFrame(
            {
                ConfusionMatrixColumns.MODEL: self.models,
                ConfusionMatrixColumns.TRUE_POSITIVE: self.true_positives,
                ConfusionMatrixColumns.FALSE_POSITIVE: self.false_positives,
                ConfusionMatrixColumns.TRUE_NEGATIVE: self.true_negatives,
                ConfusionMatrixColumns.FALSE_NEGATIVE: self.false_negatives,
                ConfusionMatrixColumns.ACCURACY: self.accuracy,
                ConfusionMatrixColumns.TRUE_POSITIVE_RATE: self.recall,
                ConfusionMatrixColumns.TRUE_NEGATIVE_RATE: self.specificity,
                ConfusionMatrixColumns.F1_SCORE: self.f1_score,
            }
        )

    def plot(self: AnalyticsCollection) -> list[Path] | None:
        """Plot a comparison graph between the ML models.

        Returns:
            list[Path] | None: A list of file paths where the graphs were saved.
        """
        axes = self.summary_table.plot.bar(
            x=ConfusionMatrixColumns.MODEL, y=ConfusionMatrixColumns.scores(), rot=0
        )
        plt.gcf().subplots_adjust(bottom=0.3, left=0.18)

        wrap_labels(axes, 6, x_labels=True, y_labels=False)

        plt.title("ML Models Comparison")
        plt.tight_layout()
        paths: list[Path] = save_figures("ml_models_comparison")
        plt.show()
        return


class RandomForestAnalyticsCollection(AnalyticsCollection[RandomForestAnalytics]):
    """RandomForestAnalytics Collection Container class."""

    @property
    def tree_counts(self: RandomForestAnalyticsCollection) -> set[int]:
        """The tree count values that are in the collection.

        Returns:
            set[int]: The tree count values in the collection.
        """
        return {random_forest.trees for random_forest in self}

    @property
    def depths(self: RandomForestAnalyticsCollection) -> set[int]:
        """The max_depth values that are in the collection.

        Returns:
            set[int]: The max_depth values in the collection.
        """
        return {random_forest.max_depth for random_forest in self}

    def analytics_at_depth(self: RandomForestAnalyticsCollection, depth: int) -> list:
        """Get the classifier analytics at a specific max depth value.

        Args:
            depth (int): The max_depth value of the random forest.

        Returns:
            list: The list of analytics that have the max_depth provided.
        """
        return list(filter(lambda a: a.max_depth == depth, self))

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


class KMeansAnalyticsCollection(AnalyticsCollection[KMeansAnalytics]):
    """KMeansAnalytics Collection Container class."""

    @property
    def sse(self: KMeansAnalyticsCollection) -> list[MetricScore]:
        """The inertia values that are in the collection.

        Returns:
            set[MetricScore]: The inertia values in the collection.
        """
        return [analytic.inertia for analytic in self]

    @property
    def k_clusters(self: KMeansAnalyticsCollection) -> list[int]:
        """The k-clusters values that are in the collection.

        Returns:
            list[int]: The k-clusters values in the collection.
        """
        return [analytic.n_clusters for analytic in self]

    @property
    def knee(self: KMeansAnalyticsCollection) -> KneeLocator:
        """The knee of the inertia values that are in the collection.

        Calculated using the Kneedle algorithm.

        References:
            - https://github.com/arvkevi/kneed

        Returns:
            KneeLocator: The KneeLocator instance for the collection.
        """
        return KneeLocator(
            x=self.k_clusters, y=self.sse, curve="convex", direction="decreasing"
        )

    def show_knee_plot(self: KMeansAnalyticsCollection) -> int:
        """Show a knee plot for k-means clustering.

        Returns:
            int: Return the optimal k value based upon the calculated knee.
        """
        plt.plot(self.k_clusters, self.sse, "-b")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia (SSE)")
        plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
        plt.title("Knee Plot")
        save_figures(filename="k_means_knee_plot")
        plt.show()
        return self.knee.knee
