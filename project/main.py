#!/usr/env/bin python3
"""Final Project module.

Alan Szmyt
Class: CS 677
Date: April 25th, 2023
Final Project
Description:
Compare machine learning models for classifying genres based upon music audio track
attributes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns
from analyzer import (
    ClassifierAnalytics,
    LinearSVMModel,
    LogisticRegressionModel,
    ModelAnalyzer,
)
from analyzer.analytics import (
    AnalyticsCollection,
    KNNModel,
    LinearModelAnalytics,
    LinearRegressionModel,
    RandomForestAnalytics,
    RandomForestAnalyticsCollection,
    RandomForestModel,
)
from dataset.spotify.models import Genre, SpotifyColumns
from pandas import DataFrame

from project.utils import (
    DATAFRAME_NAME,
    data_directory,
    examine_correlation_matrix,
    log,
    save_df_to_checkpoints,
)

if TYPE_CHECKING:
    from pathlib import Path


class Dataset(DataFrame):
    """Dataframe extension to add a name to the dataset attributes."""

    def __init__(self: Dataset, name: str, *args, **kwargs) -> None:
        """Initialize a DataFrame with a name.

        Args:
            name (str): The name of the dataset.
            *args: Additional positional arguments to the DataFrame.
            **kwargs: Additional keyword arguments to the DataFrame.
        """
        super().__init__(*args, **kwargs)
        self.attrs[DATAFRAME_NAME] = name

    @classmethod
    def read_csv(cls: type[Dataset], name: str, *args, **kwargs) -> Dataset:
        """Get a dataframe from a csv file.

        Args:
            name (str): The name of the dataset.
            *args: Additional positional arguments to the read_csv method.
            **kwargs: Additional keyword arguments to the read_csv method.

        Returns:
            Dataset: The DataFrame with a name attribute.
        """
        return cls(name=name, data=pd.read_csv(*args, **kwargs).to_dict())


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

    # region Initial Setup.
    # Global Seaborn options.
    sns.set_theme(font_scale=1.5, style="darkgrid", rc={"text.usetex": True})

    # Spotify tracks dataset.
    dataset_csv: str = "tracks.csv"
    tracks_dataset_file: Path = data_directory.joinpath(dataset_csv)

    # Load the Spotify tracks dataset from the csv file.
    tracks_dataset: Dataset = Dataset.read_csv(
        name="spotify_tracks_dataset",
        filepath_or_buffer=tracks_dataset_file,
        engine="python",
        dtype=SpotifyColumns.dtype(),
    )

    tracks_dataset = tracks_dataset.drop(
        [SpotifyColumns.NAME, SpotifyColumns.MODE], axis=1
    )

    # Load the heavy metal tracks into their own dataframe.
    heavy_metal_tracks: DataFrame = tracks_dataset.loc[
        tracks_dataset[SpotifyColumns.GENRE].astype(Genre.dtype()) == Genre.METAL
    ]
    save_df_to_checkpoints(heavy_metal_tracks, "heavy_metal_tracks")

    # Load the country tracks into their own dataframe.
    country_tracks: DataFrame = tracks_dataset.loc[
        tracks_dataset[SpotifyColumns.GENRE].astype(Genre.dtype()) == Genre.COUNTRY
    ]
    save_df_to_checkpoints(country_tracks, "country_tracks")
    # endregion

    # region Classifiers
    # Machine learning model analyzer context class.
    analyzer: ModelAnalyzer = ModelAnalyzer(
        dataset=tracks_dataset,
        model=LinearSVMModel(
            category_col=SpotifyColumns.GENRE,
            persistence=True,
            label_transformer=Genre.from_label,
        ),
        category_col=SpotifyColumns.GENRE,
    )

    # Plot a correlation matrix between the different music attributes.
    correlation_matrix: DataFrame = analyzer.correlation_matrix()

    # Plot a radar plot of the music attributes per genre.
    analyzer.radar_plot(
        title="Music Feature Attributes by Genre",
        label_transformer=Genre.from_label
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

    # Train random forest model, make predictions, and gather analytics.
    random_forest_analytics: RandomForestAnalyticsCollection = (
        RandomForestAnalyticsCollection()
    )

    random_forests: list[RandomForestModel] = [
        RandomForestModel(
            category_col=SpotifyColumns.GENRE,
            persistence=True,
            label_transformer=Genre.from_label,
            trees=n,
            max_depth=d,
        )
        for n in range(1, 11)
        for d in range(1, 6)
    ]
    for random_forest in random_forests:
        analyzer.model = random_forest
        random_forest_analytics.append(analyzer.analyze())

    # Error plot for all random forest classifiers.
    random_forest_analytics.show_error_plot()

    # Get the random forest with the lowest error rate.
    best_random_forest: RandomForestAnalytics = random_forest_analytics.lowest_error
    log(
        f"Best combination of N and d is: "
        f"N={best_random_forest.trees}, "
        f"d={best_random_forest.max_depth}"
    )

    log(
        f"The accuracy for best random forest is: "
        f"{best_random_forest.confusion_matrix.accuracy.score}"
    )

    # Compute the confusion matrix.
    best_random_forest.show_confusion_matrix()

    # Train logistic regression model, make predictions, and gather analytics.
    analyzer.model = LogisticRegressionModel(
        category_col=SpotifyColumns.GENRE,
        persistence=True,
        label_transformer=Genre.from_label,
    )
    logistic_regression_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(
        f"Logistic regression classifier accuracy: "
        f"{logistic_regression_analytics.confusion_matrix.accuracy.score}"
    )

    # Train kNN model, make predictions, and gather analytics.
    analyzer.model = KNNModel(
        category_col=SpotifyColumns.GENRE,
        persistence=True,
        label_transformer=Genre.from_label,
    )
    knn_analytics: ClassifierAnalytics = analyzer.analyze()

    # Compute the accuracy.
    log(f"kNN classifier accuracy: {knn_analytics.confusion_matrix.accuracy.score}")

    # Compute the confusion matrix.
    knn_analytics.show_confusion_matrix()

    # Gather analytics for all classifier models.
    classifier_analytics: AnalyticsCollection = AnalyticsCollection(
        [
            logistic_regression_analytics,
            linear_svm_analytics,
            knn_analytics,
            best_random_forest,
        ]
    )

    classifier_analytics.plot()
    # endregion

    # region Regression
    # Plot a correlation matrix for heavy metal tracks.
    metal_correlation_matrix = ModelAnalyzer.create_correlation_matrix(
        dataset=heavy_metal_tracks.drop(SpotifyColumns.GENRE, axis=1),
        title="Heavy Metal Tracks Correlation Matrix"
    )

    # Examine the heavy metal tracks' correlation matrix.
    metal_correlations = examine_correlation_matrix(metal_correlation_matrix)

    # Get the highest correlated features.
    mhc_features = list(metal_correlations.head(1).to_dict().items())[0]

    # Train linear regression model, make predictions, and gather analytics.
    analyzer.dataset = heavy_metal_tracks
    analyzer.model = LinearRegressionModel(
        category_col=SpotifyColumns.GENRE,
        predictor_col=SpotifyColumns.from_column(mhc_features[0][0]),
        response_col=SpotifyColumns.from_column(mhc_features[0][1]),
        persistence=True,
        label_transformer=Genre.from_label,
        degree=3,
    )
    linear_regression_analytics: LinearModelAnalytics = analyzer.analyze()
    linear_regression_analytics.plot(title="Linear Regression for Metal Tracks")

    # Plot a correlation matrix for heavy metal tracks.
    country_correlation_matrix = ModelAnalyzer.create_correlation_matrix(
        dataset=country_tracks.drop(SpotifyColumns.GENRE, axis=1),
        title="Country Tracks Correlation Matrix"
    )

    # Examine the country tracks' correlation matrix.
    country_correlations = examine_correlation_matrix(country_correlation_matrix)

    # Get the highest correlated features.
    chc_features = list(country_correlations.head(1).to_dict().items())[0]

    # Train linear regression model, make predictions, and gather analytics.
    country_predictor_col = SpotifyColumns.from_column(chc_features[0][0])
    country_response_col = SpotifyColumns.from_column(chc_features[0][1])

    analyzer.dataset = country_tracks
    analyzer.model = LinearRegressionModel(
        category_col=SpotifyColumns.GENRE,
        predictor_col=country_predictor_col,
        response_col=country_response_col,
        persistence=True,
        label_transformer=Genre.from_label,
        degree=3,
    )
    linear_regression_analytics: LinearModelAnalytics = analyzer.analyze()
    linear_regression_analytics.plot(title="Linear Regression for Country Tracks")

    # Examine the tracks' correlation matrix.
    track_correlations = examine_correlation_matrix(correlation_matrix)

    # Get the highest correlated features.
    hc_features = list(track_correlations.head(1).to_dict().items())[0]

    # Train linear regression model, make predictions, and gather analytics.
    hc_predictor_col = SpotifyColumns.from_column(hc_features[0][0])
    hc_response_col = SpotifyColumns.from_column(hc_features[0][1])

    # Create pairwise plot for tracks with high correlation features.
    ModelAnalyzer.create_pairwise_plot(
        dataset=tracks_dataset[
            [hc_predictor_col, hc_response_col, SpotifyColumns.GENRE]
        ],
        category_col=SpotifyColumns.GENRE,
        title="Tracks Pairwise Plot for High Correlation Features",
        label_transformer=Genre.from_label_to_title
    )
    # endregion
