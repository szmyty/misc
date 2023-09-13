"""Assignment 4 module.

Alan Szmyt
Class: CS 677
Date: April 8th, 2023
Assignment #4
Description:
This module utilizes a variety of linear models to model relationships
between different clinical features for heart failure in patients.

References:
    - https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
"""
from __future__ import annotations

from operator import attrgetter
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from constants import (
    COL_CPK,
    COL_DEATH_EVENT,
    COL_SERUM_SODIUM,
    INITIAL_COLS,
    COL_SERUM_CREATININE,
    COL_PLATELETS
)
from utils import (
    artifacts,
    data,
    examine_correlation_matrix,
    save_df_to_checkpoints,
    save_figures, plot_correlation_matrix,
)


class DeathEvent(IntEnum):
    """Enum class of patient death event classifications."""

    SURVIVOR: int = 0
    DECEASED: int = 1


@dataclass
class PlotData:
    """Data class with data for plotting the linear model results."""
    x_label: str
    y_label: str
    x_scatter: object
    y_scatter: object
    x_regression: object
    y_regression: object


@dataclass
class LinearModelMetrics:
    """Data class with linear model metrics."""
    name: str
    r2_metric: float
    residuals: np.array
    coefficients: np.array
    intercept: float
    mse: float
    rmse: float
    mae: float
    sse: float
    plot_data: PlotData

    def plot(self: LinearModelMetrics, title: str) -> None:
        """Plot the linear model predicted values and actual values."""
        # Plot the x_test data in scatter plot.
        plt.scatter(
            self.plot_data.x_scatter,
            self.plot_data.y_scatter,
            color="dodgerblue"
        )

        # Plot the regression model predictions.
        plt.plot(self.plot_data.x_regression, self.plot_data.y_regression, color='red')
        plt.xlabel(self.plot_data.x_label)
        plt.ylabel(self.plot_data.y_label)
        plt.gcf().subplots_adjust(bottom=0.19, left=0.22)

        plt.title(title)

        # Save the plot to a file.
        save_figures("_".join(title.lower().split()), artifacts)

        plt.show()


class LinearModelAnalytics:
    """Linear model analytics container."""

    def __init__(
            self: LinearModelAnalytics,
            df: DataFrame,
            predictor_col: str,
            response_col: str,
    ) -> None:
        """Initialize the dataframe to use for training the models.

        Args:
            df (DataFrame): The dataframe to use for training and gathering metrics.
            predictor_col (str): The predictor variable column of the dataframe.
            response_col (str): The response variable column of the dataframe.
        """
        self.df: DataFrame = df
        self.predictor_col: str = predictor_col
        self.response_col: str = response_col

    def analyze_model(
            self: LinearModelAnalytics,
            model_name: str,
            degree: int = 1,
            x_transform_func=None,
            y_transform_func=None
    ) -> LinearModelMetrics:
        """Train and save the results of a linear regression model.

        Args:
            model_name (str): The name of the model to display.
            degree (int): The polynomial degree to use.
            x_transform_func: The function to transform x column.
            y_transform_func: The function to transform y column.

        Returns:
            LinearModelMetrics: A collection of metrics from the model results.
        """
        # Transformer function to apply to the 'X' dataset.
        x_transformer: FunctionTransformer = FunctionTransformer(x_transform_func)

        # Transformer function to apply to the 'y' dataset.
        y_transformer: FunctionTransformer = FunctionTransformer(y_transform_func)

        # Apply the transformer function to the 'y' dataset.
        y: Series = y_transformer.transform(self.df[self.response_col])

        # Apply the transformer function to 'X' dataset.
        x: DataFrame = x_transformer.transform(self.df[[self.predictor_col]])

        # Split the dataset into training/testing datasets split 50/50.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.5,
            random_state=42,
        )

        # Create polynomial features object from the degree.
        poly: PolynomialFeatures = PolynomialFeatures(degree, include_bias=False)

        # Transform x_train using the polynomial features.
        x_p_train: np.array = poly.fit_transform(x_train)

        # Fit the model on x_train.
        model: LinearRegression = LinearRegression()
        model.fit(x_p_train, y_train)

        # Make predictions using the model.
        x_p_test: np.array = poly.fit_transform(x_test)
        y_predict: np.array = model.predict(x_p_test)

        mse: float = mean_squared_error(y_test, y_predict)

        x_s, y_s = zip(*sorted(zip(x_test[self.predictor_col], y_predict)))
        x_s = x_transformer.inverse_transform(x_s)
        y_s = y_transformer.inverse_transform(y_s)

        return LinearModelMetrics(
            name=model_name,
            r2_metric=r2_score(y_test, y_predict),
            residuals=(y_test - y_predict),
            coefficients=model.coef_,
            intercept=model.intercept_,
            mse=mse,
            rmse=mean_squared_error(y_test, y_predict, squared=False),
            mae=mean_absolute_error(y_test, y_predict),
            sse=y_predict.shape[0] * mse,  # or np.sum(np.square(y_predict - y_test))
            plot_data=PlotData(
                x_label=self.predictor_col,
                y_label=self.response_col,
                x_scatter=x_test,
                y_scatter=y_test,
                x_regression=x_s,
                y_regression=y_s
            )
        )

    @property
    def simple(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the metrics of a simple linear regression model.

        y = ax + b

        Returns:
            LinearModelMetrics: The metrics results after training the model.
        """
        return self.analyze_model(model_name="Simple Regression Model", degree=1)

    @property
    def quadratic(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the metrics of a quadratic linear regression model.

        y = ax^2 + bx + c

        Returns:
            LinearModelMetrics: The metrics results after training the model.
        """
        return self.analyze_model(model_name="Quadratic Model", degree=2)

    @property
    def cubic(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the metrics of a cubic spline linear regression model.

        y = ax^3 + bx^2 + cx + d

        Returns:
            LinearModelMetrics: The metrics results after training the model.
        """
        return self.analyze_model(model_name="Cubic Spline Model", degree=3)

    @property
    def glm1(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the metrics of a generalized linear regression model (GLM).

        y = a log(x) + b

        Returns:
            LinearModelMetrics: The metrics results after training the model.
        """
        return self.analyze_model(model_name="GLM #1 Model", x_transform_func=np.log)

    @property
    def glm2(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the metrics of a generalized linear regression model (GLM).

        log(y) = a log(x) + b

        Returns:
            LinearModelMetrics: The metrics results after training the model.
        """
        return self.analyze_model(
            model_name="GLM #2 Model",
            x_transform_func=np.log,
            y_transform_func=np.log
        )

    @property
    def models(self: LinearModelAnalytics) -> list[LinearModelMetrics]:
        """Get a list of the models available.

        Returns:
            list[LinearModelAnalytics]: List of all models.
        """
        return [
            self.simple,
            self.quadratic,
            self.cubic,
            self.glm1,
            self.glm2
        ]

    @property
    def best_model(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the best performing model (the smallest SSE value)"""
        return min(self.models, key=attrgetter("sse"))

    @property
    def worst_model(self: LinearModelAnalytics) -> LinearModelMetrics:
        """Get the worst performing model (the largest SSE value)"""
        return max(self.models, key=attrgetter("sse"))

    @staticmethod
    def sse_table(
        survivors_df: LinearModelAnalytics,
        deceased_df: LinearModelAnalytics,
    ) -> DataFrame:
        """Build a table of SSE values for each model.

        Args:
            survivors_df (LinearModelAnalytics): The survivors results.
            deceased_df (LinearModelAnalytics): The deceased results.

        Returns:
            DataFrame: The dataframe table of SSE results.
        """
        return DataFrame(
            {
                "Model": Series(
                    [
                        r"$y = ax + b$",
                        r"$y = ax^2 + bx + c$",
                        r"$y = ax^3 + bx^2 + cx + d$",
                        r"$y = a\log{x} + b$",
                        r"$\log{y} = a\log{x} + b$"
                    ]
                ),
                r"SSE (death\_event=0)": Series(
                    [
                        survivors_df.simple.sse,
                        survivors_df.quadratic.sse,
                        survivors_df.cubic.sse,
                        survivors_df.glm1.sse,
                        survivors_df.glm2.sse,
                    ]
                ),
                r"SSE (death\_event=1)": Series(
                    [
                        deceased_df.simple.sse,
                        deceased_df.quadratic.sse,
                        deceased_df.cubic.sse,
                        deceased_df.glm1.sse,
                        deceased_df.glm2.sse,
                    ]
                ),
            }
        )


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

    # region Initial Setup.
    # Global Seaborn options.
    sns.set_theme(font_scale=1.5, rc={"text.usetex": True})

    # Global pandas options.
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_seq_items", 50)
    pd.set_option("display.show_dimensions", False)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.precision", 4)
    pd.set_option("styler.format.precision", 4)

    cwd: Path = Path.cwd()

    # Heart failure clinical records dataset file from UCI.
    dataset_csv: str = "heart_failure_clinical_records_dataset.csv"
    heart_failure_dataset_file: Path = data.joinpath(dataset_csv)
    # endregion

    # region Question 1

    # region Question 1.1
    # dtype mapping to use for the csv file. Set death_event to categorical.
    dtypes: dict = defaultdict(
        np.float64,
        {
            COL_DEATH_EVENT: pd.CategoricalDtype.name,
            COL_CPK: np.int64,
            COL_SERUM_SODIUM: np.int64,
        },
    )

    # Load the heart failure records into a dataframe.
    heart_failure_dataset: DataFrame = pd.read_csv(
        heart_failure_dataset_file, usecols=INITIAL_COLS, dtype=dtypes
    )
    save_df_to_checkpoints(heart_failure_dataset, "heart_failure_dataset")

    # Load the survivors into their own dataframe.
    survivors: DataFrame = heart_failure_dataset.loc[
        heart_failure_dataset[COL_DEATH_EVENT].astype(int) == DeathEvent.SURVIVOR
        ]
    save_df_to_checkpoints(survivors, "survivors")

    # Load the deceased into their own dataframe.
    deceased: DataFrame = heart_failure_dataset.loc[
        heart_failure_dataset[COL_DEATH_EVENT].astype(int) == DeathEvent.DECEASED
        ]
    save_df_to_checkpoints(deceased, "deceased")
    # endregion

    # region Question 1.2
    # Plot a correlation matrix for the 'survivors' dataset.
    survivors_correlations: DataFrame = plot_correlation_matrix(
        survivors,
        title="Survivors Correlation Matrix",
        output_dir=artifacts
    )

    # Plot a correlation matrix for the 'deceased' dataset.
    deceased_correlations: DataFrame = plot_correlation_matrix(
        deceased,
        title="Deceased Correlation Matrix",
        output_dir=artifacts
    )
    # endregion

    # region Question 1.3
    # Examine the survivor patients' correlation matrix.
    s_correlations = examine_correlation_matrix(survivors_correlations)

    # Examine the deceased patients' correlation matrix.
    d_correlations = examine_correlation_matrix(deceased_correlations)

    # Question 1.3.a
    # Survivors highest correlation features.
    shc_features = list(s_correlations.head(1).to_dict().items())[0]
    print(
        f"The features with the highest correlation for surviving patients were "
        f"'{shc_features[0][0]}' and '{shc_features[0][1]}' with a correlation value of"
        f" {shc_features[1]:.3f}."
    )

    # Question 1.3.b
    # Survivors lowest correlation features.
    slc_features = list(s_correlations.tail(1).to_dict().items())[0]
    print(
        f"The features with the lowest correlation for surviving patients were "
        f"'{slc_features[0][0]}' and '{slc_features[0][1]}' with a correlation value of"
        f" {slc_features[1]:.3f}."
    )

    # Question 1.3.c
    # Deceased patients' highest correlation features.
    dhc_features = list(d_correlations.head(1).to_dict().items())[0]
    print(
        f"The features with the highest correlation for deceased patients were "
        f"'{dhc_features[0][0]}' and '{dhc_features[0][1]}' with a correlation value of"
        f" {dhc_features[1]:.3f}."
    )

    # Question 1.3.d
    # Deceased patients' lowest correlation features.
    dlc_features = list(d_correlations.tail(1).to_dict().items())[0]
    print(
        f"The features with the lowest correlation for deceased patients were "
        f"'{dlc_features[0][0]}' and '{dlc_features[0][1]}' with a correlation value of"
        f" {dlc_features[1]:.3f}."
    )

    # Question 1.3.e
    print("The results are not the same for both cases.")

    # endregion
    print("Finished Question 1!")
    # endregion

    # region Question 2

    # region Question 2.1
    # Linear model analytics for different linear regression models for survivors.
    survivors_linear_model_analytics: LinearModelAnalytics = LinearModelAnalytics(
        survivors,
        predictor_col=COL_PLATELETS,
        response_col=COL_SERUM_CREATININE,
    )

    # Linear model analytics for different linear regression models for deceased.
    deceased_linear_model_analytics: LinearModelAnalytics = LinearModelAnalytics(
        deceased,
        predictor_col=COL_PLATELETS,
        response_col=COL_SERUM_CREATININE,
    )

    # SURVIVOR METRICS #
    # Metrics for simple linear regression model for survivors.
    s_slr_metrics: LinearModelMetrics = survivors_linear_model_analytics.simple
    print(f"Simple linear coefficients: {s_slr_metrics.coefficients}")
    print(f"Sum of Squared Errors (SSE): {s_slr_metrics.sse}")
    s_slr_metrics.plot("Simple Linear Regression")

    # Metrics for quadratic linear regression model for survivors.
    s_qlr_metrics: LinearModelMetrics = survivors_linear_model_analytics.quadratic
    print(f"Quadratic coefficients: {s_qlr_metrics.coefficients}")
    print(f"Sum of Squared Errors (SSE): {s_qlr_metrics.sse}")
    s_qlr_metrics.plot("Quadratic Linear Model")

    # Metrics for cubic linear regression model for survivors.
    s_clr_metrics: LinearModelMetrics = survivors_linear_model_analytics.cubic
    print(f"Cubic coefficients: {s_clr_metrics.coefficients}")
    print(f"Sum of Squared Errors (SSE): {s_clr_metrics.sse}")
    s_clr_metrics.plot("Cubic Linear Model")

    # Metrics for GLM #1 linear regression model for survivors.
    s_glr_metrics1: LinearModelMetrics = survivors_linear_model_analytics.glm1
    print(f"GLM coefficients: {s_glr_metrics1.coefficients}")
    print(f"Sum of Squared Errors (SSE): {s_glr_metrics1.sse}")
    s_glr_metrics1.plot("GLM Linear Model")

    # Metrics for GLM #2 linear regression model for survivors.
    s_glr_metrics2: LinearModelMetrics = survivors_linear_model_analytics.glm2
    print(f"GLM coefficients: {s_glr_metrics2.coefficients}")
    print(f"Sum of Squared Errors (SSE): {s_glr_metrics2.sse}")
    s_glr_metrics2.plot("GLM Linear Model")

    # DECEASED METRICS #
    # Metrics for simple linear regression model for deceased.
    d_slr_metrics: LinearModelMetrics = deceased_linear_model_analytics.simple
    print(f"Simple linear coefficients: {d_slr_metrics.coefficients}")
    print(f"Sum of Squared Errors (SSE): {d_slr_metrics.sse}")
    d_slr_metrics.plot("Simple Linear Regression")

    # Metrics for quadratic linear regression model for deceased.
    d_qlr_metrics: LinearModelMetrics = deceased_linear_model_analytics.quadratic
    print(f"Quadratic coefficients: {d_qlr_metrics.coefficients}")
    print(f"Sum of Squared Errors (SSE): {d_qlr_metrics.sse}")
    d_qlr_metrics.plot("Quadratic Linear Model")

    # Metrics for cubic linear regression model for deceased.
    d_clr_metrics: LinearModelMetrics = deceased_linear_model_analytics.cubic
    print(f"Cubic coefficients: {d_clr_metrics.coefficients}")
    print(f"Sum of Squared Errors (SSE): {d_clr_metrics.sse}")
    d_clr_metrics.plot("Cubic Linear Model")

    # Metrics for GLM #1 linear regression model for deceased.
    d_glr_metrics1: LinearModelMetrics = deceased_linear_model_analytics.glm1
    print(f"GLM coefficients: {d_glr_metrics1.coefficients}")
    print(f"Sum of Squared Errors (SSE): {d_glr_metrics1.sse}")
    d_glr_metrics1.plot("GLM Linear Model")

    # Metrics for GLM #2 linear regression model for deceased.
    d_glr_metrics2: LinearModelMetrics = deceased_linear_model_analytics.glm2
    print(f"GLM coefficients: {d_glr_metrics2.coefficients}")
    print(f"Sum of Squared Errors (SSE): {d_glr_metrics2.sse}")
    d_glr_metrics2.plot("GLM Linear Model")

    # endregion

    print("Finished Question 2!")
    # endregion

    # region Question 3

    # Get the best model for the survivors.
    s_best_model = survivors_linear_model_analytics.best_model
    print(
        f"The best model for the survivors is {s_best_model.name}"
        f" with an SSE value of {s_best_model.sse}"
    )

    # Get the best model for the deceased.
    d_best_model = deceased_linear_model_analytics.best_model
    print(
        f"The best model for the deceased is {d_best_model.name}"
        f" with an SSE value of {d_best_model.sse}"
    )

    # Get the worst model for the survivors.
    s_worst_model = survivors_linear_model_analytics.worst_model
    print(
        f"The worst model for the survivors is {s_worst_model.name}"
        f" with an SSE value of {s_worst_model.sse}"
    )

    # Get the worst model for the deceased.
    d_worst_model = deceased_linear_model_analytics.worst_model
    print(
        f"The worst model for the deceased is {d_worst_model.name}"
        f" with an SSE value of {d_worst_model.sse}"
    )

    # Summarize the results of SSE of all models in a table.
    sse_results: DataFrame = LinearModelAnalytics.sse_table(
        survivors_linear_model_analytics,
        deceased_linear_model_analytics
    )

    print("Finished Question 3!")
    # endregion
