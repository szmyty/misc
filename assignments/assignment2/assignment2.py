"""Assignment2 module.

Alan Szmyt
Class: CS 677
Date: March 28th, 2023
Assignment #2
Description:
This module is an exercise to splitting a dataset into training and testing subsets and
using different methods with truth labels to analyze what methods of prediction work the
best when it comes to predicting stocks when day trading.
"""
from __future__ import annotations

import operator
from enum import Enum
from logging import DEBUG, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from constants import (
    ACCURACY_KEY,
    BUY_AND_HOLD_KEY,
    DATE_KEY,
    DOWN_DAY_COLOR,
    ENSEMBLE_KEY,
    ENSEMBLE_ORACLE_KEY,
    FN_KEY,
    FP_KEY,
    RETURN_KEY,
    SONY_TICKER,
    SPY_TICKER,
    STYLER_OPTIONS,
    TICKER_KEY,
    TN_KEY,
    TNR,
    TP_KEY,
    TPR,
    TRUE_LABEL_KEY,
    UP_DAY_COLOR,
    W_KEY,
    W_ORACLE,
    YEAR_KEY,
)
from IPython.display import HTML, display, Markdown
from pandas import DataFrame, Series
from pandas.io.formats.style import Styler
from utils import (
    compute_accuracy,
    compute_probability,
    df_unique_list,
    find_occurrences,
    float_to_percentage,
    mean,
    most_common,
    resources,
    sliding_window,
)

# Matplotlib global formats.
default_datefmt = mdates.DateFormatter("%b\n-%Y")
currency_fmt = mtick.StrMethodFormatter("${x:,.0f}")

# Seaborn styling settings.
# Reference: https://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/t0c_intro_to_latex.html # noqa
# rc = {'lines.linewidth': 2,
#       'axes.labelsize': 18,
#       'axes.titlesize': 18,
#       'axes.facecolor': 'DFDFE5'}
# sns.set_context('notebook', rc=rc)
# sns.set_style('darkgrid', rc=rc)

# Flag to disable showing tables (For readability during development).
should_show_tables: bool = True

# Configure a logger to log statements.
logger: Logger = getLogger(__file__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# Sony stock ticker file path.
sony_ticker_file: Path = resources.joinpath(f"{SONY_TICKER}.csv")

# S&P-500 stock ticker file path and ticker abbreviation.
spy_ticker_file: Path = resources.joinpath(f"{SPY_TICKER}.csv")


class DayLabel(Enum):
    """Enum class of a stock day result, aka an 'up' day or 'down' day."""

    UP = "+"
    DOWN = "-"

    def opposite(self: DayLabel) -> DayLabel:
        """Get the opposite day label of the current instance.

        Returns:
            DayLabel: The opposite day label of the current instance.
        """
        return self.UP if self is DayLabel.DOWN else self.DOWN


def show_table(table: DataFrame | Styler, max_rows: int = 5) -> None:
    """Show the data table.

    Convert the styler's current state to HTML and display it in the notebook.

    Args:
        table (Styler): The table to use for converting the data to HTML.
        max_rows (int): The maximum rows to display (Defaults to 5).
    """
    if should_show_tables:
        table_to_show: Styler = table if isinstance(table, Styler) else table.style

        display(Markdown(table_to_show.data.head(max_rows).style.to_latex()))


def show_tables(tables: list[Styler], max_rows: int = 5) -> None:
    """Show each table in the list.

    Args:
        tables (list[Styler]): The list of tables to show.
        max_rows (int): The max rows to show for all the tables (Defaults to 5).
    """
    for table in tables:
        show_table(table, max_rows=max_rows)


def color_true_label(row: Series) -> Series:
    """Apply color to each 'True Label' symbol

    The color is determined based upon if it is an 'up day' or 'down day'.
    The '+' symbol will be colorized as green.
    The '-' symbol will be colorized as red.

    Args:
        row (Series): The data row to apply the style to.

    Returns:
        Series: The updated row with the applied styling.
    """
    color: str = UP_DAY_COLOR if row[RETURN_KEY] >= 0 else DOWN_DAY_COLOR
    return Series({TRUE_LABEL_KEY: f"color: {color};"})


def true_label_symbol(row: Series) -> str:
    """Determine the 'True Label' symbol based upon the daily return value of the row.

    Args:
        row (Series): The data row to add the appropriate 'True Label' to.

    Returns:
        str: The 'True Label' value to add to the row.
    """
    return "+" if row[RETURN_KEY] >= 0 else "-"


def colorize(styler: Styler) -> None:
    """Colorize the true labels of the styler's dataset.

    Args:
        styler (Styler): The styler to apply the color to.
    """
    styler.apply(color_true_label, subset=[RETURN_KEY, TRUE_LABEL_KEY], axis=1)


def style_df(
    data: DataFrame, caption: str = "Dataset", color: bool = True
 ) -> Styler:
    """Style the provided dataframe.

    Args:
        data (DataFrame): The dataframe to style.
        caption (str): The caption to use for the styled table.
        color (bool): Colorize the truth labels.

    Returns:
        Styler: The styled dataframe.
    """
    styler: Styler = Styler(data=data, caption=caption, **STYLER_OPTIONS)

    try:
        if TRUE_LABEL_KEY in data.columns and color:
            colorize(styler)
    except TypeError:
        pass

    return styler


class Assignment2Error(Exception):
    """Custom Exception for Assignment 2."""

    def __init__(
        self: Assignment2Error,
        message: str = "Assignment 2 Error",
        error: Exception | None = None,
    ) -> None:
        """Instantiate custom exception for Assignment 2.

        Args:
            message (str): Description of the error that was raised.
            error (Exception | None): Optional Exception to associate with the instance.
        """
        self.error: Exception | None = error
        self.message: str = message
        super().__init__(self.message)


# region Helper methods for operating on the dataframes.
def add_true_labels(df: DataFrame) -> None:
    """Add the 'True Label' values to the dataframe based upon each day's return.

    Args:
        df (DataFrame): The dataframe to add to.
    """
    df[TRUE_LABEL_KEY] = df.apply(true_label_symbol, axis=1)


def df_unique_years(df: DataFrame) -> list[int]:
    """Get a list of the unique years in the dataset.

    Args:
        df (DataFrame): The dataframe to get the unique years from.

    Returns:
        list[int]: The unique years in the dataset.
    """
    return df_unique_list(df, YEAR_KEY)


def get_training_data(df: DataFrame, years: list[int]) -> DataFrame:
    """Get the training dataset, aka years 1, 2, and 3.

    Args:
        df (DataFrame): The dataset to get the training data from.
        years (list[int]): The list of years.

    Returns:
        DataFrame: The training data subset dataframe.
    """
    return df.loc[lambda d: d[YEAR_KEY].isin(years[:3]), :].reset_index(drop=True)


def get_testing_data(df: DataFrame, years: list[int]) -> DataFrame:
    """Get the testing dataset, aka years 4 and 5.

    Args:
        df (DataFrame): The dataset to get the testing data from.
        years (list[int]): The list of years.

    Returns:
        DataFrame: The testing data subset dataframe.
    """
    return df.loc[lambda d: d[YEAR_KEY].isin(years[3:5]), :].reset_index(drop=True)


def get_up_days(df: DataFrame) -> DataFrame:
    """Get all rows with up days in the true label column.

    Args:
        df (DataFrame): The dataset to get the up days from.

    Returns:
        DataFrame: A dataframe with all rows with up days in the true label column.
    """
    return df[df[TRUE_LABEL_KEY] == "+"]


def get_down_days(df: DataFrame) -> DataFrame:
    """Get all rows with down days in the true label column.

    Args:
        df (DataFrame): The dataset to get the down days from.

    Returns:
        DataFrame: A dataframe with all rows with down days in the true label column.
    """
    return df[df[TRUE_LABEL_KEY] == "-"]


def get_labels(df: DataFrame) -> np.ndarray:
    """Get an array of the labels in the truth column.

    Args:
        df (DataFrame): The dataframe to get the array of labels from.

    Returns:
        np.ndarray: An array of the labels from the truth column.
    """
    return df[TRUE_LABEL_KEY].to_numpy()


def compute_ensemble(df: DataFrame, columns: list[str]) -> None:
    """Add an ensemble column to the provided dataframe based upon the columns.

    Args:
        df (DataFrame): The dataframe to compute the ensemble for.
        columns (list[str]): The column names to use for calculating the ensemble value.
    """
    df[ENSEMBLE_KEY] = df.dropna(subset=columns).apply(
        most_common, args=(columns,), axis=1
    )


def consecutive_day_probability(
    k_days: int, labels: np.ndarray, consecutive_label: DayLabel, next_label: DayLabel
) -> float:
    """Use a sliding window to compute the probability of the next day.

    After seeing consecutive days in the array, we want to know the probability of the
    next day being a specific label.

    Args:
        k_days (int): The amount of days 'k'.
        labels (np.ndarray): The original array of labels.
        consecutive_label (DayLabel): The label to repeat for consecutive days.
        next_label (DayLabel): The label to compute the probability for.

    Returns:
        float: The probability of the next day label.
    """
    # Array of repeating labels, so the pattern continues.
    consecutive: np.ndarray = np.repeat([consecutive_label.value], k_days + 1)

    # Array of the original repeating, except the last day opposed.
    opposition = np.concatenate(
        (consecutive[:-1], np.array([consecutive_label.opposite().value]))
    )

    consecutive_occurrences: int = 0
    opposition_occurrences: int = 0

    # Using a sliding window to find pattern matches of the original dataset.
    for window in sliding_window(labels, size=k_days + 1):
        pattern: np.ndarray = np.fromiter(window, dtype="U1")
        if np.array_equal(pattern, consecutive):
            consecutive_occurrences = consecutive_occurrences + 1
        elif np.array_equal(pattern, opposition):
            opposition_occurrences = opposition_occurrences + 1

    total_occurrences: int = opposition_occurrences + consecutive_occurrences

    if next_label == consecutive_label:
        return compute_probability(consecutive_occurrences, total_occurrences)
    else:
        return compute_probability(opposition_occurrences, total_occurrences)


def predict_next_day(
    df: DataFrame,
    training_labels: np.ndarray,
    window_size: int,
    default_probability: float,
) -> list:
    """Predict the next day using a particular window size for looking back.

    Args:
        df (DataFrame): The data frame to roll over.
        training_labels (np.ndarray): The array of training labels.
        window_size (int): The window size to use for the rolling.
        default_probability (float): The default probability value to use if both the
            occurrences are 0 (should be rare).

    Returns:
        list: A list of symbols to be used as predictions.
    """
    # Start the result list with 'nan' because the first day can't look back.
    result: list[str | np.nan] = [np.nan]
    for window in df.rolling(window=window_size):
        # Don't predict the next day outside the dataset.
        if len(result) >= len(df.index):
            break

        # The current window's array of labels.
        labels: np.ndarray = get_labels(window)

        # If the window size goes back beyond the start day, then append 'nan'.
        if len(labels) < window_size:
            result.append(np.nan)
            continue

        # Arrays with both 'up' and 'down' symbols added to the next day.
        down_day: np.ndarray = np.append(labels, ["-"])
        up_day: np.ndarray = np.append(labels, ["+"])

        # Find occurrences of both patterns.
        down_occurrences: int = find_occurrences(training_labels, down_day)
        up_occurrences: int = find_occurrences(training_labels, up_day)

        # Edge case: N^+(s) = N^−(s) = 0. Use default probability to compute.
        if up_occurrences == 0 and down_occurrences == 0:
            if default_probability >= 0.5:
                result.append("+")
            else:
                result.append("-")
        else:
            if up_occurrences >= down_occurrences:
                result.append("+")
            else:
                result.append("-")

    return result


def prediction_accuracy(
    df: DataFrame, column: str, symbols: tuple = ("+", "-")
) -> None:
    """Compute the prediction accuracy of the column compared to the truth column.

    Args:
        df (DataFrame): The dataframe to use.
        column (str): The column to check against the truth column.
        symbols (tuple): The symbols to verify with.
    """
    # Get all values of the provided column. Remove the 'nan' rows in the beginning.
    col_df = (
        df.dropna(subset=[TRUE_LABEL_KEY, column])
        .apply(
            lambda d: operator.and_(
                d[TRUE_LABEL_KEY] == d[column], d[column] in symbols
            ),
            axis=1,
        )
        .value_counts()
    )

    accurate_values: int = col_df[True]
    inaccurate_values: int = col_df[False]

    if len(symbols) == 1:
        column_name: str = f"{column}{symbols[0]}*"
    else:
        column_name: str = f"{column}*"

    # Save the accuracy into the dataframe's attrs.
    df.attrs[column_name] = compute_accuracy(
        accurate_values=accurate_values,
        total_values=accurate_values + inaccurate_values,
    )


def compare_truth_to_column(
    df: DataFrame, column: str, truth_symbol: str, prediction_symbol: str
) -> Series:
    """Compare a column's rows to the truth column.

    Args:
        df (DataFrame): The dataframe to compare from.
        column (str): The column to compare with.
        truth_symbol (str): The symbol of the truth label.
        prediction_symbol (str): The symbol of the prediction.

    Returns:
        Series: The row where the truth column matches the prediction symbol.
    """
    return df.loc[
        operator.and_(
            df[TRUE_LABEL_KEY] == truth_symbol, df[column] == prediction_symbol
        )
    ]


def get_true_positives(df: DataFrame, column: str) -> Series:
    """Get the TP (true positives) value of the column.

    Args:
        df (DataFrame): The data frame to get the true positives from.
        column (str): The column to compare from.

    Returns:
        Series: A row of the true positives.
    """
    return compare_truth_to_column(df, column, "+", "+")


def get_false_positives(df: DataFrame, column: str) -> Series:
    """Get the FP (false positives) value of the column.

    Args:
        df (DataFrame): The data frame to get the false positives from.
        column (str): The column to compare from.

    Returns:
        Series: A row of the false positives.
    """
    return compare_truth_to_column(df, column, "-", "+")


def get_true_negatives(df: DataFrame, column: str) -> Series:
    """Get the TN (true negatives) value of the column.

    Args:
        df (DataFrame): The data frame to get the true negatives from.
        column (str): The column to compare from.

    Returns:
        Series: A row of the true negatives.
    """
    return compare_truth_to_column(df, column, "-", "-")


def get_false_negatives(df: DataFrame, column: str) -> Series:
    """Get the FN (false negatives) value of the column.

    Args:
        df (DataFrame): The data frame to get the false negatives from.
        column (str): The column to compare from.

    Returns:
        Series: A row of the false negatives.
    """
    return compare_truth_to_column(df, column, "+", "-")


def get_statistics(df: DataFrame, column: str, accuracy: float, ticker: str) -> Series:
    """Get a new data row filled with statistics about predictions.

    Args:
        df (DataFrame): The dataframe to compute statistics from.
        column (str): The column to compute the statistics from.
        accuracy (float): The accuracy value of the particular column.
        ticker (str): The stock ticker to save in the table.

    Returns:
        Series: A row of statistics to add to the statistics table.
    """
    true_positives: int = len(get_true_positives(df, column).index)
    false_positives: int = len(get_false_positives(df, column).index)
    true_negatives: int = len(get_true_negatives(df, column).index)
    false_negatives: int = len(get_false_negatives(df, column).index)

    return Series(
        {
            W_KEY: column,
            TICKER_KEY: ticker,
            TP_KEY: true_positives,
            FP_KEY: false_positives,
            TN_KEY: true_negatives,
            FN_KEY: false_negatives,
            ACCURACY_KEY: accuracy,
            TPR: true_positives / (true_positives + false_negatives),
            TNR: true_negatives / (true_negatives + false_positives),
        }
    )


def w_prediction_accuracy(
    df: DataFrame, window_list: list[int], symbols: tuple = ("+", "-")
) -> None:
    """Predict accuracy for each w in the window list.

    Args:
        df (DataFrame): The dataframe to predict accuracy from.
        window_list (list[int]): The list of w values to use.
        symbols (list[int]): The symbols to match for predictions.
    """
    for w in window_list:
        prediction_accuracy(df, f"W{w}", symbols)


def buy_and_hold(df: DataFrame, column: str, initial_investment: float = 100.0) -> None:
    """Buy stocks and hold until the end of the dataset.

    Compute daily returns and store them in the column provided.

    Args:
        df (DataFrame): The dataset to buy stocks from.
        column (str): The column to store the results in.
        initial_investment (float): The initial investment to start with.
    """
    df[column] = np.nan
    for i, row in df.iterrows():
        if i == 0:
            df.loc[i, column] = initial_investment
        else:
            current_investment = df.loc[i - 1, column]
            df.loc[i, column] = current_investment + (
                df.loc[i - 1, RETURN_KEY] * current_investment
            )


def buy_from_prediction(
    df: DataFrame,
    column: str,
    prediction_column: str,
    initial_investment: float = 100.0,
) -> None:
    """Buy stocks using the predictions in the provided prediction column.

    Args:
        df (DataFrame): The dataset to buy stocks from.
        column (str): The column to store the results in.
        prediction_column (str): The prediction column to compare to the truth column.
        initial_investment (float): The initial investment to start with.
    """
    df[column] = np.nan
    for i, row in df.iterrows():
        if i == 0:
            df.loc[i, column] = initial_investment
        else:
            current_investment = df.loc[i - 1, column]

            if df.loc[i, prediction_column] == "+":
                current_investment = current_investment + (
                    df.loc[i, RETURN_KEY] * current_investment
                )

            df.loc[i, column] = current_investment


def linechart(df: DataFrame) -> None:
    """Create a line chart of the dataframe.

    Args:
        df (DataFrame): The dataframe to create a line chart from.
    """
    axes = sns.lineplot(
        data=df[[DATE_KEY, BUY_AND_HOLD_KEY, ENSEMBLE_ORACLE_KEY, W_ORACLE]].set_index(
            DATE_KEY
        )
    )
    axes.xaxis.set_major_formatter(default_datefmt)
    axes.set_ylabel("Investment")
    plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
    axes.yaxis.set_major_formatter(currency_fmt)
    plt.xticks(rotation=25)
    plt.show()
# endregion


# region Helper methods for running methods for both stocks.
def read_stocks() -> tuple[DataFrame, DataFrame]:
    """Read each stocks' ticker file into a dataframe."""
    return pd.read_csv(sony_ticker_file), pd.read_csv(spy_ticker_file)


def get_years(df1: DataFrame, df2: DataFrame) -> tuple[list[int], list[int]]:
    """Get the unique years for each stock dataset.

    Args:
        df1 (DataFrame): The first dataset to get the years from.
        df2 (DataFrame): The second dataset to get the years from.

    Returns:
        tuple[list[int], list[int]]: The tuple of years for each stock dataset.
    """
    return df_unique_years(df1), df_unique_years(df2)


def get_training_tables(
    df1: tuple[DataFrame, list[int]], df2: tuple[DataFrame, list[int]]
) -> tuple[DataFrame, DataFrame]:
    """Get the datasets for training, aka years 3, 4, and 5 for both input dataframes.

    Args:
        df1 (DataFrame): The first dataframe to get the training dataset from.
        df2 (DataFrame): The second dataframe to get the training dataset from.

    Returns:
        tuple[DataFrame, DataFrame]: A tuple with both datasets.
    """
    return get_training_data(df1[0], df1[1]), get_training_data(df2[0], df2[1])


def get_testing_tables(
    df1: tuple[DataFrame, list[int]], df2: tuple[DataFrame, list[int]]
) -> tuple[DataFrame, DataFrame]:
    """Get the datasets for testing, aka years 4 and 5 for both input dataframes.

    Args:
        df1 (DataFrame): The first dataframe to get the testing dataset from.
        df2 (DataFrame): The second dataframe to get the testing dataset from.

    Returns:
        tuple[DataFrame, DataFrame]: A tuple with both datasets.
    """
    return get_testing_data(df1[0], df1[1]), get_testing_data(df2[0], df2[1])


def get_both_up_days(df1: DataFrame, df2: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Get the up days of both dataframes.

    Args:
        df1 (DataFrame): The first dataframe to get the up days from.
        df2 (DataFrame): The second dataframe to get the up days from.

    Returns:
        tuple[DataFrame, DataFrame]: A tuple with both dataframes.
    """
    return get_up_days(df1), get_up_days(df2)


def get_both_down_days(df1: DataFrame, df2: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Get the down days of both dataframes.

    Args:
        df1 (DataFrame): The first dataframe to get the down days from.
        df2 (DataFrame): The second dataframe to get the down days from.

    Returns:
        tuple[DataFrame, DataFrame]: A tuple with both dataframes.
    """
    return get_down_days(df1), get_down_days(df2)


def question_1_3(k_consecutive_days: list[int], df: DataFrame) -> None:
    """Wrap question 1.3.

    Compute the probability from the 'k' consecutive days.

    Args:
        k_consecutive_days (list[int]): The 'k' count of consecutive days.
        df (DataFrame): The dataframe to use.
    """
    for k in k_consecutive_days:
        probability: float = consecutive_day_probability(
            k_days=k, labels=df, consecutive_label=DayLabel.DOWN, next_label=DayLabel.UP
        )
        print(f"Probability for k = {k}: {probability}")
    print("\n")


def question_1_4(k_consecutive_days: list[int], df: DataFrame) -> None:
    """Wrap question 1.4.

    Compute the probability from the 'k' consecutive days.

    Args:
        k_consecutive_days (list[int]): The 'k' count of consecutive days.
        df (DataFrame): The dataframe to use.
    """
    for k in k_consecutive_days:
        probability: float = consecutive_day_probability(
            k_days=k, labels=df, consecutive_label=DayLabel.UP, next_label=DayLabel.UP
        )
        print(f"Probability for k = {k}: {probability}")
    print("\n")
# endregion


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

    # TODO Don't have time.
    pd.set_option("mode.chained_assignment", None)

    # Read the stock data files into pandas dataframes.
    sony_dataframe, spy_dataframe = cast(tuple[DataFrame, DataFrame], read_stocks())

    # Set the date column to be datetime objects instead of string.
    sony_dataframe[DATE_KEY] = pd.to_datetime(sony_dataframe[DATE_KEY])
    spy_dataframe[DATE_KEY] = pd.to_datetime(spy_dataframe[DATE_KEY])

    # region QUESTION 1.

    # Add 'True Label' column based upon the daily return.
    add_true_labels(sony_dataframe)
    add_true_labels(spy_dataframe)

    # Get a list of all the years in the dataset.
    sony_years, spy_years = cast(
        tuple[list[int], list[int]], get_years(sony_dataframe, spy_dataframe)
    )

    # Separate data into training and testing data based upon the years.
    # Create training dataset from years 1, 2, and 3.
    sony_training_table, spy_training_table = cast(
        tuple[DataFrame, DataFrame],
        get_training_tables((sony_dataframe, sony_years), (spy_dataframe, spy_years)),
    )

    # Create testing dataset from years 4 and 5.
    sony_testing_table, spy_testing_table = cast(
        tuple[DataFrame, DataFrame],
        get_testing_tables((sony_dataframe, sony_years), (spy_dataframe, spy_years)),
    )

    # Get the 'up days' for the training datasets.
    sony_training_up_days, spy_training_up_days = cast(
        tuple[DataFrame, DataFrame],
        get_both_up_days(sony_training_table, spy_training_table),
    )

    # Compute the default probability that the next day is an 'up' day.
    sony_default_probability: float = compute_probability(
        len(sony_training_up_days), len(sony_training_table)
    )

    spy_default_probability: float = compute_probability(
        len(spy_training_up_days), len(spy_training_table)
    )

    # Get arrays of the training labels.
    sony_training_labels: np.ndarray = get_labels(sony_training_table)
    spy_training_labels: np.ndarray = get_labels(spy_training_table)

    k_list: list[int] = [1, 2, 3]

    print("Question 1.3 with SONY dataset:")
    question_1_3(k_list, sony_training_labels)

    print("Question 1.3 with S&P-500 dataset:")
    question_1_3(k_list, spy_training_labels)

    print("Question 1.4 with SONY dataset:")
    question_1_4(k_list, sony_training_labels)

    print("Question 1.4 with S&P-500 dataset:")
    question_1_4(k_list, spy_training_labels)

    print("Finished Question 1!")
    # endregion

    # region QUESTION 2.

    # Get arrays of the testing labels.
    sony_testing_labels: np.ndarray = get_labels(sony_testing_table)
    spy_testing_labels: np.ndarray = get_labels(spy_testing_table)

    w_list: list[int] = [2, 3, 4]

    # pd.set_option('mode.chained_assignment','raise')
    # For W = 2, 3, 4: Predict the next day and add it to the testing table.
    for w in w_list:
        # SONY dataset predictions.
        sony_testing_table[f"W{w}"] = predict_next_day(
            df=sony_testing_table,
            training_labels=sony_testing_labels,
            window_size=w,
            default_probability=sony_default_probability,
        )

        # S&P-500 dataset predictions.
        spy_testing_table[f"W{w}"] = predict_next_day(
            df=spy_testing_table,
            training_labels=spy_testing_labels,
            window_size=w,
            default_probability=spy_default_probability,
        )

    w_accuracy_keys: list[str] = [f"W{w}*" for w in w_list]

    # Compute the prediction accuracy for SONY.
    w_prediction_accuracy(sony_testing_table, w_list)
    sony_accuracies = {key: sony_testing_table.attrs[key] for key in w_accuracy_keys}

    # Compute the prediction accuracy for S&P-500.
    w_prediction_accuracy(spy_testing_table, w_list)
    spy_accuracies = {key: spy_testing_table.attrs[key] for key in w_accuracy_keys}

    print("SONY prediction accuracies: ")
    for accuracy, value in sony_accuracies.items():
        print(f"{accuracy}: {value}")

    print("S&P-500 prediction accuracies: ")
    for accuracy, value in spy_accuracies.items():
        print(f"{accuracy}: {value}")

    # Get the highest accuracy value for SONY.
    sony_highest_accuracy: tuple[str, float] = max(
        sony_accuracies.items(), key=operator.itemgetter(1)
    )
    print(
        f"SONY highest accuracy is {sony_highest_accuracy[0]} with an accuracy of "
        f"{sony_highest_accuracy[1]}."
    )

    # Get the highest accuracy value for S&P-500.
    spy_highest_accuracy: tuple[str, float] = max(
        spy_accuracies.items(), key=operator.itemgetter(1)
    )
    print(
        f"S&P-500 highest accuracy is {spy_highest_accuracy[0]} with an accuracy of "
        f"{spy_highest_accuracy[1]}."
    )

    print("Finished Question 2!")

    # endregion

    # region QUESTION 3
    w_cols: list[str] = ["W2", "W3", "W4"]

    # Compute ensemble labels for year 4 and 5 of SONY.
    compute_ensemble(sony_testing_table, w_cols)
    sony_ensemble_row: Series = sony_testing_table[ENSEMBLE_KEY].transpose()
    print(sony_ensemble_row)

    # Compute ensemble labels for year 4 and 5 of S&P-500.
    compute_ensemble(spy_testing_table, w_cols)
    spy_ensemble_row: Series = spy_testing_table[ENSEMBLE_KEY].transpose()
    print(spy_ensemble_row)

    # Compute the accuracy of the ensemble column for SONY.
    prediction_accuracy(sony_testing_table, ENSEMBLE_KEY)
    sony_ensemble_accuracy: float = sony_testing_table.attrs[f"{ENSEMBLE_KEY}*"]
    print(f"Ensemble accuracy for SONY: {float_to_percentage(sony_ensemble_accuracy)}")

    # Compute the accuracy of the ensemble column for S&P-500.
    prediction_accuracy(spy_testing_table, ENSEMBLE_KEY)
    spy_ensemble_accuracy: float = spy_testing_table.attrs[f"{ENSEMBLE_KEY}*"]
    print(
        f"Ensemble accuracy for S&P-500: {float_to_percentage(spy_ensemble_accuracy)}"
    )

    w_negative_keys: list[str] = [f"W{w}-*" for w in w_list]

    # Compute the prediction accuracy for negatives in SONY.
    w_prediction_accuracy(sony_testing_table, w_list, ("-",))
    sony_negative_accuracies = {
        key: sony_testing_table.attrs[key] for key in w_negative_keys
    }
    prediction_accuracy(sony_testing_table, ENSEMBLE_KEY, ("-",))

    # Compare '-' accuracies.
    sony_ensemble_negative_accuracy: float = sony_testing_table.attrs[
        f"{ENSEMBLE_KEY}-*"
    ]
    sony_w_negative_accuracy: float = mean(list(sony_negative_accuracies.values()))

    print(f"SONY Ensemble accuracy for '-': {sony_ensemble_negative_accuracy}")
    print(f"SONY W accuracy for '-': {sony_w_negative_accuracy}")
    if sony_ensemble_negative_accuracy >= sony_w_negative_accuracy:
        print("Ensemble for SONY had better accuracy than W for negative '-'.")
    else:
        print("W for SONY had better accuracy than Ensemble for negative '-'.")

    w_positive_keys: list[str] = [f"W{w}+*" for w in w_list]

    # Compute the prediction accuracy for positives in SONY.
    w_prediction_accuracy(sony_testing_table, w_list, ("+",))
    sony_positive_accuracies = {
        key: sony_testing_table.attrs[key] for key in w_positive_keys
    }
    prediction_accuracy(sony_testing_table, ENSEMBLE_KEY, ("+",))

    # Compare '+' accuracies.
    sony_ensemble_positive_accuracy: float = sony_testing_table.attrs[
        f"{ENSEMBLE_KEY}+*"
    ]
    sony_w_positive_accuracy: float = mean(list(sony_positive_accuracies.values()))

    print(f"SONY Ensemble accuracy for '+': {sony_ensemble_positive_accuracy}")
    print(f"SONY W accuracy for '+': {sony_w_positive_accuracy}")
    if sony_ensemble_positive_accuracy >= sony_w_positive_accuracy:
        print("Ensemble for SONY had better accuracy than W for positive '+'.")
    else:
        print("W for SONY had better accuracy than Ensemble for positive '+'.")

    print("Finished Question 3!")
    # endregion

    # region QUESTION 4
    w_keys: list[str] = ["W2", "W3", "W4", ENSEMBLE_KEY]

    # Get statistics for all W labels in S&P-500 stock data table.
    spy_statistics_table: DataFrame = DataFrame(
        [
            get_statistics(
                df=spy_testing_table,
                column=w_key,
                accuracy=spy_testing_table.attrs[f"{w_key}*"],
                ticker=SPY_TICKER,
            )
            for w_key in w_keys
        ]
    )

    # Get statistics for all W labels in SONY stock data table.
    sony_statistics_table: DataFrame = DataFrame(
        [
            get_statistics(
                df=sony_testing_table,
                column=w_key,
                accuracy=sony_testing_table.attrs[f"{w_key}*"],
                ticker=SONY_TICKER,
            )
            for w_key in w_keys
        ]
    )

    statistics_table: DataFrame = pd.concat(
        [spy_statistics_table, sony_statistics_table]
    ).reset_index(drop=True)

    print("Finished Question 4!")
    # endregion

    # region QUESTION 5
    initial_investment: float = 100.0

    # Buy and hold for SONY.
    buy_and_hold(sony_testing_table, column=BUY_AND_HOLD_KEY)

    # Buy from the predictions made by the ensemble method.
    buy_from_prediction(
        sony_testing_table, column=ENSEMBLE_ORACLE_KEY, prediction_column=ENSEMBLE_KEY
    )

    # Buy from the predictions made by the best W* method.
    buy_from_prediction(
        sony_testing_table,
        column=W_ORACLE,
        prediction_column=sony_highest_accuracy[0].replace("*", ""),
    )

    # Display linechart for SONY.
    linechart(sony_testing_table)

    # Buy and hold for S&P-500.
    buy_and_hold(spy_testing_table, column=BUY_AND_HOLD_KEY)

    # Buy from the predictions made by the ensemble method.
    buy_from_prediction(
        spy_testing_table, column=ENSEMBLE_ORACLE_KEY, prediction_column=ENSEMBLE_KEY
    )

    # Buy from the predictions made by the best W* method.
    buy_from_prediction(
        spy_testing_table,
        column=W_ORACLE,
        prediction_column=spy_highest_accuracy[0].replace("*", ""),
    )

    # Display linechart for S&P-500.
    linechart(spy_testing_table)
    # endregion
