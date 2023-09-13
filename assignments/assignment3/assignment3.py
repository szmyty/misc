"""Assignment3 module.

Alan Szmyt
Class: CS 677
Date: April 4th, 2023
Assignment #3
Description:
This assignment contains code comparing a simple classifier, k-Nearest Neighbors classifier, and logistic regression
for classifying banknotes to be legitimate or counterfeit.
"""
from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from logging import DEBUG, Logger, StreamHandler, getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constants import (
    BANKNOTE_DATASET,
    BANKNOTES,
    COL_ALL,
    COL_CLASS,
    COL_COLOR,
    COL_F1,
    COL_F2,
    COL_F3,
    COL_F4,
    COL_PREDICTION,
    FAKE_BILLS,
    FAKE_BILLS_RED,
    FEATURE_COLS,
    GOOD_BILLS,
    GOOD_BILLS_GREEN,
    INITIAL_COLS,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
    features_to_plot,
)
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from utils import (
    Color,
    TruthStats,
    artifacts,
    get_truth_statistics,
    get_truth_statistics_df,
    load_df_from_checkpoints,
    relative_path_to,
    rename_columns,
    resources,
    save_df_as_figure,
    save_df_to_checkpoints,
    save_figures,
)

# Configure a logger to log statements.
logger: Logger = getLogger(__file__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# Create a new test dataset for BUID.
bu_id_df: DataFrame = DataFrame(
    {
        COL_F1: Series([3.0], dtype=np.float64),
        COL_F2: Series([0.0], dtype=np.float64),
        COL_F3: Series([6.0], dtype=np.float64),
        COL_F4: Series([8.0], dtype=np.float64),
    }
)


class BankNote(IntEnum):
    """Enum class of banknote classifications."""

    LEGITIMATE: int = 0
    COUNTERFEIT: int = 1


def save_split_datasets(
    x_train: DataFrame,
    x_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
) -> None:
    """Save the split datasets to the 'checkpoints' directory.

    Args:
        x_train (DataFrame): The training dataset of features.
        x_test (DataFrame): The testing dataset of features.
        y_train (DataFrame): The training dataset of labels.
        y_test (DataFrame): The testing labels to compare to for accuracy.
    """
    save_df_to_checkpoints(x_train, X_TRAIN)
    save_df_to_checkpoints(x_test, X_TEST)
    save_df_to_checkpoints(y_train, Y_TRAIN)
    save_df_to_checkpoints(y_test, Y_TEST)


def load_split_datasets() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load the split datasets from the 'checkpoints' directory.

    Loads 'x_train', 'x_test', 'y_train', and 'y_test'.

    Returns:
        tuple: A tuple of the datasets.
    """
    return (
        load_df_from_checkpoints(X_TRAIN),
        load_df_from_checkpoints(X_TEST),
        load_df_from_checkpoints(Y_TRAIN),
        load_df_from_checkpoints(Y_TEST),
    )


def feature_stats_table(df: DataFrame) -> DataFrame:
    """Compute statistics about the features in the dataset.

    Args:
        df (DataFrame): The dataframe to compute statistics on.

    Returns:
        DataFrame: A dataframe table with the statistics results.
    """
    # Compute mean and standard deviation for each feature per class.
    grouped_df: DataFrameGroupBy = df.groupby(COL_CLASS)[FEATURE_COLS]
    grouped_stats: DataFrame = grouped_df.agg(["mean", "std"]).round(2)

    # Flatten the MultiIndex and join the column labels.
    grouped_stats.columns = ["_".join(col) for col in grouped_stats.columns.values]

    # Compute overall mean and standard deviation for combined both classes.
    stats: DataFrame = df.drop(columns=COL_CLASS).agg(["mean", "std"]).round(2)

    # Reshape the series into the same shape as the grouped dataframe.
    stats = stats.stack().to_frame(name=COL_ALL).transpose()
    stats.columns = ["_".join(col[::-1]) for col in stats.columns.values]

    # Concatenate the data frames for the final stats table.
    return pd.concat(grouped_stats.align(stats, join="outer", axis=1))


def simple_classifier(row: Series) -> int:
    """Simple classifier to predict if banknotes are counterfeit or legitimate.

    Args:
        row (Series): The row of the dataframe to predict the class label for.

    Returns:
        int: 0 if the label is legitimate, 1 if the label is fake.
    """
    if row[COL_F1] < 4 and row[COL_F2] < 10 < row[COL_F3]:
        return BankNote.COUNTERFEIT.value
    else:
        return BankNote.LEGITIMATE.value


def knn(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, k: int) -> np.array:
    """Make a prediction on the provided datasets using knn classifier.

    Args:
        x_train (DataFrame): The training dataset of features.
        x_test (DataFrame): The testing dataset of features.
        y_train (DataFrame): The training dataset of labels.
        k (int): The 'k' value to use for knn classifier.

    Returns:
        np.array: The array of prediction values.
    """
    # Feature scaling, generally necessary for kNN.
    scaler: StandardScaler = StandardScaler()
    x_train_sc: np.ndarray = scaler.fit_transform(x_train)
    x_test_sc: np.ndarray = scaler.transform(x_test)

    # k-Nearest Neighbor classifier for the dataframe.
    classifier: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=k, p=2, metric="euclidean"
    )
    classifier.fit(x_train_sc, y_train)

    # Return the classifier's prediction.
    return classifier.predict(x_test_sc)


def knn_accuracy(
    x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame, k: int
) -> float:
    """Compute accuracy of knn prediction.

    Args:
        x_train (DataFrame): The training dataset of features.
        x_test (DataFrame): The testing dataset of features.
        y_train (DataFrame): The training dataset of labels.
        y_test (DataFrame): The testing labels to compare to for accuracy.
        k (int): The 'k' value to use for knn classifier.

    Returns:
        float: The accuracy of the predictions compared to the truth labels.
    """
    return accuracy_score(y_test, knn(x_train, x_test, y_train, k))


def log_reg_accuracy(
    x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame
) -> float:
    """Compute accuracy of a logistic regression prediction.

    Args:
        x_train (DataFrame): The training dataset of features.
        x_test (DataFrame): The testing dataset of features.
        y_train (DataFrame): The training dataset of labels.
        y_test (DataFrame): The testing labels to compare to for accuracy.

    Returns:
        float: The accuracy of the logistic regression predictions.
    """
    # Train on the dataset using logistic regression classifier.
    logistic_regression: LogisticRegression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)
    return accuracy_score(y_test, logistic_regression.predict(x_test))


def question1() -> None:
    """Wrap question1 in a method."""
    # dtype mapping to use for the csv file. Set class to categorical.
    dtypes = defaultdict(np.float64, {COL_CLASS: pd.CategoricalDtype.name})

    # Loading the data_banknote_authentication.txt file into a dataframe.
    banknote_dataset: DataFrame = pd.read_csv(
        banknote_dataset_file, header=None, names=INITIAL_COLS, dtype=dtypes
    )

    # Add color column to dataframe based upon the class column.
    banknote_dataset[COL_COLOR] = banknote_dataset.apply(
        lambda row: Color.GREEN.lower
        if int(row[COL_CLASS]) == BankNote.LEGITIMATE
        else Color.RED.lower,
        axis=1,
    ).astype("category")

    # Compute the feature distribution stats table.
    feature_stats: DataFrame = feature_stats_table(
        banknote_dataset.drop(columns=COL_COLOR)
    )

    print(f"Banknote feature stats table:\n {feature_stats}")

    save_df_as_figure(feature_stats, "banknote_feature_stats")

    # Saving banknote dataset for checkpoint.
    save_df_to_checkpoints(banknote_dataset, filename=BANKNOTE_DATASET)

    print("Finished Question 1!")


def question2() -> None:
    """Wrap question2 in a method."""
    # Read the data from previous question.
    banknote_dataset: DataFrame = load_df_from_checkpoints(BANKNOTE_DATASET)

    # Split the dataset into training/testing datasets split 50/50.
    y: Series = banknote_dataset[COL_CLASS]
    x_train, x_test, y_train, y_test = train_test_split(
        banknote_dataset, y, test_size=0.5, random_state=42, stratify=y
    )

    # Training subset of rows with legitimate banknotes.
    train_good_bills: DataFrame = x_train.loc[
        x_train[COL_CLASS].astype(int) == BankNote.LEGITIMATE
    ]

    # Training subset of rows with counterfeit banknotes.
    train_bad_bills: DataFrame = x_train.loc[
        x_train[COL_CLASS].astype(int) == BankNote.COUNTERFEIT
    ]

    # Create, save, and plot a pairwise plot of the all bills.
    train_pairwise = sns.pairplot(
        rename_columns(x_train, features_to_plot),
        hue="color",
        palette=[GOOD_BILLS_GREEN, FAKE_BILLS_RED],
    )
    train_pairwise.fig.suptitle("Banknotes Pairwise Plot")
    banknote_figures: list[Path] = save_figures(BANKNOTES, artifacts)
    for figure in banknote_figures:
        print(f"Saved banknotes plot to {figure}")
    plt.show()

    # Create, save, and plot a pairwise plot of the good bills.
    good_pairwise = sns.pairplot(
        rename_columns(train_good_bills, features_to_plot),
        hue="color",
        palette=[GOOD_BILLS_GREEN, FAKE_BILLS_RED],
    )
    good_pairwise.fig.suptitle("Legitimate Bills Pairwise Plot")
    good_bill_figures: list[Path] = save_figures(GOOD_BILLS, artifacts)
    for figure in good_bill_figures:
        print(f"Saved legitimate bill plot to {figure}")
    plt.show()

    # Create, save, and plot a pairwise plot of the fake bills.
    fake_pairwise = sns.pairplot(
        rename_columns(train_bad_bills, features_to_plot),
        hue="color",
        palette=[GOOD_BILLS_GREEN, FAKE_BILLS_RED],
    )
    fake_pairwise.fig.suptitle("Counterfeit Bills Pairwise Plot")
    fake_bill_figures: list[Path] = save_figures(FAKE_BILLS, artifacts)
    for figure in fake_bill_figures:
        print(f"Saved counterfeit bill plot to {figure}")
    plt.show()

    # Apply the simple classifier to predict class labels.
    x_test[COL_PREDICTION] = x_test.apply(simple_classifier, axis=1)

    # Compute the truth statistics for the simple classifier.
    truth_statistics: DataFrame = get_truth_statistics(
        x_test, COL_CLASS, COL_PREDICTION
    )

    true_positives: int = truth_statistics[TruthStats.TRUE_POSITIVE.value].iloc[0]
    print(f"True positives: {true_positives}")

    false_positives: int = truth_statistics[TruthStats.FALSE_POSITIVE.value].iloc[0]
    print(f"False positives: {false_positives}")

    true_negatives: int = truth_statistics[TruthStats.TRUE_NEGATIVE.value].iloc[0]
    print(f"True negatives: {true_negatives}")

    false_negatives: int = truth_statistics[TruthStats.FALSE_NEGATIVE.value].iloc[0]
    print(f"False negatives: {false_negatives}")

    true_positive_rate: int = truth_statistics[
        TruthStats.TRUE_POSITIVE_RATE.value
    ].iloc[0]
    print(f"True positive rate: {true_positive_rate}")

    true_negative_rate: int = truth_statistics[
        TruthStats.TRUE_NEGATIVE_RATE.value
    ].iloc[0]
    print(f"True negative rate: {true_negative_rate}")

    simple_accuracy: float = truth_statistics.iloc[0][TruthStats.ACCURACY.value] * 100
    print(f"Simple classifier's accuracy: {simple_accuracy}")
    if simple_accuracy >= 50:
        print("Accuracy for simple classifier is better than a coin flip!")
    else:
        print("Accuracy for simple classifier is worse than a coin flip!")

    print(f"Simple classifier truth statistics table:\n {truth_statistics}")

    save_df_as_figure(truth_statistics, "simple_truth_classifier_stats")
    print("Finished Question 2!")


def question3() -> None:
    """Wrap question3 in a method."""
    # Load the same training and test dataset from question 2.
    banknote_dataset: DataFrame = load_df_from_checkpoints(BANKNOTE_DATASET)

    # n_neighbor values to try with the kNN classifier.
    k_vals: list[int] = list(range(3, 12, 2))

    # Split the dataset into training/testing datasets split 50/50.
    y: Series = banknote_dataset[COL_CLASS]
    x_train, x_test, y_train, y_test = train_test_split(
        banknote_dataset.drop([COL_COLOR, COL_CLASS], axis=1),
        y,
        test_size=0.5,
        random_state=42,
        stratify=y,
    )

    # Save split datasets for later use.
    save_split_datasets(x_train, x_test, y_train, y_test)

    # Feature scaling, generally necessary for kNN.
    scaler: StandardScaler = StandardScaler()
    x_train_sc: np.ndarray = scaler.fit_transform(x_train)
    x_test_sc: np.ndarray = scaler.transform(x_test)

    # Save the accuracies for each n_neighbor value.
    knn_accuracies: dict[int, float] = {}
    knn_error: dict[int, float] = {}
    for k_val in k_vals:
        # k-Nearest Neighbor classifier for classifying banknotes.
        classifier: KNeighborsClassifier = KNeighborsClassifier(
            n_neighbors=k_val, p=2, metric="euclidean"
        )
        classifier.fit(x_train_sc, y_train)

        # Use classifier to make predictions.
        y_pred: np.ndarray = classifier.predict(x_test_sc)

        # Compute the accuracy for k and save.
        knn_accuracies.update({k_val: accuracy_score(y_test, y_pred)})

        # Compute the error rate of k and save.
        # Reference: https://stackoverflow.com/a/62616556
        knn_error.update({k_val: np.mean(y_pred != y_test)})

    for k_val, knn_acc in knn_accuracies.items():
        print(f"Prediction accuracy for k={k_val}: {knn_acc}")

    # Plot a graph showing the accuracies of each k.
    plt.plot(k_vals, list(knn_accuracies.values()))
    plt.xticks(k_vals)
    plt.xlabel("k")
    plt.ylabel("Accuracy of kNN Predictions")
    plt.title("Plot of Accuracy vs k value for kNN")
    plt.gcf().subplots_adjust(bottom=0.2, left=0.18)

    # Save graph before showing.
    save_figures("knn_accuracies", artifacts)
    plt.show()

    # Error rate plot.
    plt.plot(k_vals, knn_error.values(), color="r", marker="o", markersize=9)
    plt.xlabel("k")
    plt.ylabel("Error rate of kNN Predictions")
    plt.title("Error rate for each k value for kNN")
    plt.show()

    print("Selecting optimal k value as k=9.")

    # Selecting k=9 to compute performance measures
    classifier: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=9, p=2, metric="euclidean"
    )
    classifier.fit(x_train_sc, y_train)
    y_pred: np.ndarray = classifier.predict(x_test_sc)

    # Create a confusion matrix to get metrics.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    cm: np.array = confusion_matrix(y_test, y_pred)

    # Create and show the confusion matrix plot.
    disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Confusion Matrix for kNN with k=9")
    save_figures("knn_confusion_matrix", artifacts)
    plt.show()

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    tn, fp, fn, tp = cm.ravel()

    # Create a truth statistics table from kNN metrics.
    knn_truth_statistics: DataFrame = get_truth_statistics_df(
        accuracy=accuracy_score(y_test, y_pred),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )

    save_df_as_figure(knn_truth_statistics, "knn_truth_statistics")

    # Scale this new dataset.
    bu_test_sc: np.ndarray = scaler.transform(bu_id_df)
    bu_pred_knn: np.ndarray = classifier.predict(bu_test_sc).astype(int)

    bu_bill_knn: str = "legitimate" if bu_pred_knn == 0 else "counterfeit"
    print(f"kNN predicted my BUID to be a {bu_bill_knn} banknote!")

    bu_pred_simple = bu_id_df.apply(simple_classifier, axis=1).iloc[0]
    bu_bill_simple: str = "legitimate" if bu_pred_simple == 0 else "counterfeit"
    print(f"Simple classifier predicted my BUID to be a {bu_bill_simple} banknote!")

    print("Finished Question 3!")


def question4() -> None:
    """Wrap question4 in a method."""
    # Load the same training and test dataset from question 2.
    x_train, x_test, y_train, y_test = load_split_datasets()

    # Compute the accuracy with optimal k when each feature is missing.
    k: int = 9
    f1_missing_accuracy: float = knn_accuracy(
        x_train=x_train.drop([COL_F1], axis=1),
        x_test=x_test.drop([COL_F1], axis=1),
        y_train=y_train,
        y_test=y_test,
        k=k,
    )
    print(f"kNN accuracy when f1 is missing " f"{f1_missing_accuracy}.")

    f2_missing_accuracy: float = knn_accuracy(
        x_train=x_train.drop([COL_F2], axis=1),
        x_test=x_test.drop([COL_F2], axis=1),
        y_train=y_train,
        y_test=y_test,
        k=k,
    )
    print(f"kNN accuracy when f2 is missing " f"{f2_missing_accuracy}.")

    f3_missing_accuracy: float = knn_accuracy(
        x_train=x_train.drop([COL_F3], axis=1),
        x_test=x_test.drop([COL_F3], axis=1),
        y_train=y_train,
        y_test=y_test,
        k=k,
    )
    print(f"kNN accuracy when f3 is missing " f"{f3_missing_accuracy}.")

    f4_missing_accuracy: float = knn_accuracy(
        x_train=x_train.drop([COL_F4], axis=1),
        x_test=x_test.drop([COL_F4], axis=1),
        y_train=y_train,
        y_test=y_test,
        k=k,
    )
    print(f"kNN accuracy when f4 is missing " f"{f4_missing_accuracy}.")


def question5() -> None:
    """Wrap question5 in a method."""
    # Load the same training and test dataset from question 2.
    x_train, x_test, y_train, y_test = load_split_datasets()

    # Train on the dataset using logistic regression classifier.
    logistic_regression: LogisticRegression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)
    y_pred: np.ndarray = logistic_regression.predict(x_test)

    accuracy: float = accuracy_score(y_test, y_pred)

    print(f"Accuracy for the logistic " f"regression classifier: {accuracy}")

    # Create a confusion matrix to get metrics.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    cm: np.array = confusion_matrix(y_test, y_pred)

    # Create and show the confusion matrix plot.
    disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title("Confusion Matrix for Logistic Regression")
    save_figures("logistic_regression", artifacts)
    plt.show()

    # Get truth stats from confusion matrix.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    tn, fp, fn, tp = cm.ravel()

    # Create a truth statistics table from logistic regression metrics.
    log_reg_truth_statistics: DataFrame = get_truth_statistics_df(
        accuracy=accuracy,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )

    save_df_as_figure(log_reg_truth_statistics, "log_reg_truth_statistics")

    # Predict BUID using logistic regression.
    bu_pred_log_reg: np.ndarray = logistic_regression.predict(bu_id_df).astype(int)

    bu_bill_log_reg: str = "legitimate" if bu_pred_log_reg == 0 else "counterfeit"
    print(f"Logistic regression predicted my BUID to be a {bu_bill_log_reg} banknote!")

    print("Finished Question 5!")


def question6() -> None:
    """Wrap question6 in a method."""
    # Load the same training and test dataset from question 2.
    x_train, x_test, y_train, y_test = load_split_datasets()

    # Compute the accuracy with when each feature is missing.
    f1_missing_accuracy: float = log_reg_accuracy(
        x_train=x_train.drop([COL_F1], axis=1),
        x_test=x_test.drop([COL_F1], axis=1),
        y_train=y_train,
        y_test=y_test,
    )
    print(f"logistic regression accuracy when f1 is missing " f"{f1_missing_accuracy}.")

    f2_missing_accuracy: float = log_reg_accuracy(
        x_train=x_train.drop([COL_F2], axis=1),
        x_test=x_test.drop([COL_F2], axis=1),
        y_train=y_train,
        y_test=y_test,
    )
    print(f"logistic regression when f2 is missing " f"{f2_missing_accuracy}.")

    f3_missing_accuracy: float = log_reg_accuracy(
        x_train=x_train.drop([COL_F3], axis=1),
        x_test=x_test.drop([COL_F3], axis=1),
        y_train=y_train,
        y_test=y_test,
    )
    print(f"logistic regression accuracy when f3 is missing " f"{f3_missing_accuracy}.")

    f4_missing_accuracy: float = log_reg_accuracy(
        x_train=x_train.drop([COL_F4], axis=1),
        x_test=x_test.drop([COL_F4], axis=1),
        y_train=y_train,
        y_test=y_test,
    )
    print(f"logistic regression accuracy when f4 is missing " f"{f4_missing_accuracy}.")


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

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

    # Banknote dataset file from UCI.
    banknote_dataset_file: Path = resources.joinpath("data_banknote_authentication.txt")

    print(f"Loading dataset from: {relative_path_to(cwd, banknote_dataset_file)}")

    question1()
    question2()
    question3()
    question4()
    question5()
    question6()

    print("Finished Assignment 3!")
