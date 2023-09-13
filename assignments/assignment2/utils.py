"""Utils module.

Alan Szmyt
Class: CS 677
Date: March 27th, 2023
Description:
This module contains a variety of utility methods used for any assignment. Separating
the methods out into this module improves readability.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, deque
from datetime import datetime, timezone
from itertools import islice
from logging import DEBUG, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from pandas._config import config  # noqa
from pandas._config.config import OptionError, is_bool  # noqa

# Configure a logger to log statements.
logger: Logger = getLogger(__file__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


# Directory path to the 'resources' folder.
resources: Path = Path(os.path.abspath("")).joinpath("resources").resolve()


def df_unique_list(df: DataFrame, column: Any) -> list[Any]:
    """Get a list of unique values in the specified column.

    Args:
        df (DataFrame): The dataframe to get the unique values from.
        column (Any): The column to get the values from.

    Returns:
        list[Any]: The list of unique values that were in the column.
    """
    return df[column].unique().tolist()


def compute_probability(occurrences: int, sample_space: int) -> float:
    """Compute probability based upon the occurrences of an event in the sample space.

    Args:
        occurrences (int): The count of occurrences of the event that is being computed.
        sample_space (int): The size of the total sample space.

    Returns:
        float: The computed probability.
    """
    return occurrences / sample_space


def split_array(arr: ndarray, cond: ndarray) -> tuple[ndarray, ndarray]:
    """Split ndarray into two arrays based upon condition.

    One array will be all items that returned True for the condition and the other
    array will be all items that returned False for the condition.

    Args:
        arr (ndarray): The array to split.
        cond (ndarray): The condition to split the array on.

    Returns:
        tuple[ndarray, ndarray]: A tuple of the split array.
    """
    return arr[cond], arr[~cond]


def save_df_to_resources(df: DataFrame, filename: str) -> None:
    """Save the provided dataframe to the 'resources' directory.

    Args:
        df (DataFrame): The dataframe to save.
        filename (str): The name of the file.
    """
    df.to_pickle(resources.joinpath(f"{filename}.pkl"))


def most_common(row: Series, columns: list[str]) -> Any:
    """Get the most common value amongst the provided columns in a series.

    Args:
        row (Series): The row to get the most common value from.
        columns (list[str]): The columns to compare for the most common.

    Returns:
        Any: The most common value.
    """
    return Counter(row[columns].tolist()).most_common(1)[0][0]


def sliding_window(
    sequence: Sequence[Any], size: int = 2, step: int = 1, fill: Any = None
) -> Iterator[Any]:
    """Slide over a sequence with the provided window size and step.

    Args:
        sequence (Sequence[Any]): The sequence to iterator over.
        size (int): The size of each window per iteration.
        step (int): The steps to take for each iteration.
        fill (Any): The value to fill inbetween.

    References:
        - https://stackoverflow.com/a/6822761
        - https://docs.python.org/release/2.3.5/lib/itertools-example.html

    Returns:
        Iterator[Any]: A iterator of windowed values.
    """
    if size < 0 or step < 1:
        raise ValueError
    it = iter(sequence)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fill for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fill) for _ in range(step - 1))


def nparray_tail(arr: np.array, num: int) -> np.array:
    """Get the last 'num' elements of the input array.

    Args:
        arr (np.array): The numpy array to get the tail of.
        num (int): The number of elements to get at the end of the array.

    References:
        - https://stackoverflow.com/a/69251674

    Returns:
        np.array: The last 'num' elements of the array.
    """
    if num == 0:
        return arr[0:0]
    else:
        return arr[-num:]


def windowed_iter(sequence: Sequence[Any], window_size: int = 2) -> Iterator[Any]:
    """Iterate over a sequence a yield a 'window'.

    Args:
        sequence (Sequence[Any]): The sequence to iterate over.
        window_size (int): The window size to yield for each iteration.

    Yields:
        Iterator[Any]: A windowed iterator.
    """
    iterator: Iterator = iter(sequence)
    window: deque = deque(
        (next(iterator, None) for _ in range(window_size)), maxlen=window_size
    )

    yield window
    for element in iterator:
        window.append(element)
        yield window


def clean_string(string: str) -> str:
    """Clean the provided string.

    Args:
        string (str): The string to clean.

    Returns:
        str: The cleaned string.
    """
    string = string.strip()
    string = re.sub("[!#?]", "", string)
    return string.title()


def clean_strings(strings: list[str]) -> list[str]:
    """Clean each string in the provided list.

    References:
        - Book: Python for Data Analysis Data Wrangling with pandas, NumPy & Jupyter

    Args:
        strings (list[str]): The list of strings to iterate over.

    Returns:
        list[str]: A list of the cleaned strings.
    """
    return [clean_string(s) for s in strings]


def substr_occurrences(string: str, substring: str) -> int:
    """Count the occurrences of a substring inside the provide string (overlapping).

    Args:
        string (str): The string to search for substrings over.
        substring (str): The substring to search for.

    Returns:
        int: The count of occurrences of the substring.
    """
    count = 0
    start = 0

    while start < len(string):
        pos = string.find(substring, start)
        if pos != -1:
            start = pos + 1
            count += 1
        else:
            break
    return count


def table_map_per_col_value(
    data: DataFrame, column: str, values: list[Any]
) -> dict[int, list[DataFrame]]:
    """Map a column per value.

    Args:
        data (DataFrame): The dataframe to map from.
        column (str): The column to look in.
        values (list[Any]): The values to map.

    Returns:
        dict[int, list[DataFrame]]: A map of the column per value.
    """

    return {val: data.loc[data[column] == val] for val in values}


def find_occurrences(dataset: ndarray[str], pattern: ndarray[str]) -> int:
    """Find the count of occurrences of the subarray in the provided array.

    Uses a sliding window to search for the subarray occurrences.
    For now, the data type needs to be a string (dtype="U1").

    Args:
        dataset (ndarray[str]): The array to search for the subarray over.
        pattern (ndarray[str]): The subarray to search for.

    Returns:
        int: The count of occurrences of the subarray in the array.
    """
    occurrences: int = 0
    for w in sliding_window(dataset, size=len(pattern)):
        w_pattern: np.ndarray = np.fromiter(w, dtype="U1")
        if np.array_equal(pattern, w_pattern):
            occurrences = occurrences + 1
    return occurrences


def split_df_with_overlap(df: DataFrame, window_size: int = 3, overlap: bool = True):
    """Split a dataframe with overlap with the provided window size.

    Args:
        df (DataFrame): The dataframe to split.
        window_size (int): The window size to split on.
        overlap (bool): Whether to overlap.

    References:
        - https://stackoverflow.com/questions/59737115/iterate-dataframe-with-window

    Yields:

    """
    for index in range(0, len(df) - overlap, window_size - (window_size - 1)):
        yield index, df.iloc[index : index + window_size]


def register_display_tables_pandas_option() -> None:
    """Register a new config option with pandas for whether to display tables."""
    try:
        with config.config_prefix("assignment2"):
            config.register_option(
                "display_tables",
                True,
                validator=is_bool,
            )
    except OptionError as error:
        # Already registered.
        print(error)


def get_utc(date: str, date_format: str = "%Y-%m-%d") -> float:
    """Get an utc timestamp of the provided date string.

    Args:
        date (str): The date string to convert to an utc timestamp.
        date_format (str): The format of the date string.

    Returns:
        float: The utc timestamp.
    """
    return datetime.strptime(date, date_format).replace(tzinfo=timezone.utc).timestamp()


def compute_accuracy(accurate_values: int, total_values: int) -> float:
    """Compute accuracy based upon accurate values and total.

    Args:
        accurate_values (int): The accurate values.
        total_values (int): The total values.

    Returns:
        float: The computed accuracy.
    """
    return accurate_values / total_values


def percent_change(current: int | float, previous: int | float) -> float:
    """Get the percentage change between two percent values.

    Args:
        current (int | float): The current percentage value.
        previous (int | float): The previous percentage value to compare to.

    Returns:
        float: The percentage change from the previous value to the current.

    References:
        - https://stackoverflow.com/a/30926930
    """
    if current == previous:
        return 0.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float("inf")


def float_to_currency(value: float) -> str:
    """Format a float value to a USD currency string.

    Args:
        value (float): The float value to format.

    Returns:
        str: The formatted USD currency string.
    """
    return "${:,.2f}".format(value)


def float_to_percentage(value: float) -> str:
    """Format a float value to a percentage string.

    Args:
        value (float): The float value to format.

    Returns:
        str: The formatted percentage string.
    """
    return f"{value:.2%}"


def mean(items: list[int | float]) -> float:
    """Compute the mean value of the provided list of items.

    Args:
        items (list[int | float]): The items to compute the mean value from.

    Returns:
        float: The mean value of the items list.
    """
    return sum(items) / len(items)


def square(num: int | float) -> int | float:
    """Compute the square value of the provided number.

    Args:
        num (int | float): The number to square.

    Returns:
        int | float: The square value of the provided number.
    """
    return num**2


def difference(num1: float, num2: float) -> float:
    """Get the difference between two numbers.

    Args:
        num1 (float): The first number to compare.
        num2 (float): The second number to compare.

    Returns:
        float: The difference between the two numbers.
    """
    return abs(num1 - num2)


def standard_deviation(items: list[int | float]) -> float:
    """Compute the standard deviation value of the provided list of items.

    Args:
        items (list[int | float]): The items to compute the standard deviation value
            from.

    Returns:
        float: The standard deviation value of the items list.
    """
    return math.sqrt(sum(list(map(square, items))) / len(items) - square(mean(items)))


if __name__ == "__main__":
    pass
