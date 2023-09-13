"""Utils module.

Alan Szmyt
Class: CS 677
Date: April 4th, 2023
Description:
This module contains a variety of utility methods used for any assignment. Separating
the methods out into this module improves readability.
"""
from __future__ import annotations

import inspect
import math
import operator
import pickle
import re
import textwrap
from abc import ABCMeta, abstractmethod
from collections import Counter, deque
from datetime import datetime, timezone
from enum import Enum, unique
from itertools import islice
from pathlib import Path
from random import Random
from string import punctuation
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import seaborn as sns
from loguru import logger
from marshmallow import Schema
from pandas import DataFrame, Series
from pandas._config import config  # noqa
from pandas._config.config import OptionError, is_bool  # noqa
from sklearn.tree import DecisionTreeClassifier, export_graphviz

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from matplotlib.axes import Axes


UTF8: str = "utf-8"

# Directory path to the 'resources' folder.
here: Path = Path(__file__).parent.resolve()
resources: Path = here.joinpath("resources")
artifacts_directory: Path = here.joinpath("artifacts")
checkpoints: Path = here.joinpath("checkpoints")
data_directory: Path = resources.joinpath("data")

DATAFRAME_NAME: str = "pandas_dataframe_name"


def log(message: str) -> None:
    """Log a message safely by proxying f-strings.

    Args:
        message (str): The message to log.
    """
    logger.info(message)


class ReprMixin:
    """Mixin class for setting the repr for the class with all attributes.

    References:
        - https://github.com/alexprengere/reprmixin
    """

    __slots__ = []

    def _find_attrs(self: ReprMixin) -> Any:
        """Iterate over all attributes of objects."""
        visited = set()

        if hasattr(self, "__dict__"):
            for attr in sorted(self.__dict__):
                if attr not in visited:
                    yield attr
                    visited.add(attr)

        for cls in reversed(inspect.getmro(self.__class__)):
            if hasattr(cls, "__slots__"):
                for attr in cls.__slots__:
                    if hasattr(self, attr) and attr not in visited:
                        yield attr
                        visited.add(attr)

    def __repr__(self: ReprMixin) -> str:
        """Representation of the class using the attributes.

        Returns:
            str: String representation of the class.
        """
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                f"\n{attr}={repr(getattr(self, attr))}"
                for attr in self._find_attrs()
                if not attr.startswith("_")
            ),
        )


class DataModel(ReprMixin, metaclass=ABCMeta):
    """Abstract base data model class."""

    @property
    def props(self: DataModel) -> dict[str, Any]:
        """Get a dictionary of the model's props.

        Returns:
            dict[str, Any]: The dictionary of props.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not callable(value) and not key.startswith("__")
        }

    @property
    @abstractmethod
    def schema_fields(self: DataModel) -> dict:
        """Get the schema fields for JSON serialization.

        Returns:
            dict: The schema fields for serialization.
        """
        raise NotImplementedError

    @property
    def schema(self: DataModel) -> Schema:
        """Get the marshmallow schema that is used for serialization.

        Returns:
            Schema: The marshmallow schema for this class.
        """
        return Schema.from_dict(self.schema_fields)()

    @property
    def as_json(self: DataModel) -> str:
        """Serialize the class instance as a JSON string.

        Returns:
            str: The class serialized as a JSON string.
        """
        return self.schema.dumps(self, indent=4, sort_keys=True)

    @classmethod
    def __subclasshook__(cls: type[DataModel], subclass):
        return (
            hasattr(subclass, "__pretty__")
            and callable(subclass.__pretty__)
            or hasattr(subclass, "schema_fields")
            and callable(subclass.schema_fields)
            or NotImplemented
        )

    def pretty_print(self: DataModel) -> None:
        """Pretty print the model representation."""
        log(self.__pretty__)

    @property
    @abstractmethod
    def __pretty__(self: DataModel) -> str:
        """Pretty string of model representation.

        Returns:
            str: A pretty formatted string representation of the model class instance.
        """
        raise NotImplementedError


@unique
class Color(Enum):
    """Enum class of colors."""

    GREEN: str = "Green"
    RED: str = "Red"

    @property
    def lower(self: Color) -> str:
        """Get the lower case of the color.

        Returns:
            Color: The lower case of the color.
        """
        return self.value.lower()


def create_latex_table(
    df: DataFrame, label: str, caption: str, position: str = "h!"
) -> str:
    """Create a formatted latex table string to be used in a latex document.

    Args:
        df (DataFrame): The dataframe to represent as a latex table.
        label (str): The label to associate with the table.
        caption (str): The caption to add to the table.
        position (str): The position for the table. (Defaults to 'h!').

    Returns:
        str: The dataframe represented as a latex table string.
    """
    with pd.option_context("max_colwidth", None, "styler.format.escape", "latex"):
        latex_table_str: str = df.style.hide(axis="index").to_latex(
            column_format=("c|" * (len(df.columns) + 1)).strip("|"),
            position=position,
            hrules=True,
            label=label,
            caption=caption,
            environment="longtable",
        )
        artifacts_directory.joinpath(f"{label}.tex").write_text(
            latex_table_str, encoding=UTF8
        )
        return latex_table_str


def relative_path_to(parent: Path, child: Path) -> Path | None:
    """Get the relative path that is shared between the child and parent file path.

    Args:
        parent (Path): The parent path.
        child (Path): The child path.

    References:
        - https://stackoverflow.com/a/57153766

    Returns:
        Path | None: The relative path to the child from the parent if exists.
    """
    return (
        child.relative_to(parent)
        if parent in child.parents or parent == child
        else None
    )


def df_unique_list(df: DataFrame, column: str) -> list[str]:
    """Get a list of unique values in the specified column.

    Args:
        df (DataFrame): The dataframe to get the unique values from.
        column (str): The column to get the values from.

    Returns:
        list[str]: The list of unique values that were in the column.
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


def split_array(arr: np.array, cond: np.array) -> tuple[np.array, np.array]:
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


def load_df_from_resources(filename: str) -> DataFrame:
    """Load the data frame from the file in the 'resources' directory.

    Args:
        filename (str): The name of the file.

    Returns:
        DataFrame: The deserialized dataframe from the file.
    """
    return pd.read_pickle(resources.joinpath(f"{filename}.pkl"))


def save_df_to_checkpoints(df: DataFrame, filename: str | None = None) -> None:
    """Save the provided dataframe to the 'checkpoints' directory.

    Args:
        df (DataFrame): The dataframe to save.
        filename (str | None): The name of the file.
    """
    if filename is None:
        filename: str = df.attrs.get(DATAFRAME_NAME, str(uuid4()))

    df.to_pickle(checkpoints.joinpath(f"{filename}.pkl"))


def save_dfs_to_checkpoints(dfs: list[DataFrame]) -> None:
    """Save the provided dataframes to the 'checkpoints' directory.

    Args:
        dfs (list[DataFrame]): The dataframes to save.
    """
    for df in dfs:
        save_df_to_checkpoints(df)


def load_df_from_checkpoints(filename: str) -> DataFrame:
    """Load the data frame from the file in the 'checkpoints' directory.

    Args:
        filename (str): The name of the file.

    Returns:
        DataFrame: The deserialized dataframe from the file.
    """
    return pd.read_pickle(checkpoints.joinpath(f"{filename}.pkl"))


def remove_punctuation(text: str) -> str:
    """Removes punctuation characters and replaces them with a space.

    Args:
        text (str): The text to remove punctuation from.

    Returns:
        str: The cleaned string.
    """
    return " ".join(
        re.sub(rf"""[{punctuation}]+\ *""", " ", text, flags=re.VERBOSE).strip().split()
    )


def to_title(text: str) -> str:
    """Convert text to a title.

    Args:
        text (str): The text to convert.

    Returns:
        str: The text as a title.
    """
    return remove_punctuation(text).title()


def wrap_labels(
    axes: Axes,
    width: int,
    break_long_words: bool = False,
    x_labels: bool = True,
    y_labels: bool = False,
) -> None:
    """Wrap labels when displaying them for plots.

    Args:
        axes (Axes): The axes to apply the word wrapping to.
        width (int): The index of the character to break on.
        break_long_words (bool): Whether to break words up, or just on spaces.
        x_labels (bool): Apply the wrap text to labels on the x-axis.
        y_labels (bool): Apply the wrap text to labels on the y-axis.

    References:
        - https://medium.com/dunder-data/automatically-wrap-graph-labels-in-matplotlib-and-seaborn-a48740bc9ce
    """
    if x_labels:
        labels = [
            textwrap.fill(
                remove_punctuation(label.get_text()),
                width=width,
                break_long_words=break_long_words,
            )
            for label in axes.get_xticklabels()
        ]
        axes.set_xticklabels(labels, rotation=0)

    if y_labels:
        labels = [
            textwrap.fill(
                remove_punctuation(label.get_text()),
                width=width,
                break_long_words=break_long_words,
            )
            for label in axes.get_yticklabels()
        ]
        axes.set_yticklabels(labels, rotation=0)


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
    return arr[0:0] if num == 0 else arr[-num:]


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


def find_occurrences(dataset: np.array[str], pattern: np.array[str]) -> int:
    """Find the count of occurrences of the subarray in the provided array.

    Uses a sliding window to search for the subarray occurrences.
    For now, the data type needs to be a string (dtype="U1").

    Args:
        dataset (np.array[str]): The array to search for the subarray over.
        pattern (np.array[str]): The subarray to search for.

    Returns:
        int: The count of occurrences of the subarray in the array.
    """
    occurrences: int = 0
    for w in sliding_window(dataset, size=len(pattern)):
        w_pattern: np.ndarray = np.fromiter(w, dtype="U1")
        if np.array_equal(pattern, w_pattern):
            occurrences = occurrences + 1
    return occurrences


def split_df_with_overlap(
        df: DataFrame,
        window_size: int = 3,
        overlap: bool = True
) -> tuple[int, DataFrame]:
    """Split a dataframe with overlap with the provided window size.

    Args:
        df (DataFrame): The dataframe to split.
        window_size (int): The window size to split on.
        overlap (bool): Whether to overlap.

    References:
        - https://stackoverflow.com/questions/59737115/iterate-dataframe-with-window

    Yields:
        (tuple[int, DataFrame]): The dataframe in the current window.
    """
    for index in range(0, len(df) - overlap, window_size - (window_size - 1)):
        yield index, df.iloc[index: index + window_size]


def register_display_tables_pandas_option() -> None:
    """Register a new config option with pandas for whether to display tables."""
    try:
        with config.config_prefix("assignment3"):
            config.register_option(
                "display_tables",
                True,
                validator=is_bool,
            )
    except OptionError as error:
        # Already registered.
        log(str(error))


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
    return f"${value:,.2f}"


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


def replace_path_text(path: Path, text: str, replacement: str) -> Path:
    """Replace each part in a file path that match the provided text.

    Args:
        path (Path): The file path to perform the replacement on.
        text (str): The text to match for in each path part.
        replacement (str): The string to replace the matched part with.

    References:
        - https://stackoverflow.com/a/75314241

    Returns:
        Path: The file path with the replaced parts.
    """
    return Path(*[part.replace(text, replacement) for part in list(path.parts)])


def latexify_math_var(var: str, texorpdf: bool = True) -> str:
    """Convert the provided math variable to a formatted latex string.

    The 'texorpdfstring' command allows for a PDF compatible string.
    The 'xspace' command prevents the string from overlapping.

    Args:
        var (str): The string to format.
        texorpdf (bool): Whether to add the texorpdfstring command.

    Returns:
        str: The latex formatted string.
    """
    return rf"\texorpdfstring{{${var}$}}\xspace" if texorpdf else rf"${var}$"


def dataframe_memory_usage(df: DataFrame) -> str:
    """Get the dataframe's current memory usage in MB.

    Args:
        df (DataFrame): The dataframe to get memory usage from.

    Returns:
        str: The memory usage of the dataframe in MB.
    """
    memory_usage: float = df.memory_usage().sum() / 1024**2
    return f"Memory usage of dataframe is {memory_usage:.2f} MB"


def rename_columns(df: DataFrame, col_map: dict[str, str]) -> DataFrame:
    """Rename multiple columns at once for a dataframe from the provided mapping.

    Args:
        df (DataFrame): The dataframe to rename the columns on.
        col_map (dict): The map of the old to new namings of the columns.

    Returns:
        DataFrame: A dataframe with the renamed columns.
    """
    return df.rename(col_map, axis=1)


def rename_rows(df: DataFrame, row_map: dict[str, str]) -> DataFrame:
    """Rename multiple rows at once for a dataframe from the provided mapping.

    Args:
        df (DataFrame): The dataframe to rename the rows on.
        row_map (dict): The map of the old to new namings of the rows.

    Returns:
        DataFrame: A dataframe with the renamed rows.
    """
    return df.rename(row_map, axis=0)


def save_array(arr: np.array, filename: str, output_dir: Path = checkpoints) -> Path:
    """Save a numpy array to a file.

    Args:
        arr (np.array): The array to save.
        filename (str): The filename to use for the numpy array.
        output_dir (Path): The directory to save the array to.

    Returns:
        Path: The file path where the array file was saved.
    """
    arr_file: Path = output_dir.joinpath(f"{filename}.npy")
    with open(arr_file, "wb") as f:
        np.save(f, arr)
    return arr_file


def shuffle(items: list, seed: int | None = None) -> list:
    """Shuffle a list of items.

    Args:
        items (list): The list of items to shuffle.
        seed (int | None): The random seed to use to shuffle.

    Returns:
        list: The shuffled list.
    """
    items = items.copy()
    Random(seed).shuffle(items)
    return items


def load_array(filename: str) -> np.array:
    """Load a numpy array from a file.

    Args:
        filename (str): The file's name.

    Returns:
        np.array: The loaded numpy array.
    """
    return np.load(str(checkpoints.joinpath(f"{filename}.npy")), allow_pickle=True)


def pickle_obj(
    obj: object, filename: str, output_dir: Path = artifacts_directory
) -> Path:
    """Pickle (serialize) a python object to a file.

    Args:
        obj (object): The python object to serialize.
        filename (str): The name of the file to save to.
        output_dir (Path): The directory to output the file to.

    Returns:
        Path: The file path of the pickled file.
    """
    pickled: Path = output_dir.joinpath(f"{filename}.pkl")

    with pickled.open(mode="wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pickled


def depickle_obj(filename: str, output_dir: Path = artifacts_directory) -> object:
    """Depickle (deserialize) a python object from a file.

    Args:
        filename (str): The name of the file to deserialize from.
        output_dir (Path): The output directory of where the file is located.

    Returns:
        object: The deserialized python object.
    """
    with open(output_dir.joinpath(f"{filename}.pkl"), "rb") as handle:
        return pickle.load(handle)


def save_figures(
    filename: str,
    output_dir: Path = artifacts_directory,
    ext: tuple = (".png", ".pdf", ".svg"),
) -> list[Path]:
    """Save multiple files from one plot.

    Args:
        filename (str): The filename to use for each figure file.
        output_dir (Path): The directory to save all figures to.
        ext (tuple): The file extensions to save the figure as.

    Returns:
        list[Path]: A list of file paths for the files that were saved.
    """
    files: list[Path] = [output_dir.joinpath(f"{filename}{e}") for e in ext]
    for file in files:
        plt.savefig(file, dpi=300)
    return files


def save_tree(
    classifier: DecisionTreeClassifier,
    filename: str,
    output_directory: Path = artifacts_directory,
) -> tuple[Path, Path]:
    """Save a decision tree diagram to a file.

    Args:
        classifier (DecisionTreeClassifier): The decision tree classifier to diagram.
        filename (str): The filename to use for the output file.
        output_directory (Path): The directory to file to.

    Returns:
        Path: The file path to where the diagram file was saved.
    """
    output_dot_file: Path = output_directory.joinpath(f"{filename}.dot")
    export_graphviz(
        decision_tree=classifier,
        out_file=str(output_dot_file),
        rounded=True,
        filled=True,
        special_characters=True,
    )
    output_png_file: Path = convert_dot_to_png(output_dot_file)
    return output_dot_file, output_png_file


def convert_dot_to_png(dot_file: Path) -> Path:
    """Convert a dot diagram file to a png.

    Args:
        dot_file (Path): The file path to the dot file.

    Returns:
        Path: The file path to the converted png file.
    """
    (graph,) = pydot.graph_from_dot_file(dot_file)
    png_file: Path = Path(f"{dot_file.with_suffix('')}.png").resolve()
    graph.write_png(png_file)
    return png_file


def compare_truth_to_column(
    df: DataFrame,
    truth_column: str,
    prediction_column: str,
    truth_symbol: int,
    prediction_symbol: int,
) -> Series:
    """Compare a column's rows to the truth column.

    Args:
        df (DataFrame): The dataframe to compare from.
        truth_column (str): The truth column to compare to.
        prediction_column (str): The column to compare with.
        truth_symbol (int): The symbol of the truth label.
        prediction_symbol (int): The symbol of the prediction.

    Returns:
        Series: The row where the truth column matches the prediction symbol.
    """
    return df.loc[
        operator.and_(
            df[truth_column].astype(int) == truth_symbol,
            df[prediction_column].astype(int) == prediction_symbol,
        )
    ]


def get_true_positives(
    df: DataFrame, truth_column: str, prediction_column: str
) -> Series:
    """Get the TP (true positives) value of the column.

    Args:
        df (DataFrame): The data frame to get the true positives from.
        truth_column (str): The truth column to compare to.
        prediction_column (str): The column to compare with.

    Returns:
        Series: A row of the true positives.
    """
    return compare_truth_to_column(df, truth_column, prediction_column, 0, 0)


def get_false_positives(
    df: DataFrame, truth_column: str, prediction_column: str
) -> Series:
    """Get the FP (false positives) value of the column.

    Args:
        df (DataFrame): The data frame to get the false positives from.
        truth_column (str): The truth column to compare to.
        prediction_column (str): The column to compare with.

    Returns:
        Series: A row of the false positives.
    """
    return compare_truth_to_column(df, truth_column, prediction_column, 1, 0)


def get_true_negatives(
    df: DataFrame, truth_column: str, prediction_column: str
) -> Series:
    """Get the TN (true negatives) value of the column.

    Args:
        df (DataFrame): The data frame to get the true negatives from.
        truth_column (str): The truth column to compare to.
        prediction_column (str): The column to compare with.

    Returns:
        Series: A row of the true negatives.
    """
    return compare_truth_to_column(df, truth_column, prediction_column, 1, 1)


def get_false_negatives(
    df: DataFrame, truth_column: str, prediction_column: str
) -> Series:
    """Get the FN (false negatives) value of the column.

    Args:
        df (DataFrame): The data frame to get the false negatives from.
        truth_column (str): The truth column to compare to.
        prediction_column (str): The column to compare with.

    Returns:
        Series: A row of the false negatives.
    """
    return compare_truth_to_column(df, truth_column, prediction_column, 0, 1)


def create_correlation_matrix(
    df: DataFrame,
) -> DataFrame:
    """Get a dataframe's correlation matrix.

    Args:
        df (DataFrame): The dataframe to compute the correlation matrix for.

    Returns:
        DataFrame: The correlation dataframe.
    """
    return df.corr(numeric_only=True)


def plot_correlation_matrix(
    df: DataFrame, title: str, output_dir: Path = artifacts_directory
) -> DataFrame:
    """Plot the provided dataframe's correlation matrix.

    Args:
        df (DataFrame): The dataframe to compute the correlation matrix for.
        title (str): The title of the plot to use.
        output_dir (Path): The directory to save the plots to.

    Returns:
        DataFrame: The correlation dataframe.
    """
    correlation_df: DataFrame = create_correlation_matrix(df)
    plt.figure(figsize=(20, 20))

    # Create a correlation matrix heatmap with seaborn.
    correlation_matrix = sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".1f",
        linewidths=1,
        annot_kws={"size": 80 / np.sqrt(len(correlation_df))},
        cmap=sns.cubehelix_palette(as_cmap=True),
        square=True,
    )
    correlation_matrix.set_xticklabels(
        correlation_matrix.get_xticklabels(), rotation=30
    )
    correlation_matrix.figure.tight_layout()
    correlation_matrix.figure.suptitle(title)

    # Save the plot to a file.
    save_figures("_".join(title.lower().split()), output_dir)

    # Wrap the labels for the plot and display.
    wrap_labels(correlation_matrix, 6, x_labels=False, y_labels=True)
    plt.show()

    return correlation_df


def examine_correlation_matrix(
    df: DataFrame,
) -> Series:
    """Examine a dataframe's correlation matrix.

    Args:
        df (DataFrame): The correlation matrix dataframe to examine.

    Returns:
        Series: The sorted correlation series.
    """
    correlations: Series = df.abs().unstack().sort_values(ascending=False)

    seen = []
    for features, correlation in correlations.items():
        # Remove correlations with the same name, aka a feature correlated to itself.
        if features[0] == features[1]:
            correlations = correlations.drop(features)
        else:
            # Drop duplicate correlation pairs.
            sorted_features = sorted(features)
            if sorted_features not in seen:
                seen.append(sorted_features)
            else:
                correlations = correlations.drop(features)

    return correlations


class Weekday(Enum):
    """Enum representation of a weekday.

    References:
        - https://docs.python.org/3/howto/enum.html
    """

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7

    @classmethod
    def from_date(cls: type[Weekday], date: datetime) -> Weekday:
        """Get weekday representation from a datetime instance.

        Args:
            date (datetime): The datetime to get the weekday from.

        Returns:
            Weekday: The weekday representation of the datetime.
        """
        return cls(date.isoweekday())
