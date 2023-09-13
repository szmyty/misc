"""Assignment1 module.

Alan Szmyt
Class: CS 677
Date: March 18th, 2023
Assignment #1
Description:
This assignment uses Python to analyze the distribution of stock returns and a number of
trading strategies.

References:
    - https://google.github.io/styleguide/pyguide.html
    - https://www.markdownguide.org/cheat-sheet/
    - https://www.markdownguide.org/basic-syntax/
    - https://www.investopedia.com/ask/answers/021915/how-standard-deviation-used-determine-risk.asp # noqa
    - https://nbviewer.org/
    - https://realpython.com/python-interface/
    - https://ipython.readthedocs.io/en/stable/
    - https://jupyter-notebook.readthedocs.io/en/stable/config.html
    - https://refactoring.guru/design-patterns/strategy/python/example
"""
from __future__ import annotations

import math
import os
from abc import ABCMeta, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import DEBUG, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Any

# Configure a logger to log statements.
logger: Logger = getLogger(__file__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())

# Weekday order mapping to use with sort.
# https://docs.python.org/3/howto/sorting.html#key-functions
WEEKDAYS: dict[str, int] = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}


class Assignment1Error(Exception):
    """Custom Exception for Assignment 1."""

    def __init__(
        self: Assignment1Error,
        message: str = "Assignment 1 Error",
        error: Exception | None = None,
    ) -> None:
        """Instantiate custom exception for Assignment 1.

        Args:
            message (str): Description of the error that was raised.
            error (Exception | None): Optional Exception to associate with the instance.
        """
        self.error: Exception | None = error
        self.message: str = message
        super().__init__(self.message)


class Model(metaclass=ABCMeta):
    """Abstract base data model class."""

    @property
    def props(self: Model) -> dict[str, Any]:
        """Get a dictionary of the model's props.

        Returns:
            dict[str, Any]: The dictionary of props.
        """
        return dict(
            (key, value)
            for key, value in self.__dict__.items()
            if not callable(value) and not key.startswith("__")
        )

    @classmethod
    def __subclasshook__(cls: type[Model], subclass):
        return (
            hasattr(subclass, "__pretty__")
            and callable(subclass.__pretty__)
            or NotImplemented
        )

    def pretty_print(self: Model) -> None:
        """Pretty print the model representation."""
        print(self.__pretty__)

    @property
    @abstractmethod
    def __pretty__(self: Model) -> str:
        """Pretty string of model representation.

        Returns:
            str: A pretty formatted string representation of the model class instance.
        """
        raise NotImplementedError


def nums_as_range_str(nums: list[int]) -> str:
    """Get a list of numbers as a formatted range string.

    If the length of the list is 1. Get the value of the item as a string.

    Args:
        nums (list[int]): The list of numbers to get as a range string.

    Returns:
        list[int]: The list of numbers as a range string.
    """
    if len(nums) == 0:
        return "Undefined"
    elif len(nums) == 1:
        return str(nums[0])
    else:
        return f"{min(nums)} - {max(nums)}"


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


def write_to_file(file: Path, text: str) -> None:
    """Write the provided text to the file at the provided file path.

    Args:
        file (Path): The file path to write the text to.
        text (str): The text to write to the file.
    """
    try:
        file.write_text(text, encoding="utf-8")
    except OSError as error:
        raise Assignment1Error(
            message="Failed to save text as file!", error=error
        ) from error


def float_to_currency(value: float) -> str:
    """Format a float value to a USD currency string.

    Args:
        value (float): The float value to format.

    Returns:
        str: The formatted USD currency string.
    """
    return "${:,.2f}".format(value)


def read_ticker_file(path: str | Path, ticker: str) -> list[str]:
    """Read stock data ticker csv file to a list of the rows.

    Returns:
        list[str]: The stock data represented as a list of strings.
    """
    try:
        with open(path, encoding="utf-8") as file:
            print(f"Opening file {Path(path)} for ticker: {ticker}")
            return file.readlines()
    except OSError as error:
        logger.debug("Error")
        raise Assignment1Error(
            message=f"Error reading file from {path}",
            error=error,
        ) from error


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


class PredictionStrategy(metaclass=ABCMeta):
    """Strategy interface for Oracle predictions."""

    @classmethod
    def __subclasshook__(cls: type[PredictionStrategy], subclass):
        return (
            hasattr(subclass, "predict")
            and callable(subclass.predict)
            or NotImplemented
        )

    @abstractmethod
    def predict(self, initial_investment: float, stocks: StockData) -> float:
        """Predict the total returns based upon an initial investment.

        Uses the provided stock dataset to calculate the predictions.

        Args:
            initial_investment (float): The initial investment to start with.
            stocks (StockData): The stock dataset to calculate the prediction over.

        Returns:
            float: The predicted total return for the end of the stock dataset.
        """
        pass


class NormalPredictionStrategy(PredictionStrategy):
    """Concrete prediction strategy that is 'normal'.

    In other words, the Oracle will provide the investor with an accurate prediction.
    """

    def predict(self, initial_investment: float, stocks: StockData) -> float:
        """Predict the total returns by providing the investor with the best options.

        Args:
            initial_investment (float): The initial investment to start with.
            stocks (StockData): The stock dataset to calculate the prediction over.

        Returns:
            float: The predicted total return for the end of the stock dataset.
        """
        current_investment: float = initial_investment

        # Add to running total if the return for the day is positive.
        for e in stocks.oldest_to_newest:
            if e.is_non_negative_return:
                current_investment = current_investment + (
                    e.daily_return * current_investment
                )
        return current_investment


class RevengeScenarioA(PredictionStrategy):
    """Revenge scenario 'a' for the Oracle's revenge."""

    def predict(self, initial_investment: float, stocks: StockData) -> float:
        """Predict the total returns based upon an initial investment.

        The Oracle uses revenge strategy 'a', which has the investor miss out on the top
        10 best trading days.

        Args:
            initial_investment (float): The initial investment to start with.
            stocks (StockData): The stock dataset to calculate the prediction over.

        Returns:
            float: The predicted total return for the end of the stock dataset.
        """
        current_investment: float = initial_investment
        best_trading_days: list[StockDataEntry] = stocks.best_trading_days()

        for e in stocks.oldest_to_newest:
            if e.is_non_negative_return and e not in best_trading_days:
                current_investment = current_investment + (
                    e.daily_return * current_investment
                )
        return current_investment


class RevengeScenarioB(PredictionStrategy):
    """Revenge scenario 'b' for the Oracle's revenge."""

    def predict(self, initial_investment: float, stocks: StockData) -> float:
        """Predict the total returns based upon an initial investment.

        The Oracle uses revenge strategy 'b', which has the investor realize on the top
        10 worst trading days.

        Args:
            initial_investment (float): The initial investment to start with.
            stocks (StockData): The stock dataset to calculate the prediction over.

        Returns:
            float: The predicted total return for the end of the stock dataset.
        """
        current_investment: float = initial_investment
        worst_trading_days: list[StockDataEntry] = stocks.worst_trading_days()

        for e in stocks.oldest_to_newest:
            if e.is_non_negative_return or e in worst_trading_days:
                current_investment = current_investment + (
                    e.daily_return * current_investment
                )
        return current_investment


class RevengeScenarioC(PredictionStrategy):
    """Revenge scenario 'c' for the Oracle's revenge."""

    def predict(self, initial_investment: float, stocks: StockData) -> float:
        """Predict the total returns based upon an initial investment.

        The Oracle uses revenge strategy 'c', which has the investor realize on the top
        5 worst trading days and to miss out on the 5 best trading days.

        Args:
            initial_investment (float): The initial investment to start with.
            stocks (StockData): The stock dataset to calculate the prediction over.

        Returns:
            float: The predicted total return for the end of the stock dataset.
        """
        current_investment: float = initial_investment
        best_trading_days: list[StockDataEntry] = stocks.best_trading_days(top=5)
        worst_trading_days: list[StockDataEntry] = stocks.worst_trading_days(top=5)

        for e in stocks.oldest_to_newest:
            if (
                e.is_non_negative_return and e not in best_trading_days
            ) or e in worst_trading_days:
                current_investment = current_investment + (
                    e.daily_return * current_investment
                )
        return current_investment


class Oracle:
    """Oracle that makes predictions and provides an investor with insights."""

    def __init__(
        self: Oracle, strategy: PredictionStrategy = NormalPredictionStrategy()
    ) -> None:
        """Instantiates an Oracle with a prediction strategy.

        Args:
            strategy (PredictionStrategy): The prediction strategy that the oracle will
                use.
        """
        self._strategy: PredictionStrategy = strategy

    @property
    def strategy(self) -> PredictionStrategy:
        """

        Returns:
            PredictionStrategy: The current prediction strategy.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: PredictionStrategy) -> None:
        """Set the provided prediction strategy as the active strategy.

        Args:
            strategy (PredictionStrategy): The strategy to use.
        """
        self._strategy = strategy

    def make_predictions(
        self: Oracle, stocks: StockData, initial_investment: float = 100.0
    ) -> float:
        """Make predictions using the current prediction strategy.

        Args:
            stocks (StockData): The stock dataset to calculate the prediction over.
            initial_investment (float): The initial investment to start with.

        Returns:
            float: The total returns that the oracle predicts.
        """
        return self._strategy.predict(initial_investment, stocks)


class LossGain:
    """Data class for holding loss and gain results of a stock."""

    def __init__(self: LossGain, gain: int | float, loss: int | float) -> None:
        """Instantiate loss/gain information based upon the provided values.

        Args:
            gain (int | float): The gain value for the stock.
            loss (int | float): The loss value for the stock.
        """
        self.gain: float = float(gain)
        self.loss: float = float(loss)
        self.difference: float = difference(self.gain, self.loss)
        self.status: tuple = (
            ("gain", "up") if self.gain >= self.loss else ("loss", "down")
        )

    @property
    def comparison(self: LossGain) -> str:
        """Get a status message to inform the investor.

        Tell the investor if they have lost more or a 'down' day or gained more on an
        'up' day.

        Returns:
            str: The status message.
        """
        return f"You {self.status[0]} more on a '{self.status[1]}' day."


@dataclass(frozen=True)
class StockDataTableRow:
    """Data model for a table row in a stock data table."""

    day: str
    mean: float
    standard_deviation: float
    negative_returns: int
    negative_mean: float
    negative_standard_deviation: float
    non_negative_returns: int
    non_negative_mean: float
    non_negative_standard_deviation: float

    def as_table_row(self: StockDataTableRow) -> list[str]:
        """Get a table row represented as a list of strings.

        Returns:
            list[str]: The table row represented as a list of strings.
        """
        return [
            self.day,
            str(self.mean),
            str(self.standard_deviation),
            str(self.negative_returns),
            str(self.negative_mean),
            str(self.negative_standard_deviation),
            str(self.non_negative_returns),
            str(self.non_negative_mean),
            str(self.non_negative_standard_deviation),
        ]


@dataclass(frozen=True)
class StockDataTableData:
    """Container class for the stock data table's data rows."""

    rows: list[StockDataTableRow]


@dataclass(frozen=True)
class StockDataEntry(Model):
    """Data model of a stock data entry representation.

    Each entry is a row in the table and extracted from one line of the csv file.
    """

    date: str
    utc: float
    year: int
    month: int
    day: int
    weekday: str
    week_number: str
    year_week: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float
    daily_return: float
    short_ma: float
    long_ma: float

    @property
    def __pretty__(self: StockDataEntry) -> str:
        """Pretty string of model representation.

        Returns:
            str: A pretty formatted string representation of the model class instance.
        """
        entries: str = "\n\t\t\t".join(
            [f"'{key}': {value}" for (key, value) in self.props.items()]
        )
        return f"<{type(self).__name__}\n" f"\t\t\t{entries}\n" f"\t\t/>"

    @property
    def is_non_negative_return(self: StockDataEntry) -> bool:
        """Get whether the data entry is a non-negative return.

        Returns:
            bool: Whether the data entry is a non-negative return.
        """
        return self.daily_return >= 0

    @staticmethod
    def parse_stock_data_entry(entry: str) -> list[str]:
        """Parse stock data entry into a list.

        Args:
            entry (str): The stock data entry to parse.

        Returns:
            list[str]: The stock data entry represented as a list.
        """
        return entry.strip().split(",")

    @classmethod
    def from_raw_data_entry(
        cls: type[StockDataEntry], raw_entry: str
    ) -> StockDataEntry:
        """Instantiate an instance from a raw stock data entry comma-separated string.

        Args:
            raw_entry (str): The raw stock data entry comma-separated string.

        Returns:
            StockDataEntry: The data model class representation of the data entry.

        Raises:
            Assignment1Error: An error occurred creating a class instance from a raw
                stock data entry.
        """
        try:
            entry: list[str] = StockDataEntry.parse_stock_data_entry(raw_entry)
            date: str = entry[0]

            # Store the UTC timestamp of the date for easy sorting.
            utc: float = (
                datetime.strptime(date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
            return StockDataEntry(
                date=date,
                utc=utc,
                year=int(entry[1]),
                month=int(entry[2]),
                day=int(entry[3]),
                weekday=entry[4],
                week_number=entry[5],
                year_week=entry[6],
                open=float(entry[7]),
                high=float(entry[8]),
                low=float(entry[9]),
                close=float(entry[10]),
                volume=int(entry[11]),
                adj_close=float(entry[12]),
                daily_return=float(entry[13]),
                short_ma=float(entry[14]),
                long_ma=float(entry[15]),
            )
        except ValueError as error:
            raise Assignment1Error(
                message="Failed to create StockDataEntry from raw stock data entry.",
                error=error,
            ) from error


@dataclass(frozen=True)
class StockData(Model):
    """Data model for the entire stock data set."""

    ticker: str
    entries: list[StockDataEntry]

    @property
    def __pretty__(self: StockData) -> str:
        """Pretty string of model representation.

        Returns:
            str: A pretty formatted string representation of the model class instance.
        """
        entries: str = "\n\t\t".join([f"{entry.__pretty__}" for entry in self.entries])
        return (
            f"<{type(self).__name__}\n"
            f"\t'ticker': {self.ticker}\n"
            f"\t'entries': [\n"
            f"\t\t{entries}\n"
            f"\t]\n"
            f"/>"
        )

    @property
    def years(self: StockData) -> set[int]:
        """Get the set of years that are in the stock dataset.

        Returns:
            set[int]: The set of years that are in the stock dataset.
        """
        return {entry.year for entry in self.entries}

    @property
    def years_str(self: StockData) -> str:
        """Get the dataset's years as a range string.

        Returns:
            str: The dataset's years as a range string.
        """
        return nums_as_range_str(list(self.years))

    @property
    def weekdays(self: StockData) -> set[str]:
        """Get the set of weekdays that are in the stock dataset.

        Returns:
            set[str]: The set of years that are in the stock dataset.
        """
        return {entry.weekday for entry in self.entries}

    def stock_data_for_weekday(self: StockData, day: str) -> StockData:
        """Get a subset stock dataset for the provided weekday.

        Args:
            day (str): The day of the week to filter a subset on.

        Returns:
            StockData: A subset dataset for the weekday.
        """
        return StockData(
            self.ticker,
            entries=list(
                filter(
                    lambda entry: entry.weekday == day,
                    self.entries,
                )
            ),
        )

    def stock_data_for_year(self: StockData, year: int | str) -> StockData | None:
        """Get a subset stock dataset for the provided year.

        Args:
            year (str): The year to filter a subset on.

        Returns:
            StockData: A subset dataset for the year.
        """
        return StockData(
            self.ticker,
            entries=list(
                filter(
                    lambda entry: entry.year == int(year),
                    self.entries,
                )
            ),
        )

    def stock_data_for_years(self: StockData) -> list[StockData]:
        """Get a stock dataset for each year.

        Returns:
            list[StockData]: A stock dataset for each year as a list.
        """
        return [self.stock_data_for_year(year) for year in self.years]

    @property
    def daily_returns(self: StockData) -> list[float]:
        """Get all daily returns for the stock dataset.

        Returns:
            list[float]: The daily returns for the stock dataset.
        """
        return [entry.daily_return for entry in self.entries]

    @property
    def mean_of_daily_returns(self: StockData) -> float:
        """Get the mean of all the daily returns in the stock dataset.

        Returns:
            float: The mean of all the daily returns in the stock dataset.
        """
        return mean(self.daily_returns)

    @property
    def standard_deviation_of_daily_returns(self: StockData) -> float:
        """Get the standard deviation of all the daily returns in the stock dataset.

        Returns:
            float: The standard deviation of all the daily returns in the stock dataset.
        """
        return standard_deviation(self.daily_returns)

    @property
    def non_negative_daily_returns(self: StockData) -> list[StockDataEntry]:
        """Get all stock data entries that have non-negative returns.

        Returns:
            list[StockDataEntry]: The stock data entries that have non-negative returns.
        """
        return list(filter(lambda entry: entry.daily_return >= 0, self.entries))

    @property
    def non_negative_mean(self: StockData) -> float:
        """Get the mean of all data entries that have non-negative returns.

        Returns:
            float: The mean of all data entries that have non-negative returns.
        """
        return mean([entry.daily_return for entry in self.non_negative_daily_returns])

    @property
    def non_negative_standard_deviation(self: StockData) -> float:
        """Get the standard deviation of all data entries that have negative returns.

        Returns:
            float: The standard deviation of all data entries that have negative
                returns.
        """
        return standard_deviation(
            [entry.daily_return for entry in self.non_negative_daily_returns]
        )

    @property
    def negative_daily_returns(self: StockData) -> list[StockDataEntry]:
        """Get all stock data entries that have negative returns.

        Returns:
            list[StockDataEntry]: The stock data entries that have negative returns.
        """
        return list(filter(lambda entry: entry.daily_return < 0, self.entries))

    @property
    def negative_mean(self: StockData) -> float:
        """Get the mean of all data entries that have negative returns.

        Returns:
            float: The mean of all data entries that have negative returns.
        """
        return mean([entry.daily_return for entry in self.negative_daily_returns])

    @property
    def negative_standard_deviation(self: StockData) -> float:
        """Get the standard deviation of all data entries that have negative returns.

        Returns:
            float: The standard deviation of all data entries that have negative
                returns.
        """
        return standard_deviation(
            [entry.daily_return for entry in self.non_negative_daily_returns]
        )

    @property
    def overall_negative_days(self: StockData) -> int:
        """Get the overall count of days that had negative returns.

        Returns:
            int: The count of days that had negative returns.
        """
        return len(self.negative_daily_returns)

    @property
    def overall_non_negative_days(self: StockData) -> int:
        """Get the overall count of days that had non-negative returns.

        Returns:
            int: The count of days that had non-negative returns.
        """
        return len(self.non_negative_daily_returns)

    @property
    def is_non_negative_overall(self: StockData) -> bool:
        """Get whether the dataset has more non-negative days overall.

        Returns:
            bool: Whether the dataset has more non-negative days overall.
        """
        return self.overall_non_negative_days >= self.overall_negative_days

    @property
    def overall_return_status(self: StockData) -> str:
        """Get a status message for the overall daily returns.

        Returns:
            str: A status message for the overall daily returns.
        """
        status: str = (
            "non-negative"
            if self.overall_non_negative_days > self.overall_negative_days
            else "negative"
        )
        return (
            f"Total Negative Days: {self.overall_negative_days}\n"
            f"Total Non-Negative Days: {self.overall_non_negative_days}\n"
            f"There are more {status} days for this dataset!"
        )

    @property
    def oldest_to_newest(self: StockData) -> list[StockDataEntry]:
        """Get the stock data entries sorted from oldest to newest.

        Returns:
            list[StockDataEntry]: The stock data entries sorted from oldest to newest.
        """
        return sorted(self.entries, key=lambda entry: entry.utc)

    @property
    def entries_per_day(self: StockData) -> dict[str, list[StockDataEntry]]:
        """Get a map of stock data entries for each day.

        Returns:
            dict[str, list[StockDataEntry]: A map of stock data entries for each day.
        """
        d: dict[str, list[StockDataEntry]] = {}
        for entry in self.entries:
            d.setdefault(entry.weekday, []).append(entry)
        return d

    @property
    def average_return_per_day(self: StockData) -> dict[str, float]:
        """Get a map of the mean for daily returns for each day.

        Returns:
            dict[str, float]: A map of the mean for daily returns for each day.
        """
        return {
            day: mean(list(map(lambda e: e.daily_return, entries)))
            for (day, entries) in self.entries_per_day.items()
        }

    @property
    def best_return_day(self: StockData) -> str:
        """Get the day that has the best returns overall.

        Returns:
            str: The day that has the best returns.
        """
        return max(self.average_return_per_day, key=self.average_return_per_day.get)

    @property
    def worst_return_day(self: StockData) -> str:
        """Get the day that has the worst returns overall.

        Returns:
            str: The day that has the worst returns.
        """
        return min(self.average_return_per_day, key=self.average_return_per_day.get)

    def best_trading_days(self: StockData, top: int = 10) -> list[StockDataEntry]:
        """Get the best trading days in the dataset.

        Args:
            top (int): The 'top' best days. Defaults to the top 10 best days.

        Returns:
            list[StockDataEntry]: The best trading days.
        """
        return sorted(self.entries, key=lambda entry: entry.daily_return, reverse=True)[
            :top
        ]

    def worst_trading_days(self: StockData, top: int = 10) -> list[StockDataEntry]:
        """Get the worst trading days in the dataset.

        Args:
            top (int): The 'top' worst days. Defaults to the top 10 worst days.

        Returns:
            list[StockDataEntry]: The worst trading days.
        """
        return sorted(self.entries, key=lambda entry: entry.daily_return)[:top]

    def loss_gain_comparison(self: StockData) -> LossGain:
        """Get a loss/gain comparison of the overall negative and non-negative returns.

        Returns:
            LossGain: A loss/gain comparison of the overall negative and non-negative
                returns.
        """
        return LossGain(self.non_negative_mean, self.negative_mean)

    def loss_gain_per_day(self: StockData) -> dict[str, LossGain]:
        """Get a map of the loss/gain comparison for each day.

        Returns:
            dict[str, LossGain]: A map of the loss/gain comparison for each day.
        """
        d: dict[str, LossGain] = {}
        for weekday in self.weekdays:
            d.update(
                {weekday: self.stock_data_for_weekday(weekday).loss_gain_comparison()}
            )
        return d

    def buy_and_hold(self: StockData, initial_investment: float = 100.0) -> float:
        """Buy and hold the stock from start of dataset to finish.

        initial_investment (float): The initial investment to start with.

        Returns:
            float: The total returns that the result from holding.
        """
        current_investment: float = initial_investment
        for e in self.oldest_to_newest:
            current_investment = current_investment + (
                e.daily_return * current_investment
            )
        return current_investment

    @classmethod
    def from_ticker_file(
        cls: type[StockData], ticker_file: str | Path, ticker: str
    ) -> StockData:
        """Instantiate an instance from the provided stock ticker csv file.

        Args:
            ticker_file (str | Path): The file path to the stock ticker file.
            ticker (str): The stock's ticker symbol to use.

        Returns:
            StockData: A data model instance representation of the ticker file's data.
        """
        try:
            raw_stock_data: list[str] = read_ticker_file(ticker_file, ticker)
            entries: list[StockDataEntry] = list(
                map(StockDataEntry.from_raw_data_entry, raw_stock_data[1:])
            )
            return StockData(ticker=ticker, entries=entries)
        except Assignment1Error as error:
            raise Assignment1Error(
                message="Failed to create StockData instance from ticker file.",
                error=error,
            ) from error

    def as_table(self: StockData) -> StockDataTable:
        """Get the dataset as a table data model.

        Returns:
            StockDataTable: The dataset as a table data model.
        """
        try:
            rows: list[StockDataTableRow] = []
            for weekday in self.weekdays:
                _d: StockData = self.stock_data_for_weekday(weekday)
                nn_std_dev: float = _d.non_negative_standard_deviation
                rows.append(
                    StockDataTableRow(
                        day=weekday,
                        mean=_d.mean_of_daily_returns,
                        standard_deviation=_d.standard_deviation_of_daily_returns,
                        negative_returns=len(_d.negative_daily_returns),
                        negative_mean=_d.negative_mean,
                        negative_standard_deviation=_d.negative_standard_deviation,
                        non_negative_returns=len(_d.non_negative_daily_returns),
                        non_negative_mean=_d.non_negative_mean,
                        non_negative_standard_deviation=nn_std_dev,
                    )
                )
            return StockDataTable(
                title=f"Data Table for {self.ticker} in {self.years_str}",
                data=StockDataTableData(rows=rows),
            )
        except ValueError as error:
            raise Assignment1Error(
                message="Failed to create StockDataTable from stock data.",
                error=error,
            ) from error


@dataclass
class StockDataTable:
    """Data model container class for a stock data table."""

    title: str
    data: StockDataTableData

    @property
    def headers(self: StockDataTable) -> list[str]:
        """Table headers for the stock data table.

        Returns:
            list[str]: The table headers.
        """
        return [
            "Day",
            "µ(R)",
            "σ(R)",
            "|R− |",
            "µ(R− )",
            "σ(R− )",
            "|R+ |",
            "µ(R+ )",
            "σ(R+ )",
        ]

    @property
    def latex_headers(self: StockDataTable) -> list[str]:
        """Latex formatted table headers for the stock data table.

        Returns:
            list[str]: The latex formatted table headers.
        """
        return [
            "Day",
            "$\\mu(R)$",
            "$\\sigma(R)$",
            "|R− |",
            "$\\mu(R− )$",
            "$\\sigma(R− )$",
            "|R+ |",
            "$\\mu(R+ )$",
            "$\\sigma(R+ )$",
        ]

    @property
    def rows(self: StockDataTable) -> list[StockDataTableRow]:
        """Get table rows sorted by weekday.

        Returns:
            list[StockDataTableRow]: Table rows sorted by weekday.
        """
        return sorted(self.data.rows, key=lambda row: WEEKDAYS.__getitem__(row.day))

    @property
    def table_data(self: StockDataTable) -> list[list[str]]:
        """Get table data to be used to generate a table with tabulate.

        Returns:
            list[list[str]]: Raw table data.
        """
        return [row.as_table_row() for row in self.rows]

    def display(
        self: StockDataTable, simple: bool = True, tablefmt: str = "pretty"
    ) -> None:
        """Display the data table using tabulate or as a simple table.

        If tabulate is not installed, fallback to the simple table.

        Args:
            simple (bool): Whether to print a simple table.
            tablefmt (str): The table format to use with tabulate.
        """
        if simple:
            self._display_simple()
        else:
            try:
                from tabulate import tabulate

                print(tabulate(self.table_data, self.headers, tablefmt=tablefmt))
            except ImportError:
                # tabulate package is not installed.
                self._display_simple()

    def _display_simple(self: StockDataTable) -> None:
        """Helper method to display a simple table of the table data."""
        print(
            f"Stock Ticker: {self.title}",
        )
        print(
            "{:<12} {:<25} {:<25} {:<10} {:<25} {:<25} {:<10} {:<25} {:<25}".format(
                *self.headers
            )
        )
        for row in self.table_data:
            print(
                "{:<12} {:<25} {:<25} {:<10} {:<25} {:<25} {:<10} {:<25} {:<25}".format(
                    *row
                )
            )


if __name__ == "__main__":
    # Mirror of Jupyter Notebook to run directly and debug.

    # Directory path to the 'resources' folder.
    resources: Path = Path(__file__).parent.joinpath("resources").resolve()

    # Sony stock ticker file path and ticker abbreviation.
    sony_ticker: str = "SONY"
    sony_ticker_file: Path = resources.joinpath("SONY.csv")

    # Parse Sony stock data from csv file to a StockData instance.
    sony_stock_data: StockData = StockData.from_ticker_file(
        sony_ticker_file, sony_ticker
    )

    # S&P-500 stock ticker file path and ticker abbreviation.
    spy_ticker: str = "SPY"
    spy_ticker_file: Path = resources.joinpath("SPY.csv")

    # Parse S&P-500 stock data from csv file to a StockData instance.
    spy_stock_data: StockData = StockData.from_ticker_file(spy_ticker_file, spy_ticker)

    # region QUESTION 1
    # Print the parsed Sony stock data with pretty format.
    sony_stock_data.pretty_print()

    # Save to file.
    write_to_file(
        file=Path(os.path.abspath("")).joinpath("sony_stock_data.pretty").resolve(),
        text=sony_stock_data.__pretty__,
    )

    # Get stock data for each year in the main dataset.
    sony_stock_data_per_year: list[StockData] = sony_stock_data.stock_data_for_years()

    # Display each year as a table.
    for stock_data in sony_stock_data_per_year:
        table: StockDataTable = stock_data.as_table()
        table.display()

    # Separate lists of all negative and non-negative daily returns entries.
    negative_days: list[StockDataEntry] = sony_stock_data.negative_daily_returns
    non_negative_days: list[StockDataEntry] = sony_stock_data.non_negative_daily_returns

    overall_return_status: str = sony_stock_data.overall_return_status
    print(overall_return_status)

    # Compare the overall loss/gain means for the entire dataset.
    loss_gain_comparison: LossGain = sony_stock_data.loss_gain_comparison()
    print(loss_gain_comparison.comparison)

    # Get the loss/gain mean for each day of the week in the dataset.
    loss_gain_per_day: dict[str, LossGain] = sony_stock_data.loss_gain_per_day()
    for _day, loss_gain in loss_gain_per_day.items():
        print(f"Loss/gain for {_day}: {loss_gain.comparison}")
    # endregion

    # region QUESTION 2
    # Get a count of how many times each day was the best over the years for SONY
    # dataset.
    best_sony_days_count: Counter = Counter(
        [entry.best_return_day for entry in sony_stock_data_per_year]
    )
    print(best_sony_days_count)

    # Get a count of how many times each day was the best over the years for S&P-500
    # dataset.
    best_spy_days_count: Counter = Counter(
        [entry.best_return_day for entry in spy_stock_data.stock_data_for_years()]
    )
    print(best_spy_days_count)

    # For each year, decide the best day and worst day based upon on average daily
    # return for each day.
    for data in sony_stock_data_per_year:
        print(f"Best day for {data.years_str}: {data.best_return_day}")
        print(f"Worst day for {data.years_str}: {data.worst_return_day}")

    # Best days across all years.
    best_days_across_years: set[str] = {
        entry.best_return_day for entry in sony_stock_data_per_year
    }
    print(best_days_across_years)

    # Worst days across all years.
    worst_days_across_years: set[str] = {
        entry.worst_return_day for entry in sony_stock_data_per_year
    }
    print(worst_days_across_years)
    # endregion

    # region QUESTION 3
    # Display the SONY stock data in a table.
    sony_stock_table: StockDataTable = sony_stock_data.as_table()
    sony_stock_table.display()

    # Display the SPY stock data in a table.
    spy_stock_table: StockDataTable = spy_stock_data.as_table()
    spy_stock_table.display()

    # Calculate the best day of week for the SONY dataset by taking the mean for all
    # the entries for each day.
    best_sony_return_day: str = sony_stock_data.best_return_day
    print(f"The best day of the week for the SONY dataset is {best_sony_return_day}.")

    # Calculate the worst day of week for the SONY dataset by taking the mean for all
    # the entries for each day.
    worst_sony_return_day: str = sony_stock_data.worst_return_day
    print(f"The worst day of the week for the SONY dataset is {worst_sony_return_day}.")

    # Calculate the best day of week for the S&P-500 dataset by taking the mean for all
    # the entries for each day.
    best_spy_return_day: str = spy_stock_data.best_return_day
    print(f"The best day of the week for the S&P-500 dataset is {best_spy_return_day}.")

    # Calculate the worst day of week for the S&P-500 dataset by taking the mean for all
    # the entries for each day.
    worst_spy_return_day: str = spy_stock_data.worst_return_day
    print(
        f"The worst day of the week for the S&P-500 dataset is {worst_spy_return_day}."
    )
    # endregion

    # region QUESTION 4
    oracle: Oracle = Oracle()

    # Listen to the oracle for the SONY stock.
    sony_oracle_return: float = oracle.make_predictions(
        stocks=sony_stock_data, initial_investment=100.0
    )
    print(
        f"Current investment of SONY after listening to oracle: "
        f"{float_to_currency(sony_oracle_return)}"
    )

    # Listen to the oracle for the SPY stock.
    spy_oracle_return: float = oracle.make_predictions(
        stocks=spy_stock_data, initial_investment=100.0
    )
    print(
        f"Current investment after of S&P-500 listening to oracle: "
        f"{float_to_currency(spy_oracle_return)}"
    )
    # endregion

    # region QUESTION 5
    # Ignore the oracle and hold the stock throughout the duration of the dataset for
    # SONY.
    sony_buy_and_hold_return: float = sony_stock_data.buy_and_hold(
        initial_investment=100.0
    )
    print(
        f"Current investment of SONY after ignoring the oracle: "
        f"{float_to_currency(sony_buy_and_hold_return)}"
    )

    # Ignore the oracle and hold the stock throughout the duration of the dataset for
    # S&P-500.
    spy_buy_and_hold_return: float = spy_stock_data.buy_and_hold(
        initial_investment=100.0
    )
    print(
        f"Current investment of SPY after ignoring the oracle: "
        f"{float_to_currency(spy_buy_and_hold_return)}"
    )

    # Difference between listening to the oracle and ignoring the oracle for SONY.
    sony_return_difference: float = percent_change(
        sony_buy_and_hold_return, sony_oracle_return
    )

    if sony_oracle_return > sony_buy_and_hold_return:
        print(
            f"By not listening to the oracle, you have lost "
            f"{sony_return_difference:.2f}% of your potential gains for the SONY stock!"
        )
    else:
        print(
            f"You have overcome all odds and beat the oracle by "
            f"{(100.0 - sony_return_difference):.3f}% for the SONY stock!"
        )

    # Difference between listening to the oracle and ignoring the oracle for S&P-500.
    spy_return_difference: float = percent_change(
        spy_buy_and_hold_return, spy_oracle_return
    )

    if spy_oracle_return > spy_buy_and_hold_return:
        print(
            f"By not listening to the oracle, you have lost "
            f"{spy_return_difference:.2f}% of your potential gains for the S&P-500 "
            f"stock!"
        )
    else:
        print(
            f"You have somehow overcome all odds and beat the oracle by "
            f"{(100.0 - spy_return_difference):.3f}% for the S&P-500 stock!"
        )
    # endregion

    # region QUESTION 6
    # An angry oracle instance.
    angry_oracle: Oracle = Oracle(RevengeScenarioA())

    # Scenario A for SONY: Missing the top 10 best trading days.
    revenge_sony_oracle_return_a: str = float_to_currency(
        angry_oracle.make_predictions(stocks=sony_stock_data)
    )
    print(revenge_sony_oracle_return_a)

    # Scenario A for S&P-500: Missing the top 10 best trading days.
    revenge_spy_oracle_return_a: str = float_to_currency(
        angry_oracle.make_predictions(stocks=spy_stock_data)
    )
    print(revenge_spy_oracle_return_a)

    # Switch oracle's strategy to scenario B.
    angry_oracle.strategy = RevengeScenarioB()

    # Scenario B for SONY: Realizing the top 10 worst trading days.
    revenge_sony_oracle_return_b: str = float_to_currency(
        angry_oracle.make_predictions(stocks=sony_stock_data)
    )
    print(revenge_sony_oracle_return_b)

    # Scenario B for S&P-500: Realizing the top 10 worst trading days.
    revenge_spy_oracle_return_b: str = float_to_currency(
        angry_oracle.make_predictions(stocks=spy_stock_data)
    )
    print(revenge_spy_oracle_return_b)

    # Switch oracle's strategy to scenario C.
    angry_oracle.strategy = RevengeScenarioC()

    # Scenario C for SONY: Realizing the top 5 worst trading days and missing the top 5
    # best days.
    revenge_sony_oracle_return_c: str = float_to_currency(
        angry_oracle.make_predictions(stocks=sony_stock_data)
    )
    print(revenge_sony_oracle_return_c)

    # Scenario C for S&P-500: Realizing the top 5 worst trading days and missing the
    # top 5 best days.
    revenge_spy_oracle_return_c: str = float_to_currency(
        angry_oracle.make_predictions(stocks=spy_stock_data)
    )
    print(revenge_spy_oracle_return_c)

    # Compare missing the best days versus missing the worst days for SONY.
    if revenge_sony_oracle_return_a >= revenge_sony_oracle_return_b:
        print("You gained more by missing the best days for SONY.")
    else:
        print("You gain more by missing the worst days for SONY.")

    # Compare missing the best days versus missing the worst days for S&P-500.
    if revenge_spy_oracle_return_a >= revenge_spy_oracle_return_b:
        print("You gained more by missing the best days for S&P-500.")
    else:
        print("You gain more by missing the worst days for S&P-500.")

    # Compare question 4 to scenario c.
    revenge_versus_normal: str = float_to_currency(
        difference(
            angry_oracle.make_predictions(stocks=sony_stock_data), sony_oracle_return
        )
    )
    print(
        f"The oracle's revenge in scenario c cost you "
        f"{revenge_versus_normal} in losses."
    )
    # endregion

    print("Finished!")
