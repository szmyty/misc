from __future__ import annotations

from pathlib import Path

import pytest as pytest

from assignment1.assignment1 import (
    Assignment1Error,
    StockData,
    StockDataEntry,
    mean,
    read_ticker_file,
    standard_deviation,
)


@pytest.mark.usefixtures("resources")
class TestAssignment1:
    """Perform testing of assignment1 functionality."""

    @pytest.mark.usefixtures("resources")
    def test_read_ticker_file(self: TestAssignment1, resources: Path) -> None:
        ticker: str = "SONY"
        sony_ticker_file: Path = resources.joinpath("SONY.csv")

        data: list[str] = read_ticker_file(sony_ticker_file, ticker)

        assert len(data) > 0

    def test_raises(self: TestAssignment1) -> None:
        with pytest.raises(Assignment1Error) as exc_info:
            read_ticker_file("invalid_file_path", "ticker")
        assert exc_info.value.args[0] == "Error reading file from invalid_file_path"
        assert str(exc_info.value) == "Error reading file from invalid_file_path"
        assert isinstance(exc_info.value.error, FileNotFoundError)

    def test_mean(self: TestAssignment1) -> None:
        calculated_mean: float = mean([1, 2, 3, 4, 5])
        assert calculated_mean == 3.0

    def test_standard_deviation(self: TestAssignment1) -> None:
        calculated_standard_deviation: float = standard_deviation([1, 2, 3, 4, 5])
        assert calculated_standard_deviation == 1.4142135623730951

    def test_parse_stock_data_entry(self: TestAssignment1) -> None:
        raw_entry: str = "2016-01-04,2016,1,4,Monday,01,2016-01,24.45,24.8,24.31,24.73,2482200,24.73,0.0,24.73,24.73\n"

        entry: list[str] = StockDataEntry.parse_stock_data_entry(raw_entry)

        assert len(entry) == 16

    @pytest.mark.usefixtures("resources")
    def test_stock_data_from_ticker_file(
        self: TestAssignment1, resources: Path
    ) -> None:
        ticker: str = "SONY"
        sony_ticker_file: Path = resources.joinpath("SONY.csv")

        stock_data: StockData = StockData.from_ticker_file(sony_ticker_file, ticker)

        assert stock_data
