"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os
from pathlib import Path

ticker: str = "SPY"
input_dir: Path = Path(__file__).parent.absolute()
ticker_file: Path = Path(input_dir.joinpath(f"{ticker}.csv"))

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print("opened file for ticker: ", ticker)
    print(lines)
    """    your code for assignment 1 goes here
    """

except Exception as e:
    print(e)
    print("failed to read stock data for ticker: ", ticker)
