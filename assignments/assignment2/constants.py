"""Assignment2 Constants module.

Alan Szmyt
Class: CS 677
Date: March 28th, 2023
Assignment #2
Description:
Constants to be used throughout assignment 2.
"""
# Stock ticker abbreviations.
SPY_TICKER: str = "SPY"
SONY_TICKER: str = "SONY"

# Statistics summary table keys.
W_KEY: str = "W"
TICKER_KEY: str = "ticker"
TP_KEY: str = "TP"
FP_KEY: str = "FP"
TN_KEY: str = "TN"
FN_KEY: str = "FN"
ACCURACY_KEY: str = "accuracy"
TPR: str = "TPR"
TNR: str = "TNR"

# Stock data column keys.
DATE_KEY: str = "Date"
RETURN_KEY: str = "Return"
YEAR_KEY: str = "Year"
TRUE_LABEL_KEY: str = "True Label"
BUY_AND_HOLD_KEY: str = "Buy and Hold"
ENSEMBLE_KEY: str = "Ensemble"
ENSEMBLE_ORACLE_KEY: str = "Ensemble Oracle"
W_ORACLE: str = "W* Oracle"

# Colors for the 'up day'/'down day' symbols.
UP_DAY_COLOR: str = "#54b254"
DOWN_DAY_COLOR: str = "#e43d30"

# Global table styles.
TABLE_STYLES: list[dict] = [
    dict(selector="td, th", props=[("text-align", "center")]),
]

# Global styler options.
STYLER_OPTIONS: dict = dict(
    uuid_len=0, cell_ids=False, precision=5, table_styles=TABLE_STYLES
)
