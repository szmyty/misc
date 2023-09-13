"""Assignment3 Constants module.

Alan Szmyt
Class: CS 677
Date: April 4th, 2023
Assignment #3
Description:
Constants to be used throughout assignment 3.
"""
from utils import latexify_math_var

# Filenames.
X_TRAIN: str = "x_train"
Y_TRAIN: str = "y_train"
X_TEST: str = "x_test"
Y_TEST: str = "y_test"
BANKNOTE_DATASET: str = "banknote_dataset"
GOOD_BILLS: str = "good_bills"
FAKE_BILLS: str = "fake_bills"
BANKNOTES: str = "banknotes"

# Colors
GOOD_BILLS_GREEN: tuple[float, float, float] = (84 / 255, 178 / 255, 84 / 255)
FAKE_BILLS_RED: tuple[float, float, float] = (228 / 255, 61 / 255, 48 / 255)

# Column names.
COL_CLASS: str = "class"
COL_COLOR: str = "color"
COL_PREDICTION: str = "prediction"
COL_F1: str = "f1"
COL_F2: str = "f2"
COL_F3: str = "f3"
COL_F4: str = "f4"
COL_F1_MEAN: str = "f1_mean"
COL_F2_MEAN: str = "f2_mean"
COL_F3_MEAN: str = "f3_mean"
COL_F4_MEAN: str = "f4_mean"
COL_F1_STD: str = "f1_std"
COL_F2_STD: str = "f2_std"
COL_F3_STD: str = "f3_std"
COL_F4_STD: str = "f4_std"
COL_ALL: str = "all"

FEATURE_COLS: list[str] = [
    COL_F1,
    COL_F2,
    COL_F3,
    COL_F4,
]

INITIAL_COLS: list[str] = FEATURE_COLS + [COL_CLASS]

STATS_COLS: list[str] = [
    COL_F1_MEAN,
    COL_F1_STD,
    COL_F2_MEAN,
    COL_F2_STD,
    COL_F3_MEAN,
    COL_F3_STD,
    COL_F4_MEAN,
    COL_F4_STD,
]

# Column name to latex mapping for features.
features_to_latex: dict = {
    COL_F1: latexify_math_var("f_1"),
    COL_F2: latexify_math_var("f_2"),
    COL_F3: latexify_math_var("f_3"),
    COL_F4: latexify_math_var("f_4"),
}

features_to_plot: dict = {
    COL_F1: latexify_math_var("f_1", False),
    COL_F2: latexify_math_var("f_2", False),
    COL_F3: latexify_math_var("f_3", False),
    COL_F4: latexify_math_var("f_4", False),
}

# Column name to latex mapping for stats.
stats_to_latex: dict = {
    COL_F1_MEAN: latexify_math_var(r"\mu(f_1)"),
    COL_F1_STD: latexify_math_var(r"\sigma(f_1)"),
    COL_F2_MEAN: latexify_math_var(r"\mu(f_2)"),
    COL_F2_STD: latexify_math_var(r"\sigma(f_2)"),
    COL_F3_MEAN: latexify_math_var(r"\mu(f_3)"),
    COL_F3_STD: latexify_math_var(r"\sigma(f_3)"),
    COL_F4_MEAN: latexify_math_var(r"\mu(f_4)"),
    COL_F4_STD: latexify_math_var(r"\sigma(f_4)"),
}
