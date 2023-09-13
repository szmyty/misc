"""Assignment 4 Constants module.

Alan Szmyt
Class: CS 677
Date: April 4th, 2023
Assignment 4
Description:
Constants to be used throughout the assignment 4.
"""

# Column names.
COL_CPK: str = "creatinine_phosphokinase"
COL_SERUM_CREATININE: str = "serum_creatinine"
COL_SERUM_SODIUM: str = "serum_sodium"
COL_PLATELETS: str = "platelets"
COL_DEATH_EVENT: str = "DEATH_EVENT"

# Feature columns.
FEATURE_COLS: list[str] = [
    COL_CPK,
    COL_SERUM_CREATININE,
    COL_SERUM_SODIUM,
    COL_PLATELETS,
]

# Initial columns.
INITIAL_COLS: list[str] = FEATURE_COLS + [COL_DEATH_EVENT]
