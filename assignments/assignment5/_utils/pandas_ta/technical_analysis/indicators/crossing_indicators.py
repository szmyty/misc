from __future__ import annotations

import pandas as pd
import tulipy
from pandas_df_commons.indexing import get_columns
from pandas_df_commons.indexing.decorators import (
    foreach_top_level_column,
    foreach_top_level_row,
    rename_with_parameters,
)


@foreach_top_level_row
@foreach_top_level_column
@rename_with_parameters(
    function_name="crossany", parameter_names=[], output_names=["crossany"]
)
def ta_crossany(df: pd.DataFrame, real=None, real1=None, **kwargs) -> pd.DataFrame:
    """
    Crossany (math)
    """

    # get_columns always returns a dataframe
    data = get_columns(df, [real or df.columns[0], real1 or df.columns[1]])

    # result gets converted by rename_with_parameters to DataFrame
    columns = data.columns.tolist()
    values = data.values.astype("float64")
    return columns, tulipy.crossany(
        *[values[:, i].copy(order="C") for i in range(values.shape[1])]
    )


@foreach_top_level_row
@foreach_top_level_column
@rename_with_parameters(
    function_name="crossover", parameter_names=[], output_names=["crossover"]
)
def ta_crossover(df: pd.DataFrame, real=None, real1=None, **kwargs) -> pd.DataFrame:
    """
    Crossover (math)
    """

    # get_columns always returns a dataframe
    data = get_columns(df, [real or df.columns[0], real1 or df.columns[1]])

    # result gets converted by rename_with_parameters to DataFrame
    columns = data.columns.tolist()
    values = data.values.astype("float64")
    return columns, tulipy.crossover(
        *[values[:, i].copy(order="C") for i in range(values.shape[1])]
    )
