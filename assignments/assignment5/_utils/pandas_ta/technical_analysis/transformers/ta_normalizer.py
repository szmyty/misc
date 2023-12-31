from __future__ import annotations

from typing import List

import pandas as pd
from pandas_df_commons.extensions.functions import rolling_apply
from pandas_df_commons.indexing.decorators import (
    convert_series_as_data_frame,
    foreach_top_level_row_and_column,
)
from sklearn.preprocessing import StandardScaler


def ta_standardized(
    df: pd.DataFrame | pd.Series,
    period: int = 14,
    columns: str | List[str] = None,
    parallel=False,
) -> pd.DataFrame | pd.Series:
    @foreach_top_level_row_and_column()
    def f(df):
        if columns is not None:
            df = df[columns]

        scaled_window_df = rolling_apply(
            df, period, StandardScaler().fit_transform, parallel
        )
        return None  # FIXME

    return f(df)
