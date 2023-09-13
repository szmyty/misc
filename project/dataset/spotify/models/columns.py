"""

"""
from __future__ import annotations

from collections import defaultdict
from enum import StrEnum, unique

import numpy as np
import pandas as pd


@unique
class SpotifyColumns(StrEnum):
    """Enum representation for the Spotify dataset columns."""

    NAME: str = "name"
    DURATION: str = "duration"
    POPULARITY: str = "popularity"
    DANCEABILITY: str = "danceability"
    ENERGY: str = "energy"
    KEY: str = "key"
    LOUDNESS: str = "loudness"
    MODE: str = "mode"
    SPEECHINESS: str = "speechiness"
    ACOUSTICNESS: str = "acousticness"
    INSTRUMENTALNESS: str = "instrumentalness"
    LIVENESS: str = "liveness"
    VALENCE: str = "valence"
    TEMPO: str = "tempo"
    TIME_SIGNATURE: str = "time_signature"
    GENRE: str = "genre"

    @classmethod
    def columns(cls: type[SpotifyColumns]) -> list[str]:
        """Get a list of the columns for the dataframe.

        Returns:
            list[str]: The list of columns.
        """
        return list(cls)

    @classmethod
    def from_column(cls: type[SpotifyColumns], column: int) -> SpotifyColumns:
        """Get a SpotifyColumn enum from a column.

        Args:
            column (int): The column's value.

        Returns:
            SpotifyColumns: The enum representation of the spotify column.
        """
        return cls(column)

    @staticmethod
    def dtype() -> dict:
        """Get the dtype mapping for the columns.

        Returns:
            dict: The dtype mapping for the columns.
        """
        from dataset.spotify.models import Genre
        from dataset.spotify.models.modality import Modality

        # noinspection PyTypeChecker
        return defaultdict(
            np.float64,
            {
                SpotifyColumns.NAME: pd.StringDtype(),
                SpotifyColumns.DURATION: np.int64,
                SpotifyColumns.POPULARITY: np.int64,
                SpotifyColumns.KEY: np.int64,
                SpotifyColumns.MODE: Modality.dtype(),
                SpotifyColumns.GENRE: Genre.dtype(),
                SpotifyColumns.TIME_SIGNATURE: np.int64,
            },
        )

    @staticmethod
    def feature_columns() -> list[SpotifyColumns]:
        return [
            SpotifyColumns.DURATION,
            SpotifyColumns.POPULARITY,
            SpotifyColumns.DANCEABILITY,
            SpotifyColumns.ENERGY,
            SpotifyColumns.KEY,
            SpotifyColumns.LOUDNESS,
            SpotifyColumns.MODE,
            SpotifyColumns.SPEECHINESS,
            SpotifyColumns.ACOUSTICNESS,
            SpotifyColumns.INSTRUMENTALNESS,
            SpotifyColumns.LIVENESS,
            SpotifyColumns.VALENCE,
            SpotifyColumns.TEMPO,
            SpotifyColumns.TIME_SIGNATURE,
        ]
