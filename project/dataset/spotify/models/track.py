from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dataset.spotify.errors import SpotifyClientError
from dataset.spotify.models import Album, Artist, AudioFeatures, SpotifyColumns
from pandas import Series


@dataclass
class Track:
    """Spotify Track Data Model"""

    duration_ms: int
    href: str
    id: str
    name: str
    popularity: int
    uri: str
    album: Album
    artist: Artist
    audio_features: AudioFeatures

    @property
    def danceability(self):
        return self.audio_features.danceability

    @property
    def energy(self):
        return self.audio_features.energy

    @property
    def key(self):
        return self.audio_features.key

    @property
    def loudness(self):
        return self.audio_features.loudness

    @property
    def mode(self):
        return self.audio_features.mode

    @property
    def speechiness(self):
        return self.audio_features.speechiness

    @property
    def acousticness(self):
        return self.audio_features.acousticness

    @property
    def instrumentalness(self):
        return self.audio_features.instrumentalness

    @property
    def liveness(self):
        return self.audio_features.liveness

    @property
    def valence(self):
        return self.audio_features.valence

    @property
    def tempo(self):
        return self.audio_features.tempo

    @property
    def time_signature(self):
        return self.audio_features.time_signature

    @property
    def genre(self):
        try:
            return self.artist.genres[0]
        except IndexError:
            return np.nan

    @classmethod
    def from_api(cls: type[Track], api_response: dict) -> Track:
        try:
            return cls(
                duration_ms=int(api_response.get("duration_ms")),
                href=api_response.get("href"),
                id=api_response.get("id"),
                name=api_response.get("name"),
                popularity=int(api_response.get("popularity")),
                uri=api_response.get("uri"),
                album=Album.from_api(api_response.get("album")),
                artist=Artist.from_api(api_response.get("artist")),
                audio_features=AudioFeatures.from_api(
                    api_response.get("audio_features")
                ),
            )
        except Exception as error:
            raise SpotifyClientError(
                message=f"Error parsing {cls.__name__} "
                f"data model from API response.",
                error=error,
            ) from error

    def as_series(self: Track) -> Series:
        return Series(
            {
                SpotifyColumns.NAME: self.name,
                SpotifyColumns.DURATION: self.duration_ms,
                SpotifyColumns.POPULARITY: self.popularity,
                SpotifyColumns.DANCEABILITY: self.danceability,
                SpotifyColumns.ENERGY: self.energy,
                SpotifyColumns.KEY: self.key,
                SpotifyColumns.LOUDNESS: self.loudness,
                SpotifyColumns.MODE: self.mode,
                SpotifyColumns.SPEECHINESS: self.speechiness,
                SpotifyColumns.ACOUSTICNESS: self.acousticness,
                SpotifyColumns.INSTRUMENTALNESS: self.instrumentalness,
                SpotifyColumns.LIVENESS: self.liveness,
                SpotifyColumns.VALENCE: self.valence,
                SpotifyColumns.TEMPO: self.tempo,
                SpotifyColumns.TIME_SIGNATURE: self.time_signature,
                SpotifyColumns.GENRE: self.genre,
            }
        )


class Tracks(list[Track]):
    def __init__(self, tracks):
        super().__init__(tracks)

    def series(self):
        return [row.as_series() for row in self]
