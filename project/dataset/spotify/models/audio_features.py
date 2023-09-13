from __future__ import annotations

from dataclasses import dataclass

from dataset.spotify.errors import SpotifyClientError
from dataset.spotify.models.artwork import Artwork


@dataclass
class AudioFeatures:
    """Spotify Audio Features Data Model"""

    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    id: str
    time_signature: int
    duration_ms: int

    @classmethod
    def from_api(cls: type[AudioFeatures], api_response: dict) -> AudioFeatures:
        try:
            return cls(
                danceability=float(api_response.get("danceability")),
                energy=float(api_response.get("energy")),
                key=int(api_response.get("key")),
                loudness=float(api_response.get("loudness")),
                mode=int(api_response.get("mode")),
                speechiness=float(api_response.get("speechiness")),
                acousticness=float(api_response.get("acousticness")),
                instrumentalness=float(api_response.get("instrumentalness")),
                liveness=float(api_response.get("liveness")),
                valence=float(api_response.get("valence")),
                tempo=float(api_response.get("tempo")),
                id=str(api_response.get("id")),
                time_signature=int(api_response.get("time_signature")),
                duration_ms=int(api_response.get("duration_ms")),
            )
        except Exception as error:
            raise SpotifyClientError(
                message=f"Error parsing {cls.__name__} "
                f"data model from API response.",
                error=error,
            ) from error
