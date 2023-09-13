from __future__ import annotations

from dataclasses import dataclass

from dataset.spotify.errors import SpotifyClientError


@dataclass
class Artwork:
    """Spotify Artwork Data Model"""

    height: int
    url: str
    width: int

    @classmethod
    def from_api(cls: type[Artwork], api_response: dict) -> Artwork:
        try:
            return cls(
                height=int(api_response.get("height")),
                url=api_response.get("url"),
                width=int(api_response.get("width")),
            )
        except Exception as error:
            raise SpotifyClientError(
                message=f"Error parsing {cls.__name__} "
                f"data model from API response.",
                error=error,
            ) from error
