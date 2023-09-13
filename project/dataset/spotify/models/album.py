from __future__ import annotations

from dataclasses import dataclass

from dataset.spotify.errors import SpotifyClientError
from dataset.spotify.models.artwork import Artwork


@dataclass
class Album:
    """Spotify Album Data Model"""

    href: str
    id: str
    name: str
    release_date: str
    total_tracks: int
    uri: str
    images: list[Artwork]

    @classmethod
    def from_api(cls: type[Album], api_response: dict) -> Album:
        try:
            return cls(
                href=api_response.get("href"),
                id=api_response.get("id"),
                name=api_response.get("name"),
                release_date=api_response.get("release_date"),
                total_tracks=int(api_response.get("total_tracks")),
                uri=api_response.get("uri"),
                images=[
                    Artwork.from_api(image) for image in api_response.get("images")
                ],
            )
        except Exception as error:
            raise SpotifyClientError(
                message=f"Error parsing {cls.__name__} "
                f"data model from API response.",
                error=error,
            ) from error
