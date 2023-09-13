#!/usr/env/bin python3
""""""
from __future__ import annotations

from dataclasses import dataclass

from ...spotify.errors import SpotifyClientError


@dataclass
class Artist:
    """Spotify Artist Data Model"""

    href: str
    id: str
    name: str
    uri: str
    popularity: str
    followers: int
    genres: list[str]

    @classmethod
    def from_api(cls: type[Artist], api_response: dict) -> Artist:
        """

        Args:
            api_response:

        Returns:

        """
        try:
            return cls(
                href=api_response.get("href"),
                id=api_response.get("id"),
                name=api_response.get("name"),
                uri=api_response.get("uri"),
                popularity=api_response.get("popularity"),
                genres=api_response.get("genres"),
                followers=int(api_response.get("followers").get("total")),
            )
        except Exception as error:
            raise SpotifyClientError(
                message="Error parsing Artist data model from API response.",
                error=error,
            ) from error
