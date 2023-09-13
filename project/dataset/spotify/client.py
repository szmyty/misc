from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from dataset.spotify.models import Genre, SpotifyColumns, Track, Tracks
from loguru import logger
from pandas import DataFrame, Series
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyClient(Spotify):
    exclude_track_attrs_list: list[str] = [
        "external_urls",
        "disc_number",
        "explicit",
        "external_ids",
        "is_local",
        "track_number",
        "type",
        "preview_url",
        "is_playable",
    ]

    exclude_track_attrs: str = ",".join(exclude_track_attrs_list)

    def __init__(self: SpotifyClient, **kwargs) -> None:
        self.auth_manager: SpotifyClientCredentials = SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        )
        super().__init__(auth_manager=self.auth_manager, **kwargs)

    def download_playlist(
        self: SpotifyClient, playlist_id: str, limit=sys.maxsize
    ) -> list[Track]:
        offset = 0
        tracks = []

        while True:
            spots_left = limit - len(tracks)
            request_limit = 100 if spots_left >= 100 else spots_left

            response: list[Track] = [
                Track.from_api(
                    dict(
                        item.get("track"),
                        **{
                            "audio_features": self.audio_features(
                                tracks=item.get("track").get("id")
                            )[0],
                            "artist": self.artist(
                                artist_id=item.get("track").get("artists")[0]["id"]
                            ),
                        },
                    )
                )
                for item in self.playlist_items(
                    playlist_id=playlist_id,
                    fields=f"items(track(!{self.exclude_track_attrs}))",
                    limit=request_limit,
                    offset=offset,
                    market="US",
                    additional_types=("track",),
                ).get("items")
            ]
            tracks.extend(response)

            offset = offset + len(response)
            if len(tracks) >= limit or len(response) == 0:
                break
        return tracks

    def download_playlists(self: SpotifyClient, playlists: list[str]) -> list[Track]:
        tracks = []
        for playlist in playlists:
            logger.info(f"Downloading playlist {playlist}")
            tracks.extend(self.download_playlist(playlist_id=playlist))
        return tracks


def download_training_playlists():
    tracks_dataset_file: Path = Path(__file__).parent.resolve().joinpath("tracks.csv")

    client: SpotifyClient = SpotifyClient(requests_timeout=45)

    heavy_metal_playlists: list[str] = [
        "https://open.spotify.com/playlist/37i9dQZF1DX9qNs32fujYe",
        "https://open.spotify.com/playlist/37i9dQZF1DWWOaP4H0w5b0",
    ]

    country_playlists: list[str] = [
        "https://open.spotify.com/playlist/37i9dQZF1DWZBCPUIUs2iR",
        "https://open.spotify.com/playlist/37i9dQZF1DXadasIcsfbqh",
    ]

    heavy_metal_tracks: Tracks = Tracks(
        client.download_playlists(playlists=heavy_metal_playlists)
    )
    for track in heavy_metal_tracks:
        track.artist.genres = [Genre.METAL]

    country_tracks: Tracks = Tracks(
        client.download_playlists(playlists=country_playlists)
    )
    for track in country_tracks:
        track.artist.genres = [Genre.COUNTRY]

    dataset_tracks: Tracks = Tracks(heavy_metal_tracks + country_tracks)

    dataset: DataFrame = DataFrame(data=dataset_tracks.series())

    dataset.to_csv(
        path_or_buf=tracks_dataset_file,
        sep=",",
        header=True,
        index=False,
        index_label=None,
        mode="w",
        encoding="utf-8",
        compression="infer",
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors="strict",
    )

    print(f"Exported Spotify dataset to {tracks_dataset_file}.")


def download_test_playlist():
    tracks_dataset_file: Path = (
        Path(__file__).parent.resolve().joinpath("unknown_tracks.csv")
    )

    client: SpotifyClient = SpotifyClient(requests_timeout=45)

    heavy_metal_tracks: Tracks = Tracks(
        client.download_playlist(
            playlist_id="https://open.spotify.com/playlist/37i9dQZF1EIetewBshGEPK"
        )
    )
    for track in heavy_metal_tracks:
        track.artist.genres = [Genre.METAL]

    dataset_tracks: Tracks = Tracks(heavy_metal_tracks)

    dataset: DataFrame = DataFrame(data=dataset_tracks.series())

    dataset = dataset.drop([SpotifyColumns.GENRE, SpotifyColumns.MODE], axis=1)

    dataset.to_csv(
        path_or_buf=tracks_dataset_file,
        sep=",",
        header=True,
        index=False,
        index_label=None,
        mode="w",
        encoding="utf-8",
        compression="infer",
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors="strict",
    )

    print(f"Exported Spotify test dataset to {tracks_dataset_file}.")


if __name__ == "__main__":
    download_test_playlist()
