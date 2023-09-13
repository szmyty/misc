from __future__ import annotations


class SpotifyClientError(Exception):
    """Custom Exception for the Spotify Client."""

    def __init__(
        self: SpotifyClientError,
        message: str = "Error using the Spotify API client.",
        error: Exception | None = None,
    ) -> None:
        """Instantiate custom exception for the Spotify Client.

        Args:
            message (str): Description of the error that was raised.
            error (Exception | None): Optional Exception to associate with the instance.
        """
        self.error: Exception | None = error
        self.message: str = message
        super().__init__(self.message)
