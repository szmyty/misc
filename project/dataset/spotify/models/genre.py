from __future__ import annotations

from enum import IntEnum

from pandas import CategoricalDtype


class Genre(IntEnum):
    """Enum class of music genres."""

    METAL: int = 0
    COUNTRY: int = 1

    @staticmethod
    def dtype() -> CategoricalDtype:
        """Get the categorical dtype associated with the genre."""
        return CategoricalDtype(categories=[Genre.METAL, Genre.COUNTRY])

    def to_title(self: Genre) -> str:
        """Get the enum class as a title string.

        Returns:
            str: The enum class as a title string.
        """
        return self.name.title()

    @classmethod
    def from_label(cls: type[Genre], label: int | str, title: bool = False) -> Genre | str:
        """Get a Genre enum from a label.

        Args:
            label (int | str): The label's value.
            title (bool): Whether to return the class as a title.

        Returns:
            Genre | str: The enum representation of the genre class label.
        """
        return cls(int(label)) if not title else cls(int(label)).to_title()

    @classmethod
    def from_label_to_title(cls: type[Genre], label: int | str) -> str:
        """Get a Genre enum from a label.

        Args:
            label (int | str): The label's value.

        Returns:
            str: The genre label as a title.
        """
        return cls(int(label)).to_title()

