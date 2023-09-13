from __future__ import annotations

from enum import IntEnum

from pandas import CategoricalDtype


class Modality(IntEnum):
    """Enum class of modality of a track.

    The type of scale from which a track's melodic content is derived.
    """

    MINOR: int = 0
    MAJOR: int = 1

    @staticmethod
    def dtype() -> CategoricalDtype:
        """Get the categorical dtype associated with the track modality."""
        return CategoricalDtype(categories=[Modality.MINOR, Modality.MAJOR])

    def to_title(self: Modality) -> str:
        """Get the enum class as a title string.

        Returns:
            str: The enum class as a title string.
        """
        return self.name.title()

    @classmethod
    def from_label(
        cls: type[Modality], label: int, title: bool = False
    ) -> Modality | str:
        """Get a Modality enum from a label.

        Args:
            label (int): The label's value.
            title (bool): Whether to return the class as a title.

        Returns:
            Modality | str: The enum representation of the modality class label.
        """
        return cls(label) if not title else cls(label).to_title()
