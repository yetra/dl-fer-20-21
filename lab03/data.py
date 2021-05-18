from dataclasses import dataclass


@dataclass
class NLPDataItem:
    """Class representing an item in a dataset of texts."""
    text: str
    label: str
