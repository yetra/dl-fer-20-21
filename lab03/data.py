from dataclasses import dataclass


@dataclass
class NLPDataItem:
    """Class representing an item in a dataset of texts."""
    text: str
    label: str


class Vocab:
    """Class for transforming tokens into integers."""

    def __init__(self, frequencies, max_size, min_freq):
        """
        Inits a Vocab instance.

        :param frequencies: a dict of dataset token frequencies
        :param max_size: the largest number of tokens that can be stored
        :param min_freq: the minimal frequency needed for storing a token
        """
        self.frequencies = frequencies
        self.max_size = max_size
        self.min_freq = min_freq

        self.stoi = self._stoi()

    def _stoi(self):
        """Creates a mapping of dataset tokens to integers."""
        stoi = {'<PAD>': 0, '<UNK>': 1}

        sorted_tokens = sorted(self.frequencies, key=self.frequencies.get)
        stoi.update({token: (i + 2) for i, token in enumerate(sorted_tokens)})

        return stoi
