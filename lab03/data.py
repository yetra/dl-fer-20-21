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
        self.max_size = max_size
        self.min_freq = min_freq

        self.stoi = self._stoi(frequencies)
        self.itos = {i: s for s, i in self.stoi.items()}

    def _stoi(self, frequencies):
        """
        Creates a mapping of dataset tokens to integers.

        :param frequencies: a dict of dataset token frequencies
        :return: the token->int map
        """
        stoi = {'<PAD>': 0, '<UNK>': 1}

        sorted_tokens = sorted(frequencies, key=frequencies.get)

        for i, token in enumerate(sorted_tokens):
            if len(stoi) <= self.max_size and frequencies[token] >= self.min_freq:
                stoi[token] = i + 2

        return stoi
