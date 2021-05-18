from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data

# special tokens
_PAD_ = '<PAD>'
_UNK_ = '<UNK>'


@dataclass
class NLPDataItem:
    """Class representing an item in a dataset of texts."""
    text: str
    label: str


class NLPDataset(torch.utils.data.Dataset):
    """Class representing an NLP dataset."""

    def __init__(self, items, text_vocab, label_vocab):
        """
        Inits a NLPDataset instance.

        :param items: the NLPDataItem instances in the dataset
        :param text_vocab: the Vocab for text data
        :param label_vocab: the Vocab for the corresponding labels
        """
        self.items = items

        self._text_vocab = text_vocab
        self._label_vocab = label_vocab

    def __getitem__(self, index):
        pass


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
        stoi = {_PAD_: 0, _UNK_: 1}

        sorted_tokens = sorted(frequencies, key=frequencies.get)

        for i, token in enumerate(sorted_tokens):
            if len(stoi) == self.max_size:
                break

            if frequencies[token] >= self.min_freq:
                stoi[token] = i + 2

        return stoi

    def encode(self, tokens):
        """
        Maps the given tokens to integers.

        :param tokens: a list of tokens to map (or a single token)
        :return: the mapped tokens (torch.Tensor)
        """
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, _UNK_))

        return torch.tensor([self.stoi.get(token, _UNK_) for token in tokens])


def embedding_matrix(vocab, emb_length, file_name=None):
    """
    Builds an embedding matrix for the given vocab.

    If file_name is provided, embeddings are loaded from the file
    (missing embeddings are initialized randomly).

    If no file_name is provided, the matrix is randomly initialized
    from the standard normal distribution.

    :param vocab: a Vocab instance
    :param emb_length: length of an embedding vector
    :param file_name: the file containing the embeddings
    :return: the embedding matrix (torch.nn.Embedding)
    """
    emb_matrix = torch.randn((len(vocab), emb_length))
    emb_matrix[0] = torch.zeros(emb_length)

    if file_name:
        with open(file_name) as emb_file:
            for line in emb_file:
                token, emb_string = line.split(maxsplit=1)
                index = vocab.stoi[token]

                emb_matrix[index] = torch.tensor(map(float, emb_string.split()))

    return nn.Embedding.from_pretrained(emb_matrix, padding_idx=0)
