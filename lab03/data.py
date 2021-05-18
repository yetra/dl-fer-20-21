import csv
from collections import defaultdict
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
    text: [str]
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

    @staticmethod
    def from_csv(file_name):
        """
        Constructs an NLPDataset from the given CSV file.

        :param file_name: path to the dataset CSV file
        :return: an NLPDataset instance
        """
        items = []

        with open(file_name) as data_file:
            reader = csv.reader(data_file)

            for (text, label) in reader:
                items.append(NLPDataItem(text.split(), label.strip()))

        text_vocab = Vocab.from_csv(file_name)
        label_vocab = Vocab.from_csv(file_name, for_labels=True)

        return NLPDataset(items, text_vocab, label_vocab)

    def __len__(self):
        """Returns the number of NLPDataItems in the dataset."""
        return len(self.items)

    def __getitem__(self, index):
        """Returns an int representation of the item at the given index."""
        item = self.items[index]

        return (self._text_vocab.encode(item.text),
                self._label_vocab.encode(item.label))


class Vocab:
    """Class for transforming tokens into integers."""

    def __init__(self, frequencies, max_size=-1, min_freq=0, for_labels=False):
        """
        Inits a Vocab instance.

        :param frequencies: a dict of dataset token frequencies
        :param max_size: the largest number of tokens that can be stored
                         (-1 if all tokens should be stored)
        :param min_freq: the minimal frequency needed for storing a token
        :param for_labels: True if Vocab is for labels else False
        """
        self.max_size = len(frequencies) if max_size == -1 else max_size
        self.min_freq = max(0, min_freq)

        self.stoi = self._stoi(frequencies, for_labels)
        self.itos = {i: s for s, i in self.stoi.items()}

    def _stoi(self, frequencies, for_labels):
        """
        Creates a mapping of dataset tokens to integers.

        :param frequencies: a dict of dataset token frequencies
        :param for_labels: True if Vocab is for labels else False
        :return: the token->int map
        """
        stoi = {_PAD_: 0, _UNK_: 1} if not for_labels else {}
        shift = len(stoi)  # for correct indexing if not for_labels
        self.max_size += shift

        sorted_tokens = sorted(frequencies, key=frequencies.get, reverse=True)

        for i, token in enumerate(sorted_tokens):
            if len(stoi) == self.max_size:
                break

            if frequencies[token] >= self.min_freq:
                stoi[token] = i + shift

        return stoi

    def encode(self, tokens):
        """
        Maps the given tokens to integers.

        :param tokens: a list of tokens to map (or a single token)
        :return: the mapped tokens (torch.Tensor)
        """
        if isinstance(tokens, str):
            key = tokens if tokens in self.stoi else _UNK_
            return torch.tensor([self.stoi[key]])

        return torch.tensor([self.stoi.get(token, self.stoi[_UNK_])
                             for token in tokens])

    @staticmethod
    def from_csv(file_name, max_size=-1, min_freq=0, for_labels=False):
        """
        Constructs a Vocab instance from the given CSV file.

        :param file_name: path to the dataset CSV file
        :param max_size: the largest number of tokens that can be stored
                         (-1 if all tokens should be stored)
        :param min_freq: the minimal frequency needed for storing a token
        :param for_labels: True if Vocab is for labels else False
        :return: a Vocab instance
        """
        frequencies = defaultdict(int)

        with open(file_name) as data_file:
            reader = csv.reader(data_file)

            for (text, label) in reader:
                if not for_labels:
                    tokens = text.split()

                    for token in tokens:
                        frequencies[token] += 1
                else:
                    frequencies[label.strip()] += 1

        return Vocab(frequencies, max_size, min_freq, for_labels)


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


def pad_collate(batch, padding_value=0.0):
    """
    Collates and pads batch data.

    :param batch: a list of NLPDataItems returned by `NLPDataset.__getitem__`
    :param padding_value: the value with which to pad the data
    :return: tensors representing the input batch
    """
    texts, labels = zip(*batch)

    return (nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=padding_value),
            torch.tensor(labels),
            torch.tensor([len(text) for text in texts]))


if __name__ == '__main__':
    batch_size = 2
    shuffle = False

    train_dataset = NLPDataset.from_csv('data/sst_train_raw.csv')
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=shuffle, collate_fn=pad_collate)

    texts, labels, lengths = next(iter(train_data_loader))

    print(f'Texts: {texts}')
    print(f'Labels: {labels}')
    print(f'Lengths: {lengths}')
