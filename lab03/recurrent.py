import torch
import torch.nn as nn
import numpy as np

from main import prepare_data, main

SEED = 7052020


class Recurrent(nn.Module):
    """
    Recurrent model for Stanford Sentiment Treebank sentiment analysis.

    rnn(150) -> rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)
    """

    def __init__(self, embeddings, recurrent_unit, num_layers=2,
                 bidirectional=False, hidden_size=150):
        """
        Inits the baseline model.

        :param embeddings: the embedding matrix (torch.nn.Embedding)
        :param recurrent_unit: torch.nn.RNN / torch.nn.LSTM / torch.nn.GRU
        :param num_layers: number of recurrent layers
        :param bidirectional: True for bidirectional recurrent units
        :param hidden_size: hidden layer size
        """
        super().__init__()

        self.embeddings = embeddings
        self.recurrent_module = recurrent_unit(
            300, hidden_size, num_layers=num_layers,
            bidirectional=bidirectional)
        self.seq_modules = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """Performs the forward pass."""
        emb = self.embeddings(x)
        recurrent_output, _ = self.recurrent_module(emb)
        output = self.seq_modules(recurrent_output[-1])

        return output


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dataloader, valid_dataloader, \
        test_dataloader, embeddings = prepare_data()

    model = Recurrent(embeddings, nn.RNN)
    # model = Recurrent(embeddings, nn.LSTM)
    # model = Recurrent(embeddings, nn.GRU)
    # model = Recurrent(embeddings, nn.GRU, num_layers=1)
    # model = Recurrent(embeddings, nn.GRU, hidden_size=200)
    # model = Recurrent(embeddings, nn.GRU, bidirectional=True)

    main(model, train_dataloader, valid_dataloader, test_dataloader, clip=0.25)
