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

    def __init__(self, embeddings, recurrent_unit, num_layers=2, dropout=0.,
                 bidirectional=False, hidden_size=150, activation=nn.ReLU):
        """
        Inits the baseline model.

        :param embeddings: the embedding matrix (torch.nn.Embedding)
        :param recurrent_unit: torch.nn.RNN / torch.nn.LSTM / torch.nn.GRU
        :param num_layers: number of recurrent layers
        :param dropout: if non-zero, dropout probability for recurrent modules
        :param bidirectional: True for bidirectional recurrent units
        :param hidden_size: hidden layer size
        :param activation: the activation function to use after nn.Linear
        """
        super().__init__()

        self.embeddings = embeddings
        self.recurrent_module = recurrent_unit(
            300, hidden_size, num_layers=num_layers,
            dropout=dropout, bidirectional=bidirectional)
        self.seq_modules = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
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
        test_dataloader, embeddings = prepare_data()  # freeze=False

    model = Recurrent(embeddings, nn.RNN)
    # model = Recurrent(embeddings, nn.RNN, num_layers=3)
    # model = Recurrent(embeddings, nn.RNN, num_layers=3, dropout=0.4)
    # model = Recurrent(embeddings, nn.LSTM)
    # model = Recurrent(embeddings, nn.LSTM, activation=nn.LeakyReLU)
    # model = Recurrent(embeddings, nn.GRU)
    # model = Recurrent(embeddings, nn.GRU, num_layers=1)
    # model = Recurrent(embeddings, nn.GRU, hidden_size=200)
    # model = Recurrent(embeddings, nn.GRU, bidirectional=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    main(model, optimizer, train_dataloader,
         valid_dataloader, test_dataloader, clip=0.25)  # clip=0.5
