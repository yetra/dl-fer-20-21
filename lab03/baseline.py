import torch
import torch.nn as nn
import numpy as np

from main import prepare_data, main

SEED = 7052020


class Baseline(nn.Module):
    """
    Baseline model for Stanford Sentiment Treebank sentiment analysis.

    avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)
    """

    def __init__(self, embeddings, hidden_size=150, activation=nn.ReLU):
        """
        Inits the baseline model.

        :param embeddings: the embedding matrix (torch.nn.Embedding)
        :param hidden_size: hidden layer size
        :param activation: the activation function to use after nn.Linear
        """
        super().__init__()

        self.embeddings = embeddings
        self.seq_modules = nn.Sequential(
            nn.Linear(300, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """Performs the forward pass."""
        emb_pooled = torch.mean(self.embeddings(x), dim=0)
        output = self.seq_modules.forward(emb_pooled)

        return output


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dataloader, valid_dataloader, \
        test_dataloader, embeddings = prepare_data()

    model = Baseline(embeddings)
    # model = Baseline(embeddings, hidden_size=100)
    # model = Baseline(embeddings, hidden_size=200)
    # model = Baseline(embeddings, activation=nn.LeakyReLU)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    main(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)
