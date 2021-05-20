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

    def __init__(self, embeddings, hidden_size=150):
        """
        Inits the baseline model.

        :param embeddings: the embedding matrix (torch.nn.Embedding)
        :param hidden_size: hidden layer size
        """
        super().__init__()

        self.embeddings = embeddings
        self.seq_modules = nn.Sequential(
            nn.Linear(300, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
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

    main(model, train_dataloader, valid_dataloader, test_dataloader)
