import torch
import torch.nn as nn
import torch.utils.data

# paths to datasets
TRAIN_PATH = 'data/sst_train_raw.csv'
VALID_PATH = 'data/sst_valid_raw.csv'
TEST_PATH = 'data/sst_test_raw.csv'


class Baseline(nn.Module):
    """
    Baseline model for Stanford Sentiment Treebank sentiment analysis.

    avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)
    """

    def __init__(self, embeddings):
        """
        Inits the baseline model.

        :param embeddings: the embedding matrix (torch.nn.Embedding)
        """
        super().__init__()

        self.embeddings = embeddings
        self.seq_modules = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        """Performs the forward pass."""
        emb_pooled = torch.mean(self.embeddings(x), dim=0)
        output = self.seq_modules.forward(emb_pooled)

        return output
