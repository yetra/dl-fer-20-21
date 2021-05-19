import torch.nn as nn


class Baseline(nn.Module):
    """
    Baseline model for Stanford Sentiment Treebank sentiment analysis.

    avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)
    """

    def __init__(self):
        """Inits the baseline model."""
        super().__init__()

        self.modules = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        """Returns a prediction for the given input."""
        logits = self.modules.forward(x)

        return nn.BCEWithLogitsLoss(logits)
