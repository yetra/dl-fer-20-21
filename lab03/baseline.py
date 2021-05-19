import torch
import torch.nn as nn
import numpy as np


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


def train(dataloader, model, loss_fn, optimizer):
    """Performs one train loop iteration."""
    size = len(dataloader.dataset)

    for batch_num, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            loss, current = loss.item(), batch_num * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
