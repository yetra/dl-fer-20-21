import torch
import torch.nn as nn
import torch.utils.data

import numpy as np

from data import NLPDataset, pad_collate, embedding_matrix


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

    for batch_num, (X, y, _) in enumerate(dataloader):
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


def evaluate(dataloader, model, loss_fn):
    """Performs one test loop iteration."""
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, '
          f'Avg loss: {test_loss:>8f} \n')


def main(seed=7052020, epochs=10):
    """Performs SST sentiment analysis using Baseline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = NLPDataset.from_csv('data/sst_train_raw.csv')
    embeddings = embedding_matrix(train_dataset.text_vocab, 300)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=10,
        shuffle=True, collate_fn=pad_collate)
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=NLPDataset.from_csv('data/sst_valid_raw.csv'),
        batch_size=32, shuffle=True, collate_fn=pad_collate)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=NLPDataset.from_csv('data/sst_test_raw.csv'),
        batch_size=32, shuffle=True, collate_fn=pad_collate)

    model = Baseline(embeddings)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(f'Epoch {epoch}\n-------------------------------')
        train(train_dataloader, model, loss_fn, optimizer)
        evaluate(valid_dataloader, model, loss_fn)

    print(f'Test data performance\n-------------------------------')
    evaluate(test_dataloader, model, loss_fn)


if __name__ == '__main__':
    main()
