import torch
import torch.nn as nn
import torch.utils.data

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data import NLPDataset, pad_collate, embedding_matrix

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


def train(dataloader, model, loss_fn, optimizer, clip=None):
    """Performs one train loop iteration."""
    size = len(dataloader.dataset)

    for batch_num, (X, y, _) in enumerate(dataloader):
        # compute prediction and loss
        output = model(X)
        loss = loss_fn(torch.squeeze(output), y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping (optional)
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if batch_num % 100 == 0:
            loss, current = loss.item(), batch_num * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def evaluate(dataloader, model, loss_fn):
    """Performs one test loop iteration."""
    y_true, y_pred = [], []
    loss = 0

    with torch.no_grad():
        for X, y, _ in dataloader:
            output = torch.squeeze(model(X))
            loss += loss_fn(output, y).item()

            pred = torch.sigmoid(output).round()
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy())

    loss /= len(dataloader.dataset)
    acc = accuracy_score(y_true, y_pred)

    print(f'Test Error:\n'
          f'\tAccuracy: {(100 * acc):>0.1f}%\n'
          f'\tF1 score: {f1_score(y_true, y_pred):>8f}\n'
          f'\tAvg loss: {loss:>8f}\n'
          f'Confusion matrix:\n{confusion_matrix(y_true, y_pred)}\n')

    return loss, acc


def main(seed=7052020, epochs=10):
    """Performs SST sentiment analysis using Baseline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = NLPDataset.from_csv(TRAIN_PATH)
    text_vocab, label_vocab = train_dataset.text_vocab, train_dataset.label_vocab
    embeddings = embedding_matrix(text_vocab, 300)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=10,
        shuffle=True, collate_fn=pad_collate)
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=NLPDataset.from_csv(VALID_PATH, text_vocab, label_vocab),
        batch_size=32, shuffle=True, collate_fn=pad_collate)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=NLPDataset.from_csv(TEST_PATH, text_vocab, label_vocab),
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
