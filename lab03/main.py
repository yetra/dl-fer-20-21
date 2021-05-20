import torch
import torch.nn as nn
import torch.utils.data

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data import NLPDataset, pad_collate, embedding_matrix

# paths to datasets
TRAIN_PATH = 'data/sst_train_raw.csv'
VALID_PATH = 'data/sst_valid_raw.csv'
TEST_PATH = 'data/sst_test_raw.csv'


def prepare_data(batch_sizes=(10, 32, 32), freeze=True):
    """
    Prepares SST data.

    :param batch_sizes: the batch sizes for train, valid, and test sets
    :param freeze: the embeddings won't be updated during training if True
    :return: train, valid, and test DataLoaders; and the embeddings
    """
    train_dataset = NLPDataset.from_csv(TRAIN_PATH)
    text_vocab, label_vocab = train_dataset.text_vocab, train_dataset.label_vocab
    embeddings = embedding_matrix(text_vocab, 300, freeze)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_sizes[0],
        shuffle=True, collate_fn=pad_collate)
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=NLPDataset.from_csv(VALID_PATH, text_vocab, label_vocab),
        batch_size=batch_sizes[1], shuffle=True, collate_fn=pad_collate)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=NLPDataset.from_csv(TEST_PATH, text_vocab, label_vocab),
        batch_size=batch_sizes[2], shuffle=True, collate_fn=pad_collate)

    return train_dataloader, valid_dataloader, test_dataloader, embeddings


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
            loss, current = loss.item(), batch_num * X.shape[1]
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

    print(f'Validation set performance\n'
          f'Accuracy: {(100 * acc):>0.1f}%\n'
          f'F1 score: {f1_score(y_true, y_pred):>8f}\n'
          f'Avg loss: {loss:>8f}\n'
          f'Confusion matrix:\n{confusion_matrix(y_true, y_pred)}\n')

    return loss, acc


def plot_performance(loss_vals, acc_vals, epochs):
    """
    Plots validation set loss and accuracy per epoch.

    :param loss_vals: the losses per epoch
    :param acc_vals: the accuracies per epoch
    :param epochs: the number of epochs
    """
    plt.plot(range(epochs), loss_vals, label='loss')
    plt.plot(range(epochs), acc_vals, label='acc')
    plt.title('validation set loss and acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def main(model, optimizer, train_dataloader, valid_dataloader,
         test_dataloader, epochs=5, clip=None):
    """
    Performs SST sentiment analysis using the given model.

    :param model: the model for sentiment analysis
    :param optimizer: the optimizer to use
    :param train_dataloader: training set DataLoader
    :param valid_dataloader: training set DataLoader
    :param test_dataloader: training set DataLoader
    :param epochs: the number of epochs
    :param clip: max gradient norm for gradient clipping
    """
    loss_fn = nn.BCEWithLogitsLoss()
    loss_vals, acc_vals = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch}\n-------------------------------')
        train(train_dataloader, model, loss_fn, optimizer, clip)
        loss, acc = evaluate(valid_dataloader, model, loss_fn)
        loss_vals.append(loss)
        acc_vals.append(acc)

    plot_performance(loss_vals, acc_vals, epochs)

    print(f'Test set performance\n-------------------------------')
    evaluate(test_dataloader, model, loss_fn)
