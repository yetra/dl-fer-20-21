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
EMB_PATH = 'data/sst_glove_6b_300d.txt'


def prepare_data(batch_sizes=(10, 32, 32), freeze=True):
    """
    Prepares SST data.

    :param batch_sizes: the batch sizes for train, valid, and test sets
    :param freeze: the embeddings won't be updated during training if True
    :return: train, valid, and test DataLoaders; and the embeddings
    """
    train_dataset = NLPDataset.from_csv(TRAIN_PATH)
    text_vocab, label_vocab = train_dataset.text_vocab, train_dataset.label_vocab
    embeddings = embedding_matrix(text_vocab, 300, freeze, EMB_PATH)

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
    total_loss = 0

    for batch_num, (X, y, _) in enumerate(dataloader):
        # compute prediction and loss
        output = model(X)
        loss = loss_fn(torch.squeeze(output), y)
        total_loss += loss

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

    return total_loss / size


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


def plot_performance(train_losses, valid_losses, valid_accs, epochs):
    """
    Plots validation set loss and accuracy per epoch.

    :param train_losses: the training set losses per epoch
    :param valid_losses: the validation set losses per epoch
    :param valid_accs: the validation set accuracies per epoch
    :param epochs: the number of epochs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.4)

    ax1.plot(range(1, epochs + 1), train_losses, label='train')
    ax1.plot(range(1, epochs + 1), valid_losses, label='valid')
    ax1.set_title('train and valid loss per epoch')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend()

    ax2.plot(range(1, epochs + 1), valid_accs, label='valid')
    ax2.set_title('valid accuracy per epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')

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
    train_losses, valid_losses, valid_accs = [], [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch}\n-------------------------------')
        loss = train(train_dataloader, model, loss_fn, optimizer, clip)
        train_losses.append(loss)

        loss, acc = evaluate(valid_dataloader, model, loss_fn)
        valid_losses.append(loss)
        valid_accs.append(acc)

    plot_performance(train_losses, valid_losses, valid_accs, epochs)

    print(f'Test set performance\n-------------------------------')
    evaluate(test_dataloader, model, loss_fn)
