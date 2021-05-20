import torch.nn as nn


class Recurrent(nn.Module):
    """
    Recurrent model for Stanford Sentiment Treebank sentiment analysis.

    rnn(150) -> rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)
    """

    def __init__(self, embeddings, recurrent_unit):
        """
        Inits the baseline model.

        :param embeddings: the embedding matrix (torch.nn.Embedding)
        :param recurrent_unit: torch.nn.RNN / torch.nn.LSTM / torch.nn.GRU
        """
        super().__init__()

        self.embeddings = embeddings
        self.recurrent_module = recurrent_unit(300, 150, num_layers=2)
        self.seq_modules = nn.Sequential(
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        """Performs the forward pass."""
        embeddings = self.embeddings(x)
        recurrent_output, _ = self.recurrent_module(embeddings)
        output = self.seq_modules(recurrent_output)

        return output
