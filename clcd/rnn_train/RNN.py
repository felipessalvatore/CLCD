import torch
import torch.nn as nn


class RNN(nn.Module):

    name = "rnn"

    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.rnn = nn.LSTM(config.embedding_dim, config.rnn_dim, config.layers)
        self.fc = nn.Linear(config.rnn_dim, config.output_dim)

    def forward(self, x):
        """
        Apply the model to the input x

        :param x: indices of the sentence
        :type x: torch.Tensor(shape=[sent len, batch size]
                              dtype=torch.int64)
        """
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        # hidden = [config.layers, batch size, hid dim]

        self.output = output

        out = output[-1]
        out = self.fc(out)
        return out

    def predict(self, x):
        out = self.forward(x)
        softmax = nn.Softmax(dim=1)
        out = softmax(out)
        indices = torch.argmax(out, 1)
        return indices

    def evaluate_bach(self,
                      batch,
                      device):
        labels = batch.label.type('torch.LongTensor')
        labels = labels.to(device)
        batch_text = batch.text.to(device)
        prediction = self.predict(batch_text)
        correct = torch.sum(torch.eq(prediction, labels)).float()
        accuracy = float(correct / labels.shape[0])
        return accuracy, prediction, labels
