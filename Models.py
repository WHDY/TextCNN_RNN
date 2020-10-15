import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTextClassifier(nn.Module):
    def __init__(self, embeddings, numFilter, filterShape, dropout, classes):
        super(CNNTextClassifier, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=numFilter,  kernel_size=(k, embeddings.size(1))) for k in filterShape])
        self.dropot = dropout
        self.fc = nn.Linear(in_features=numFilter*len(filterShape), out_features=classes, bias=True)

    def forward(self, input):
        batchTextVectors = self.embeddings(input)
        batchTextVectors = batchTextVectors.unsqueeze(1)
        tensors = []
        for conv in self.conv:
            tensor = conv(batchTextVectors)
            tensor = tensor.squeeze(3)
            tensor = F.relu(tensor)
            tensor = F.max_pool1d(tensor, tensor.size(2))
            tensor = tensor.squeeze(2)
            tensors.append(tensor)
        out = torch.cat(tensors, 1)

        if self.training:
            out = F.dropout(out, self.dropot)

        out = self.fc(out)
        return out


class LSTMTextClassifier(nn.Module):
    def __init__(self, embeddings, numLayers, hLayerSzie, dropout, classes):
        super(LSTMTextClassifier, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.LSTM = nn.LSTM(embeddings.size(1), hidden_size=hLayerSzie, num_layers=numLayers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hLayerSzie*2, classes, bias=True)

    def forward(self, input):
        batchTextVectors = self.embeddings(input)
        tensors, _ = self.LSTM(batchTextVectors)
        last_hidden_features = tensors[:, -1, :]
        out = self.fc(last_hidden_features)
        return out


if __name__=="__main__":
    Net = CNNTextClassifier(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), 256, [2, 3, 4], 0.5, 2)
