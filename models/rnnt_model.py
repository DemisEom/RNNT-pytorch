import torch
import torch.nn as nn
import torch.nn.functional as F


# simple cnn network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Bidirectional recurrent neural network
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        output, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(output[:, -1, :])
        return output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv = nn.Conv2d(1, 6, 6)
        self.BLSTM = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.pBLSTM = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        x = self.BLSTM(x, num_layers=1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        for i in range(self.num_layers_pBLSTM):
            output, (hn, cn) = self.pBLSTM(x, (h0, c0))
            x = torch.cat((hn, cn))
            h0 = hn
            c0 = cn

        output = self.BLSTM(x, num_layers=2)
        return output


class PredictionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PredictionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(100, 200)
        self.lstm = nn.LSTM(input_size, hidden_size = 512, num_layers= 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Set initial states
        x = self.embedding(x)
        output = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        return output


class JointNetwork(nn.Module):
    def __init__(self, Encoder, PredictionNet, num_classes, batch_size, hidden_nodes):
        self.tanH = nn.Tanh()
        self.linear = nn.Linear()
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.num_classed = num_classes

    def forward(self, encoder_output, prediction_output):

        encoder_output = encoder_output.view(self.battch_size, self.num_classed, self.hidden_nodes)
        prediction_output = prediction_output.view(self.battch_size, self.num_classed, self.hidden_nodes)

        encoder_output = self.linear(encoder_output)
        prediction_output = self.linear(prediction_output)

        output = self.tanH(encoder_output + prediction_output)

        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

