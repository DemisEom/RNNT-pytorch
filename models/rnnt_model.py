import math
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
    def __init__(self, hidden_size=4, num_layers_pBLSTM=4, num_layers=4, num_classes=10, audio_conf=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_layers_pBLSTM = num_layers_pBLSTM
        self.num_hidden_pBLSTM = 2 ** num_layers_pBLSTM
        self.conv_1 = nn.Conv2d(1, 1, (6, 6), stride=1)
        self.conv_2 = nn.Conv2d(1, 1, (6, 6), stride=1)
        self.audio_conf = audio_conf

        self.BLSTM = nn.LSTM(91, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=True)
        self.pBLSTM = nn.LSTM(8, hidden_size=self.hidden_size, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.BLSTM2 = nn.LSTM(8, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = x.view(x.size(0), x.size(2), x.size(3))

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        x, h_c_0 = self.BLSTM(x, (h0, c0))

        num_hiddne_state = self.num_hidden_pBLSTM

        for i in range(self.num_layers_pBLSTM-1):
            num_hiddne_state = int(num_hiddne_state / (i + 1))

            hn = torch.zeros(1 * 2, x.size(0), num_hiddne_state)
            cn = torch.zeros(1 * 2, x.size(0), num_hiddne_state)

            pBlstm = nn.LSTM(input_size=x.size(2), hidden_size=num_hiddne_state, num_layers=1,
                              batch_first=True, bidirectional=True)
            output, _ = pBlstm(x, (hn, cn))

            output = pyramid_stack(output)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        output, _ = self.BLSTM2(output, (h0, c0))

        return output


def pyramid_stack(inputs):

    size = inputs.size()

    if int(size[1]) % 2 == 1:
        padded_inputs = F.pad(inputs, (0, 0, 0, 1, 0, 0))
        sequence_length = int(size[1]) + 1
    else:
        padded_inputs = inputs
        sequence_length = int(size[1])

    odd_ind = torch.arange(1, sequence_length, 2, dtype=torch.long)
    even_ind = torch.arange(0, sequence_length, 2, dtype=torch.long)

    odd_inputs = padded_inputs[:, odd_ind, :]
    even_inputs = padded_inputs[:, even_ind, :]

    outputs_stacked = torch.cat([even_inputs, odd_inputs], 2)

    return outputs_stacked


class PredictionNet(nn.Module):
    def __init__(self, input_size=29, embedding_size=20, hidden_size=4, num_layers=4):
        super(PredictionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size=256, num_layers=2, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return output


class JointNetwork(nn.Module):
    def __init__(self, num_classes=29, batch_size=20, hidden_nodes=2):
        super(JointNetwork, self).__init__()
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes

        # reshape to "N x T x U x H"
        self.N = batch_size
        self.T = 151
        self.U = 29
        self.H = self.hidden_nodes
        self.output_feature = self.N * self.T * self.U * self.H

        self.linear_enc = nn.Linear(in_features=12169, out_features=self.output_feature)
        self.linear_pred = nn.Linear(in_features=653312, out_features=self.output_feature)

    def forward(self, encoder_output, prediction_output):

        encoder_output = self.linear_enc(encoder_output)
        prediction_output = self.linear_pred(prediction_output)

        encoder_output = encoder_output.view(self.N, self.T, self.U, self.H)
        prediction_output = prediction_output.view(self.N, self.T, self.U, self.H)

        output = F.tanh(encoder_output + prediction_output)

        return output

