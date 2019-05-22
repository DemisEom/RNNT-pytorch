import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from models.transducer_np import RNNTLoss

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
    def __init__(self, hidden_size=256, num_layers_pBLSTM=4, num_layers=4, audio_conf=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_layers_pBLSTM = num_layers_pBLSTM
        self.num_hidden_pBLSTM = 2 ** num_layers_pBLSTM
        self.conv_1 = nn.Conv2d(1, 1, (6, 6), stride=1)
        self.conv_2 = nn.Conv2d(1, 1, (6, 6), stride=1)
        self.audio_conf = audio_conf

        self.BLSTM = nn.LSTM(190, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=True)
        self.BLSTM2 = nn.LSTM(8, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = x.squeeze()

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
    def __init__(self, labels_len, embedding_size=20, hidden_size=512, num_layers=2):
        super(PredictionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(labels_len, labels_len-1, padding_idx=1)
        self.embedding_one_hot = nn.Linear(in_features=labels_len, out_features=embedding_size)
        self.lstm = nn.LSTM(labels_len-1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x, one_hot=False):
        zero = torch.zeros((x.shape[0], 1), dtype=torch.long)
        x_mat = torch.cat((zero, x), dim=1)

        if True == one_hot:
            x_mat = self.embedding_one_hot(x_mat)
        else:
            x_mat = self.embedding(x_mat)

        output, _ = self.lstm(x_mat)
        return output


class JointNetwork(nn.Module):
    def __init__(self, num_classes, batch_size=20, hidden_nodes=2):
        super(JointNetwork, self).__init__()
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.num_classes = num_classes

        # reshape to "N x T x U x H"
        self.N = batch_size
        self.T = 76
        self.U = 20
        self.H = self.hidden_nodes
        self.output_feature = self.T * self.U * self.H

        self.linear_enc = nn.Linear(in_features=38912, out_features=self.output_feature)
        self.linear_pred = nn.Linear(in_features=15872, out_features=self.output_feature)
        self.linear_feed_forward = nn.Linear(in_features=self.output_feature, out_features=num_classes*11)
        self.tanH = nn.Tanh()

        self.loss = RNNTLoss()

        self.fc1 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, num_classes)

    def forward(self, encoder_output, prediction_output, xs, xlen, ys, ylen):
        # print(encoder_output.size())
        # print(prediction_output.size())

        encoder_output = encoder_output.reshape(self.batch_size, -1)
        prediction_output = prediction_output.reshape(self.batch_size, -1)

        encoder_output = self.linear_enc(encoder_output)
        prediction_output = self.linear_pred(prediction_output)

        encoder_output = encoder_output.view(self.N, self.T, self.U, self.H)
        prediction_output = prediction_output.view(self.N, self.T, self.U, self.H)

        output = encoder_output + prediction_output
        output = self.fc2(output)
        output = self.tanH(output)

        # output = output.reshape(self.batch_size, -1)

        # output = self.linear_feed_forward(output)
        # output = output.view(self.batch_size, self.num_classes, self.U)
        output = F.log_softmax(output, dim=3)

        xlen_temp = [i.shape[0] for i in output]
        xlen = torch.LongTensor(xlen_temp)

        # ys_temp = ys.view(1, -1)
        loss = self.loss(output, ys, xlen, ylen)
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """

        return loss

