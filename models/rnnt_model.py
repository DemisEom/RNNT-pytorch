import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
# from models.transducer_np import RNNTLoss
from warprnnt_pytorch import RNNTLoss


class Encoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers_pBLSTM=2, num_layers=1, audio_conf=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_layers_pBLSTM = num_layers_pBLSTM
        self.num_hidden_pBLSTM = 2 ** num_layers_pBLSTM
        self.conv_1 = nn.Conv2d(1, 1, (6, 6), stride=1, padding=0)
        self.conv_2 = nn.Conv2d(1, 1, (6, 6), stride=1, padding=0)
        self.audio_conf = audio_conf

        self.BLSTM = nn.LSTM(151, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=True)
        self.BLSTM2 = nn.LSTM(16, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = x.squeeze()
        x = torch.transpose(x, 1, 2)

        x, _ = self.BLSTM(x)

        num_hiddne_state = self.num_hidden_pBLSTM

        for i in range(self.num_layers_pBLSTM-1):
            num_hiddne_state = int(num_hiddne_state / (i + 1))

            pBlstm = nn.LSTM(input_size=x.size(2), hidden_size=num_hiddne_state, num_layers=1,
                              batch_first=True, bidirectional=True).cuda()
            output, _ = pBlstm(x)

            output = pyramid_stack(output)
            x = output

        output, _ = self.BLSTM2(x)

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
    def __init__(self, vocab_size, embedding_size=256, hidden_size=256, num_layers=2):
        super(PredictionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, vocab_size-1)
        self.embedding_one_hot = nn.Linear(in_features=vocab_size, out_features=embedding_size)
        self.lstm = nn.LSTM(vocab_size-1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)

    def forward(self, x, one_hot=False):
        zero = torch.zeros((x.shape[0], 1), dtype=torch.long).cuda()
        x_mat = torch.cat((zero, x), dim=1)

        # select prediction networks input as one-hot vector or mapping number
        if one_hot is True:
            x_mat = self.embedding_one_hot(x_mat)
        else:
            x_mat = self.embedding(x_mat)

        output, _ = self.lstm(x_mat)
        return output


class JointNetwork(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(JointNetwork, self).__init__()
        self.vocab_size = vocab_size

        self.linear_enc = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_pred = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_feed_forward = nn.Linear(in_features=hidden_size*2, out_features=vocab_size)
        self.tanH = nn.Tanh()
        self.loss = RNNTLoss()

    def forward(self, encoder_output, prediction_output, xs, xlen, ys, ylen):
        encoder_output = torch.unsqueeze(encoder_output, dim=2)
        prediction_output = torch.unsqueeze(prediction_output, dim=1)

        ##
        sz = [max(i, j) for i, j in zip(encoder_output.size()[:-1], prediction_output.size()[:-1])]
        encoder_output = encoder_output.expand(torch.Size(sz+[encoder_output.shape[-1]]));
        prediction_output = prediction_output.expand(torch.Size(sz+[prediction_output.shape[-1]]))
        ##

        encoder_output = self.linear_enc(encoder_output)
        prediction_output = self.linear_pred(prediction_output)

        # output size is '[batch, time_length, word_length, feature_length]'
        dim = len(encoder_output.shape) - 1
        output = torch.cat((encoder_output, prediction_output), dim=dim)

        # output = encoder_output + prediction_output
        output = self.linear_feed_forward(output)
        # output = self.tanH(output)
        # output = F.log_softmax(output, dim=3)

        xlen_temp = [i.shape[0] for i in output]
        xlen = torch.LongTensor(xlen_temp)

        ys = ys.type(torch.int32).cuda()
        xlen = xlen.type(torch.int32).cuda()
        ylen = ylen.type(torch.int32).cuda()

        loss = self.loss(output, ys, xlen, ylen)

        return loss, output

