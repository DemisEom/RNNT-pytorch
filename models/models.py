import math
import torch
from torch import nn, autograd
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss


class DecoderModel(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers=1, dropout=0.5, blank=0, LM=False):
        super(DecoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        self.embed_size = embed_size

        if LM:
            test_size = 26
        else:
            test_size = vocab_size - 1

        self.embed = nn.Embedding(vocab_size, test_size, padding_idx=blank)
        # self.embed.weight.data[1:] = torch.eye(test_size)
        self.embed.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=test_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, y_mat, hid=None):

        y_mat = self.embed(y_mat)
        y_mat, h = self.lstm(y_mat, hid)

        out = y_mat.reshape(y_mat.size(0) * y_mat.size(1), y_mat.size(2))

        out = self.linear(out)

        return out, y_mat, h


class EncoderModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=.2, blank=0, bidirectional=False):
        super(EncoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            hidden_size *= 2

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.conv_1 = nn.Conv2d(1, 1, (1, 1), stride=1)
        self.conv_2 = nn.Conv2d(1, 1, (1, 1), stride=1)

    def forward(self, xs, hid=None):
        xs = torch.transpose(xs, 2, 3)

        xs = self.conv_1(xs)
        xs = self.conv_2(xs)

        xs = xs.squeeze()

        h, hid = self.lstm(xs, hid)

        # TO DO, ADD pyramidal BI-LSTM
        # num_hiddne_state = self.num_hidden_pBLSTM
        #
        # for i in range(self.num_layers_pBLSTM-1):
        #     num_hiddne_state = int(num_hiddne_state / (i + 1))
        #
        #     pBlstm = nn.LSTM(input_size=x.size(2), hidden_size=num_hiddne_state, num_layers=1,
        #                       batch_first=True, bidirectional=True).cuda()
        #     output, _ = pBlstm(x)
        #
        #     output = pyramid_stack(output)
        #     x = output
        #
        # output, _ = self.BLSTM2(x)

        return self.linear(h), hid

    def greedy_decode(self, xs):
        xs = self(xs)[0][0] # only one sequence
        xs = F.log_softmax(xs, dim=1)
        logp, pred = torch.max(xs, dim=1)
        return pred.data.cpu().numpy(), -float(logp.sum())


class Transducer(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=0.5, blank=0, bidirectional=False, decoder_load=False):
        super(Transducer, self).__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.loss = RNNTLoss()

        self.decoder = DecoderModel(embed_size=vocab_size,
                                    vocab_size=vocab_size,
                                    hidden_size=hidden_size,
                                    dropout=dropout)
        if decoder_load:
            self.decoder.load_state_dict(torch.load('./models/decoder_LM_model'), strict=False)

        self.encoder = EncoderModel(input_size=input_size,
                                    vocab_size=hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    bidirectional=bidirectional)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def joint(self, f, g):
        dim = len(f.shape) - 1

        out = torch.cat((f, g), dim=dim)
        out = F.tanh(self.fc1(out))

        return self.fc2(out)

    def forward(self, xs, ys, xlen, ylen):
        # encoder
        xs, _ = self.encoder(xs)

        # concat first zero
        zero = autograd.Variable(torch.zeros((ys.shape[0], 1)).long())

        if ys.is_cuda:
            zero = zero.cuda()

        # decoder(prediction network)
        y_mat = torch.cat((zero, ys), dim=1)
        _, y_mat, _ = self.decoder(y_mat)

        xs = xs.unsqueeze(dim=2)
        y_mat = y_mat.unsqueeze(dim=1)

        # expand
        sz = [max(i, j) for i, j in zip(xs.size()[:-1], y_mat.size()[:-1])]
        xs = xs.expand(torch.Size(sz + [xs.shape[-1]]));
        y_mat = y_mat.expand(torch.Size(sz + [y_mat.shape[-1]]))

        out = self.joint(xs, y_mat)

        if ys.is_cuda:
            xlen = xlen.cuda()
            ylen = ylen.cuda()
        else:
            out = F.log_softmax(out, dim=3)
            # NOTE loss function need flatten label
            ys = torch.cat([ys[i, :j] for i, j in enumerate(ylen.data)], dim=0).cpu()

        xlen_temp = [i.shape[0] for i in out]
        xlen = torch.LongTensor(xlen_temp)
        xlen = xlen.type(torch.int32).cuda()

        loss = self.loss(out, ys.int(), xlen, ylen)
        return loss

    def greedy_decode(self, x):
        x = self.encoder(x)[0][0]
        vy = autograd.Variable(torch.LongTensor([0]), volatile=True).view(1, 1)  # vector preserve for embedding
        if x.is_cuda:
            vy = vy.cuda()
        _, y, h = self.decoder(y_mat=vy)
        y_seq = [];
        logp = 0

        for i in x:
            ytu = self.joint(i, y[0][0])
            out = F.log_softmax(ytu, dim=0)
            p, pred = torch.max(out, dim=0)
            pred = int(pred);
            logp += float(p)
            if pred != self.blank:
                y_seq.append(pred)
                vy.data[0][0] = pred
                _, y, h = self.decoder(y_mat=vy, hid=h)
        return y_seq, -logp

    def beam_search(self, xs, labels_map, W=10, prefix=False):
        '''''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''''
        use_gpu = xs.is_cuda

        def forward_step(label, hidden):
            ''' `label`: int '''
            label = autograd.Variable(torch.LongTensor([label]), volatile=True).view(1, 1)
            if use_gpu: label = label.cuda()
            label = self.embed(label)
            pred, hidden = self.decoder(label, hidden)
            return pred[0][0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        xs = self.encoder(xs)[0][0]
        B = [Sequence(labels_map=labels_map, blank=self.blank)]
        for i, x in enumerate(xs):
            sorted(B, key=lambda a: len(a.k), reverse=True)  # larger sequence first add
            A = B
            B = []
            if prefix:
                # for y in A:
                #     y.logp = log_aplusb(y.logp, prefixsum(y, A, x))
                for j in range(len(A) - 1):
                    for i in range(j + 1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        # A[i] -> A[j]
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        idx = len(A[i].k)
                        ytu = self.joint(x, pred)
                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k) - 1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k + 1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                # y* = most probable in A
                A.remove(y_hat)
                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                ytu = self.joint(x, pred)
                logp = F.log_softmax(ytu, dim=0)  # log probability for each k
                # TODO only use topk vocab
                for k in range(self.vocab_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    if k == self.blank:
                        B.append(yk)  # next move
                        continue
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden;
                    yk.k.append(k);
                    if prefix: yk.g.append(pred)
                    A.append(yk)
                # sort A
                # sorted(A, key=lambda a: a.logp, reverse=True) # just need to calculate maximum seq

                # sort B
                # sorted(B, key=lambda a: a.logp, reverse=True)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp: break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]

        # return highest probability sequence
        print(B[0])
        return B[0].k, -B[0].logp


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


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))


class Sequence():
    def __init__(self, labels_map, seq=None, blank=0):
        if seq is None:
            self.g = []  # predictions of phoneme language model
            self.k = [blank]  # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0  # probability of this sequence, in log scale
        else:
            self.g = seq.g[:]  # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp
        self.labels_map = labels_map

    def __str__(self):
        return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(' '.join([self.labels_map[i] for i in self.k]), -self.logp)


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

