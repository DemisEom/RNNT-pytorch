import torch
# import editdistance
import math


def log_sum_exp(a, b):
    """
    Stable log sum exp.
    """
    return max(a, b) + math.log1p(math.exp(-abs(a-b)))


def encode(self, text):
    text = list(text)
    if self.start_and_end:
        text = [self.START] + text + [self.END]
    return [self.char_to_int[t] for t in text]


def decode(self, seq):
    text = [self.int_to_char[s] for s in seq]
    if not self.start_and_end:
        return text

    s = text[0] == self.START
    e = len(text)
    if text[-1] == self.END:
        e = text.index(self.END)
    return text[s:e]


def compute_cer(results):

    dist = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total


def eval_loop(model, ldr):
    all_preds = []; all_labels = []
    for batch in tqdm.tqdm(ldr):
        preds = model.infer(batch)
        all_preds.extend(preds)
        all_labels.extend(batch[1])
    return list(zip(all_labels, all_preds))


def eval_dev(encoder_model,
             prediction_network_model,
             joint_network_model,
             test_loader):

    losses = []
    all_preds = []
    all_labels = []

    for i, (data) in enumerate(test_loader):
        inputs, targets, input_percentages, target_sizes, _, targets_list, labels_map = data


        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        inputs = inputs.cuda()
        targets_list = targets_list.cuda()

        encoder_output = encoder_model(inputs)
        prediction_network_output = prediction_network_model(targets_list, one_hot=False)

        encoder_output = encoder_output.cuda()
        prediction_network_output = prediction_network_output.cuda()

        loss, output = joint_network_model(encoder_output, prediction_network_output,
                                   inputs, input_sizes, targets_list, target_sizes)

        preds = inference(inputs=inputs, targets_list=targets_list,
                          target_sizes=target_sizes, output=output)
        losses.append(loss.data[0])
        all_preds.extend(preds)

        for i in range(len(target_sizes)):
            temp_size = target_sizes[i].tolist()
            temp_list = targets_list[i].tolist()
            all_labels.append(temp_list[:temp_size])

    loss = sum(losses) / len(losses)
    # results = [(preproc.decode(l), preproc.decode(p))
    #            for l, p in zip(all_labels, all_preds)]
    # cer = speech.compute_cer(results)
    # print("Dev: Loss {:.3f}, CER {:.3f}".format(loss, cer))
    return loss


def inference(inputs, targets_list, target_sizes, output, beam_size=4):
    preds = []

    xlen_temp = [i.shape[0] for i in output]
    xlen = torch.LongTensor(xlen_temp)

    for i in range(inputs.shape[0]):
        T = xlen[i]
        U = target_sizes[i]+1
        lp = output[i, :T, :U, :]
        preds.append(decode_static(lp, beam_size)[0])
    return preds


def decode_static(log_probs, beam_size=1, blank=28):
    T, U, V = log_probs.shape
    beam = [((), 0)];
    for i in range(T + U - 2):
        new_beam = {}
        for hyp, score in beam:
            u = len(hyp)
            t = i - u
            for v in range(V):
                if v == blank:
                    if t < T - 1:
                        new_hyp = hyp
                        new_score = score + log_probs[t, u, v]
                elif u < U - 1:
                    new_hyp = hyp + (v,)
                    new_score = score + log_probs[t, u, v]
                else:
                    continue

                old_score = new_beam.get(new_hyp, None)
                if old_score is not None:
                    new_beam[new_hyp] = log_sum_exp(old_score, new_score)
                else:
                    new_beam[new_hyp] = new_score

        new_beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    hyp, score = beam[0]
    return hyp, score + log_probs[-1, -1, blank]


##################################
# Hawk Aron
##################################


def greedy_decode(self, x):
    x = self.encoder(x)[0][0]
    vy = autograd.Variable(torch.LongTensor([0]), volatile=True).view(1, 1)  # vector preserve for embedding
    if x.is_cuda: vy = vy.cuda()
    y, h = self.decoder(self.embed(vy))  # decode first zero
    y_seq = [];
    logp = 0
    for i in x:
        ytu = self.joint(i, y[0][0])
        out = F.log_softmax(ytu, dim=0)
        p, pred = torch.max(out, dim=0)  # suppose blank = -1
        pred = int(pred);
        logp += float(p)
        if pred != self.blank:
            y_seq.append(pred)
            vy.data[0][0] = pred  # change pm state
            y, h = self.decoder(self.embed(vy), h)
    return y_seq, -logp


def beam_search(self, xs, W=10, prefix=False):
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
    B = [Sequence(blank=self.blank)]
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
