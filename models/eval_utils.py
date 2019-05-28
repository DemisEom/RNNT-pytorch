import torch
import editdistance


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


def eval_dev(model, ldr,
             encoder_model,
             prediction_network_model,
             joint_network_model,
             test_loader):
    losses = []; all_preds = []; all_labels = []

    model.set_eval()

    for i, (batch) in enumerate(ldr):
        inputs, targets, input_percentages, target_sizes, targets_one_hot, targets_list, labels_map = data
        preds = model.infer(batch)
        encoder_output = encoder_model(batch)
        prediction_network_output = prediction_network_model(targets_list, one_hot=False)

        encoder_output = encoder_output.to(device)
        prediction_network_output = prediction_network_output.to(device)

        loss = joint_network_model(encoder_output, prediction_network_output,
                                   inputs, input_sizes, targets_list, target_sizes)

        loss = model.loss(batch)
        losses.append(loss.data[0])
        all_preds.extend(preds)
        all_labels.extend(batch[1])

    model.set_train()

    loss = sum(losses) / len(losses)
    results = [(preproc.decode(l), preproc.decode(p))
               for l, p in zip(all_labels, all_preds)]
    cer = speech.compute_cer(results)
    print("Dev: Loss {:.3f}, CER {:.3f}".format(loss, cer))
    return loss, cer


def infer(batch, beam_size=4):
    out = batch
    out = out.cpu().data.numpy()
    preds = []
    for e, (i, l) in enumerate(zip(*batch)):
        T = i.shape[0]
        U = len(l) + 1
        lp = out[e, :T, :U, :]
        preds.append(decode_static(lp, beam_size, blank=self.blank)[0])
    return preds


def decode_static(log_probs, beam_size=1, blank=0):
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