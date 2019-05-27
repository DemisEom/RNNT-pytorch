import torch
from warprnnt_pytorch import RNNTLoss
rnnt_loss = RNNTLoss()
cuda = False # whether use GPU version
acts = torch.FloatTensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.6, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.8, 0.1]],
                            [[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.1, 0.1],
                            [0.7, 0.1, 0.2, 0.1, 0.1]]]])
labels = torch.IntTensor([[1, 2]])
act_length = torch.IntTensor([2])
label_length = torch.IntTensor([2])
if cuda:
    acts = acts.cuda()
    labels = labels.cuda()
    act_length = act_length.cuda()
    label_length = label_length.cuda()
acts = torch.autograd.Variable(acts, requires_grad=True)
labels = torch.autograd.Variable(labels)
act_length = torch.autograd.Variable(act_length)
label_length = torch.autograd.Variable(label_length)
loss = rnnt_loss(acts, labels, act_length, label_length)
loss.backward()
print(loss)