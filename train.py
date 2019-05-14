import models.rnnt_model as models
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler

import argparse
import json
import os
import time
import codecs
import torch.distributed as dist
import torch.utils.data.distributed

# parameter setting
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels_korean.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=768, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=7, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--activation-fn', default='relu', help='Specifies the activation function')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-policy', default='poly', help='Set the policy for learning-rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=0.5, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--checkpoint-overwrite', dest='checkpoint_overwrite', action='store_true', help='Overwrite checkpoints')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true', help='using SpecAugment')

torch.manual_seed(72160258)
torch.cuda.manual_seed_all(72160258)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# logger for visualaization using tensorboard
logger = Logger('./logs')

# Hyper-parameters
args = parser.parse_args()

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

# setting dataset, data_loader
train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                   normalize=True, augment=args.augment, specaugment=args.spec_augment)
test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                  normalize=True, augment=False, specaugment=False)
if not args.distributed:
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
else:
    train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                num_replicas=args.world_size, rank=args.rank)
train_loader = AudioDataLoader(train_dataset,
                               num_workers=args.num_workers, batch_sampler=train_sampler)
test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers)

# load model
encoder_model = models.Encoder().to(device)
prediction_network_model = models.PredictionNet().to(device)
joint_network_model = models.JointNetwork().to(device)


# Loss and optimizer
criterion = nn.Softmax()
optimizer = torch.optim.Adam(joint_network_model.parameters(), lr=0.00001)

# setting training parameters


# Start training
for step in range(total_step):

    # Reset the data_iter
    if (step + 1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch images and labels
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

    # Forward pass
    encoder_output = encoder_model(train_loader)
    prediction_network_output = prediction_network_model(test_loader)
    outputs = joint_network_model(encoder_output, prediction_network_output)

    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step + 1) % 100 == 0:
        print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
              .format(step + 1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.item(), 'accuracy': accuracy.item()}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)