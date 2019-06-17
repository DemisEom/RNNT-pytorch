from models.models import DecoderModel
import argparse
import json
import os
import time
import numpy as np
import codecs
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from data.utils import Dictionary, Corpus


# parameter setting
parser = argparse.ArgumentParser(description='RNN-T decoder(prediction network) Training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/LM/train_LM.txt')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--batch-size', default=10, type=int, help='Batch size for training')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout size for training')
parser.add_argument('--num-workers', default=6, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels_eng.json', help='Contains all characters for transcription')
parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--tensorboard',default=True, help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='logs/', help='Location of tensorboard log')
parser.add_argument('--id', default='RNNT training', help='Identifier for tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/RNNT_model.pth',
                    help='Location to save best validation model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

# setting seed
torch.manual_seed(72160258)
torch.cuda.manual_seed_all(72160258)


def to_np(x):
    return x.data.cpu().numpy()


# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]


if __name__ == '__main__':
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    main_proc = True

    # ==========================================
    # PREPROCESS
    # ==========================================

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    else:
        if args.cuda and args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))

    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(args.epochs)
    best_wer = None

    # visualization setting
    if args.tensorboard:
        print("visualizing by tensorboard")
        os.makedirs(args.log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(args.log_dir)
    os.makedirs(save_folder, exist_ok=True)

    save_folder = args.save_folder

    avg_loss, start_epoch, start_iter = 0, 0, 0

    args = parser.parse_args()
    args.distributed = args.world_size > 1
    main_proc = True

    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    else:
        if args.cuda and args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))

    # ==========================================
    # DATA SET
    # ==========================================

    embed_size = 128
    hidden_size = 1024
    num_layers = 1
    num_epochs = 5
    num_samples = 1000  # number of words to be sampled
    seq_length = 30

    # setting dataset, data_loader
    corpus = Corpus()

    # ids = corpus.get_data('data/LM/train_LM.txt', batch_size)
    ids = corpus.get_data(args.train_manifest, args.batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // seq_length

    # ==========================================
    # NETWORK SETTING
    # ==========================================
    # load model
    model = DecoderModel(embed_size=embed_size,
                         vocab_size=vocab_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    # ==========================================
    # TRAINING
    # ==========================================

    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = ids[:, i:i + seq_length].to(device)
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // seq_length
            if step % 100 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

    # Test the model
    with torch.no_grad():
        with open('sample.txt', 'w') as f:
            # Set intial hidden ane cell states
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                     torch.zeros(num_layers, 1, hidden_size).to(device))

            # Select one word id randomly
            prob = torch.ones(vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

            for i in range(num_samples):
                # Forward propagate RNN
                output, state = model(input, state)

                # Sample a word id
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                # File write
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i + 1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

    # Save the model checkpoints
    print('complete trained model save!')
    torch.save(model.state_dict(), 'models/decoder_LM_model')
