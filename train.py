import models.rnnt_model as models
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler

import argparse
import json
import os
import time
import codecs
import torch.distributed as dist
import torch.utils.data.distributed
import models.eval_utils as eval_utils

# parameter setting
parser = argparse.ArgumentParser(description='RNN-T training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=5, type=int, help='Batch size for training')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--activation-fn', default='relu', help='Specifies the activation function')
parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-policy', default='poly', help='Set the policy for learning-rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=0.5, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--checkpoint-overwrite', dest='checkpoint_overwrite', action='store_true', help='Overwrite checkpoints')
parser.add_argument('--tensorboard',default=True, help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='logs/', help='Location of tensorboard log')
parser.add_argument('--id', default='RNNT training', help='Identifier for tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/RNNT_model.pth',
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
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true', help='using SpecAugment')
parser.add_argument('--prediction_one_hot', default=False,
                    help='Directory to inject noise into audio. If default, noise Inject not added')

# setting seed
torch.manual_seed(72160258)
torch.cuda.manual_seed_all(72160258)


def to_np(x):
    return x.data.cpu().numpy()

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
    # DATA
    # ==========================================

    # setting dataset, data_loader
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    with codecs.open(args.labels_path, 'r', encoding='utf-8') as label_file:
        labels = str(''.join(json.load(label_file)))


    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment, specaugment=False)
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

    # ==========================================
    # NETWORK SETTING
    # ==========================================
    # load model
    encoder_model = models.Encoder(audio_conf=audio_conf).to(device)
    print(encoder_model)

    prediction_network_model = models.PredictionNet(vocab_size=len(labels)).to(device)
    print(prediction_network_model)

    joint_network_model = models.JointNetwork(vocab_size=len(labels)).to(device)
    print(joint_network_model)

    if args.distributed:
        encoder_model = torch.nn.parallel.DistributedDataParallel(encoder_model,
                                                                  device_ids=(int(args.gpu_rank),) if args.gpu_rank else None)
        prediction_network_model = torch.nn.parallel.DistributedDataParallel(prediction_network_model,
                                                                             device_ids=(int(args.gpu_rank),) if args.gpu_rank else None)
        joint_network_model = torch.nn.parallel.DistributedDataParallel(joint_network_model,
                                                                        device_ids=(int(args.gpu_rank),) if args.gpu_rank else None)

    # Loss and optimizer
    # optimizer_1 = torch.optim.Adam(encoder_model.parameters(), lr=args.lr)
    # optimizer_2 = torch.optim.Adam(prediction_network_model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(joint_network_model.parameters(), lr=args.lr)

    # ==========================================
    # TRAINING
    # ==========================================
    for step in range(args.epochs):

        total_loss = 0
        totloss = 0; losses = []
        start_time = time.time()

        for i, (data) in enumerate(train_loader):

            if i == len(train_sampler):
                break

            inputs, targets, input_percentages, target_sizes, targets_one_hot, targets_list, labels_map = data

            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            inputs = inputs.to(device)
            targets_list = targets_list.to(device)

            # Forward pass
            encoder_output = encoder_model(inputs)
            prediction_network_output = prediction_network_model(targets_list, one_hot=False)

            encoder_output = encoder_output.to(device)
            prediction_network_output = prediction_network_output.to(device)

            loss, _ = joint_network_model(encoder_output, prediction_network_output,
                                       inputs, input_sizes, targets_list, target_sizes)

            # Loss setting
            # Backward and optimize
            # optimizer_1.zero_grad()
            # optimizer_2.zero_grad()
            optimizer.zero_grad()

            loss.backward()

            loss = float(loss.data) * len(input_sizes)
            total_loss += loss
            losses.append(loss)

            # grad_norm = nn.utils.clip_grad_norm(joint_network_model.parameters(), 200)

            # optimizer_1.step()
            # optimizer_2.step()
            optimizer.step()

            # if i % 10 == 0 and i > 0:
            #     temp_losses = total_loss / 20
            #     print('[Epoch %d Batch %d] loss %.2f' %(step, i, temp_losses))
            #     total_loss = 0

        losses = sum(losses) / len(train_loader)
        total_loss = 0

        ## TO DO eval using metric WER, CER
        loss_eval = eval_utils.eval_dev(encoder_model, prediction_network_model, joint_network_model, test_loader)
        print('[Epoch %d ] loss %.2f, eval loss %2.f' % (step, losses, loss_eval))
        # if args.tensorboard:
        #     values = {
        #         'Avg Train Loss': temp_losses
        #     }
        #     tensorboard_writer.add_scalars(args.id, values, step + 1)
        #     for tag, value in joint_network_model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         tensorboard_writer.add_histogram(tag, to_np(value), step + 1)
        #         tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), step + 1)



