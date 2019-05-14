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
parser = argparse.ArgumentParser(description='RNN-T training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
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


# setting seed
torch.manual_seed(72160258)
torch.cuda.manual_seed_all(72160258)


if __name__ == '__main__':
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    main_proc = True

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
    if args.visdom and main_proc:
        from visdom import Visdom
        viz = Visdom()
        opts = dict(title=args.id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        viz_window = None
        epochs = torch.arange(1, args.epochs + 1)

    if args.tensorboard and main_proc:
        os.makedirs(args.log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(args.log_dir)
    os.makedirs(save_folder, exist_ok=True)

    avg_loss, start_epoch, start_iter = 0, 0, 0

    args = parser.parse_args()
    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
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

    # logger for visualaization using tensorboard
    logger = Logger('./logs')

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    with codecs.open(args.labels_path, 'r', encoding='utf-8') as label_file:
        labels = str(''.join(json.load(label_file)))

    # setting dataset, data_loader
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

    # load model
    encoder_model = models.Encoder(audio_conf=audio_conf).to(device)
    print(encoder_model)

    prediction_network_model = models.PredictionNet().to(device)
    print(prediction_network_model)

    joint_network_model = models.JointNetwork().to(device)
    print(joint_network_model)

    if args.distributed:
        encoder_model = torch.nn.parallel.DistributedDataParallel(encoder_model,
                                                                  device_ids=(int(args.gpu_rank),) if args.gpu_rank else None)
        prediction_network_model = torch.nn.parallel.DistributedDataParallel(prediction_network_model,
                                                                             device_ids=(int(args.gpu_rank),) if args.gpu_rank else None)
        joint_network_model = torch.nn.parallel.DistributedDataParallel(joint_network_model,
                                                                        device_ids=(int(args.gpu_rank),) if args.gpu_rank else None)


    # Loss and optimizer
    criterion = nn.Softmax()
    optimizer = torch.optim.Adam(joint_network_model.parameters(), lr=0.00001)

    # Start training
    for step in range(args.epochs):
        for i, (data) in enumerate(train_loader):
            if i == len(train_sampler):
                break

            inputs, targets, input_percentages, target_sizes, targets_one_hot = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            inputs = inputs.to(device)

            # Forward pass
            encoder_output = encoder_model(inputs)
            prediction_network_output = prediction_network_model(targets_one_hot)
            outputs = joint_network_model(encoder_output, prediction_network_output)

            loss = criterion(outputs, targets)

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


    # for epoch in range(start_epoch, args.epochs):
    #     model.train()
    #     end = time.time()
    #     start_epoch_time = time.time()
    #     for i, (data) in enumerate(train_loader, start=start_iter):
    #         if i == len(train_sampler):
    #             break
    #         inputs, targets, input_percentages, target_sizes = data
    #         input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    #         # measure data loading time
    #         data_time.update(time.time() - end)
    #
    #         inputs = inputs.to(device)
    #
    #         out, output_sizes = model(inputs, input_sizes)
    #         out = out.transpose(0, 1)  # TxNxH
    #
    #         loss = criterion(out, targets, output_sizes, target_sizes).to(device)
    #         loss = loss / inputs.size(0)  # average the loss by minibatch
    #
    #         inf = float("inf")
    #         if args.distributed:
    #             loss_value = reduce_tensor(loss, args.world_size).item()
    #         else:
    #             loss_value = loss.item()
    #         if loss_value == inf or loss_value == -inf:
    #             print("WARNING: received an inf loss, setting loss value to 0")
    #             loss_value = 0
    #
    #         avg_loss += loss_value
    #         losses.update(loss_value, inputs.size(0))
    #
    #         # compute gradient
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    #         # SGD step
    #         optimizer.step()
    #
    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #         if not args.silent:
    #             print('Epoch: [{0}][{1}/{2}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
    #                 (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
    #         if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
    #             if args.checkpoint_overwrite:
    #                 file_path = '%s/deepspeech_checkpoint.pth' % (save_folder)
    #             else:
    #                 file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
    #             print("Saving checkpoint model to %s" % file_path)
    #             torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
    #                                             loss_results=loss_results,
    #                                             wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
    #                        file_path)
    #         del loss
    #         del out
    #     avg_loss /= len(train_sampler)
    #
    #     epoch_time = time.time() - start_epoch_time
    #     print('Training Summary Epoch: [{0}]\t'
    #           'Time taken (s): {epoch_time:.0f}\t'
    #           'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))
    #
    #     start_iter = 0  # Reset start iteration for next epoch
    #     total_cer, total_wer = 0, 0
    #     model.eval()
    #     with torch.no_grad():
    #         for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
    #             inputs, targets, input_percentages, target_sizes = data
    #             input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    #
    #             # unflatten targets
    #             split_targets = []
    #             offset = 0
    #             for size in target_sizes:
    #                 split_targets.append(targets[offset:offset + size])
    #                 offset += size
    #
    #             inputs = inputs.to(device)
    #
    #             out, output_sizes = model(inputs, input_sizes)
    #
    #             decoded_output, _ = decoder.decode(out, output_sizes)
    #             target_strings = decoder.convert_to_strings(split_targets)
    #             wer, cer = 0, 0
    #             print(len(target_strings))
    #             print(input_sizes)
    #             print(target_strings)
    #
    #             for x in range(len(target_strings)):
    #                 transcript, reference = decoded_output[x][0], target_strings[x][0]
    #                 wer += decoder.wer(transcript, reference) / float(len(reference.split()))
    #                 cer += decoder.cer(transcript, reference) / float(len(reference))
    #             total_cer += cer
    #             total_wer += wer
    #             del out
    #         wer = total_wer / len(test_loader.dataset)
    #         cer = total_cer / len(test_loader.dataset)
    #         wer *= 100
    #         cer *= 100
    #         loss_results[epoch] = avg_loss
    #         wer_results[epoch] = wer
    #         cer_results[epoch] = cer
    #         print('Validation Summary Epoch: [{0}]\t'
    #               'Average WER {wer:.3f}\t'
    #               'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
    #
    #         if args.visdom and main_proc:
    #             x_axis = epochs[0:epoch + 1]
    #             y_axis = torch.stack(
    #                 (loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
    #             if viz_window is None:
    #                 viz_window = viz.line(
    #                     X=x_axis,
    #                     Y=y_axis,
    #                     opts=opts,
    #                 )
    #             else:
    #                 viz.line(
    #                     X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
    #                     Y=y_axis,
    #                     win=viz_window,
    #                     update='replace',
    #                 )
    #         if args.tensorboard and main_proc:
    #             values = {
    #                 'Avg Train Loss': avg_loss,
    #                 'Avg WER': wer,
    #                 'Avg CER': cer
    #             }
    #             tensorboard_writer.add_scalars(args.id, values, epoch + 1)
    #             if args.log_params:
    #                 for tag, value in model.named_parameters():
    #                     tag = tag.replace('.', '/')
    #                     tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
    #                     tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
    #         if args.checkpoint and main_proc:
    #             if args.checkpoint_overwrite:
    #                 file_path = '%s/deepspeech_checkpoint.pth' % (save_folder)
    #             else:
    #                 file_path = '%s/deepspeech_%d.pth' % (save_folder, epoch + 1)
    #             torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
    #                                             wer_results=wer_results, cer_results=cer_results),
    #                        file_path)
    #             # anneal lr
    #             optim_state = optimizer.state_dict()
    #             if args.lr_policy == 'poly':
    #                 optim_state['param_groups'][0]['lr'] = args.lr * (1.0 - ((epoch+1) / args.epochs)) ** args.learning_anneal
    #             elif args.lr_policy == 'fixed':
    #                 pass
    #             else:
    #                 optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
    #             optimizer.load_state_dict(optim_state)
    #             print('Learning rate annealed to: {lr:.6f}, policy: {policy:}'.format(lr=optim_state['param_groups'][0]['lr'], policy=args.lr_policy))
    #
    #         if (best_wer is None or best_wer > wer) and main_proc:
    #             print("Found better validated model, saving to %s" % args.model_path)
    #             torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
    #                                             wer_results=wer_results, cer_results=cer_results), args.model_path)
    #             best_wer = wer
    #
    #             avg_loss = 0
    #         if not args.no_shuffle:
    #             print("Shuffling batches...")
    #             train_sampler.shuffle(epoch)