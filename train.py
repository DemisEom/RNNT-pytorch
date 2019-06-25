from models.models import Transducer
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
import argparse
import json
import os
import time
import codecs
import torch.distributed as dist
import torch.utils.data.distributed
import models.eval_utils as eval_utils
from logger import Logger

# parameter setting
parser = argparse.ArgumentParser(description='RNN-T training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=10, type=int, help='Batch size for training')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout size for training')
parser.add_argument('--decoder-num-layers', default=2, type=float, help='number of layer at RNN-T model')
parser.add_argument('--encoder-num-layers', default=3, type=float, help='number of layer at RNN-T model')
parser.add_argument('--hidden-size', default=250, type=float, help='number of hidden size of rnn layer at RNN-T model')
parser.add_argument('--num-workers', default=6, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels_eng.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--log-dir', default='logs/', help='Location of tensorboard log')
parser.add_argument('--model-path', default=None, help='Location to save best validation model')
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
parser.add_argument('--lm-model', help='path to pretrained decoder pt model file', default=None)
parser.add_argument('--beam-search', help='decoding method select default is greedy', default=None)

# setting seed
torch.manual_seed(72160258)
torch.cuda.manual_seed_all(72160258)

if __name__ == '__main__':

    # ==========================================
    # PREPROCESS
    # ==========================================

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # Tensor board setting
    # ==========================================
    data_name = args.train_manifest.split("/")[2].split("_")[0]
    logging_folder_name = args.log_dir \
                          + data_name + "_" \
                          + str(args.encoder_num_layers) + "encoder_layer_" \
                          + str(args.decoder_num_layers) + "decoder_layer_" \
                          + str(args.hidden_size) + "hidden_"\
                          + str(args.dropout) + "dropout_augment_batchnorm_specAugment_0.2fcdrop"
    print("Tensor board Log file saved : ", logging_folder_name)
    logger = Logger(logging_folder_name)

    # ==========================================
    # DATA SET
    # ==========================================

    # setting dataset, data_loader
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    # load label file(character map)
    with codecs.open(args.labels_path, 'r', encoding='utf-8') as label_file:
        labels = str(''.join(json.load(label_file)))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                       manifest_filepath=args.train_manifest,
                                       labels=labels,
                                       normalize=True,
                                       augment=True,
                                       specaugment=True)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=args.val_manifest,
                                      labels=labels,
                                      normalize=True,
                                      augment=False,
                                      specaugment=False)

    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset,
                                         batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset,
                                                    batch_size=args.batch_size,
                                                    num_replicas=args.world_size,
                                                    rank=args.rank)

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers,
                                   batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # ==========================================
    # NETWORK SETTING
    # ==========================================
    model = Transducer(input_size=161,
                       vocab_size=len(labels),
                       hidden_size=args.hidden_size,
                       decoder_num_layers=args.decoder_num_layers,
                       encoder_num_layers=args.encoder_num_layers,
                       dropout=args.dropout,
                       bidirectional=True,
                       LM_model_path=args.lm_model).to(device)

    if args.model_path:
        test = torch.load(args.model_path)
        model = torch.load(args.model_path)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, momentum=.9)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Model Parameter is ", pytorch_total_params)
    # ==========================================
    # TRAINING
    # ==========================================
    start_time = time.time()
    for step in range(args.epochs):

        total_loss = 0
        train_losses = 0
        start_epoch_time = time.time()

        for i, (data) in enumerate(train_loader):

            if i == len(train_sampler):
                break

            inputs, targets, input_percentages, target_sizes, targets_one_hot, targets_list, labels_map = data

            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            if inputs.shape[0] == 1:
                break

            inputs = inputs.to(device)
            targets_list = targets_list.to(device)

            model.train()
            optimizer.zero_grad()
            train_loss = model(inputs, targets_list, input_sizes, target_sizes)
            train_loss.backward()
            optimizer.step()
            train_losses += float(train_loss)

            if i % 1000 == 0 and i > 0:
                batch_time = time.time() - epoch_time
                temp_losses = total_loss / 10
                print('[Epoch %d Batch %d Time %f] loss %.2f' %(step, i, batch_time, temp_losses))
                total_loss = 0

        # ==========================================
        # EVALUATION
        # ==========================================
        # eval_losses = []
        eval_losses = 0
        total_cer = []
        total_wer = []

        for i, (data) in enumerate(test_loader):

            inputs, targets, input_percentages, target_sizes, targets_one_hot, targets_list, labels_map = data

            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            if inputs.shape[0] == 1:
                break

            inputs = inputs.to(device)
            targets_list = targets_list.to(device)

            eval_loss = model(inputs, targets_list, input_sizes, target_sizes)
            eval_losses += float(eval_loss)

            inverse_map = dict((v, k) for k, v in labels_map.items())

            if args.beam_search:
                y, nll = model.beam_search(inputs, labels_map=labels_map)
            else:
                y = model.greedy_decode_batch(inputs)

            # mapped_pred = [inverse_map[i] for i in y]
            mapped_pred = eval_utils.convert_to_strings(inverse_map, y)

            targets_list = targets_list.tolist()
            # mapped_target = [inverse_map[j] for j in targets_list[0]]
            mapped_target = eval_utils.convert_to_strings(inverse_map, targets_list)

            # eval using metric WER, CER
            for i in range(len(mapped_target)):
                temp_mapped_pred = mapped_pred[i]
                temp_mapped_target = mapped_target[i]
                cer = eval_utils.cer(temp_mapped_pred, temp_mapped_target)
                wer = eval_utils.wer(temp_mapped_pred, temp_mapped_target)
                total_cer.append(cer)
                total_wer.append(wer)

                # print("===========inference & RAW =============")
                # print(temp_mapped_pred)
                # print(temp_mapped_target)

        train_losses = train_losses / len(train_loader)
        eval_losses = eval_losses / len(test_loader)
        total_cer = sum(total_cer) / len(total_cer)
        total_wer = sum(total_wer) / len(total_wer)
        total_loss = 0

        epoch_time = time.time() - start_epoch_time

        print('[Epoch %d / Time %.3f ] loss %.2f, eval loss %.2f, CER %.2f, WER %.2f'
              %(step, epoch_time, train_losses, eval_losses, total_cer, total_wer))

        # save model each 50 epochs
        if step % 50 == 0:
            temp_model_name = logging_folder_name + "/" +\
                              str(step) + "_an4_rnnt_all_model.pt"
            torch.save(model, temp_model_name)

        # ==========================================
        # Tensorboard Logging
        # ==========================================

        info = {'train_loss': train_losses,
                'eval_loss': eval_losses,
                'CER': total_cer,
                'WER': total_wer
                }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)

    end_time = time.time() - start_time
    print('Training is All Done. Take %.3f' % end_time)

