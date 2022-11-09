"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

import argparse
import os

import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt

from models import HyperRes, NoiseNet

from utils import AverageMeter
from utils.DataUtils import CommonTools
from utils.DataUtils.CommonTools import calculate_psnr, postProcessForStats, saveImage, calculate_ssim
from utils.DataUtils.TrainLoader import NoisyDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--weights', required=True, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--meta_blocks', type=int, default=16,
                    help='Number of Meta Blocks')
parser.add_argument('--reg_blocks', type=int, default=0,
                    help='Number of Regular Blocks')
parser.add_argument('--steps', type=int, default=0, help='Number of steps')
parser.add_argument('--valid', type=str, default='valid',
                    help='Validation folder')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run on,[cpu,cuda..]')
parser.add_argument('--sigmas', type=int, nargs='+',
                    default=[15], help='input resolutions.')
parser.add_argument('--no_bn', dest='bn', default=True, action='store_false',
                    help='Add Batch Normalization int the meta blocks')
parser.add_argument('--no_bias', dest='bias', default=True, action='store_false',
                    help='Add Batch Normalization int the meta blocks')
parser.add_argument('-y', '-Y', '--gray', dest='y_channel', default=False, action='store_true',
                    help='Train on Grayscale only')
parser.add_argument('--data_type', type=str, default='n', choices=['n', 'sr', 'j'],
                    help='Defines the task data, de(n)oise, super-resolution(sr), de(j)peg.')
parser.add_argument('--norm_f', type=float, default=255,
                    help='The normalization factor for the distortion levels.')
parser.add_argument('--ins', type=float, default=-1,
                    help='The normalization factor for the distortion levels.')

best_prec1 = 0


def loadNoiseNet(device):
    noise_model = NoiseNet().to(device)
    noise_model.load_state_dict(torch.load('pre_trained/noise_net_latest.pth', map_location=device))
    return noise_model


def main():
    global args, best_prec1
    args = parser.parse_args()
    lvls_in = list(args.sigmas)
    if args.ins > 0:
        lvls_in = [args.ins for _ in args.sigmas]
        print(lvls_in)
    # Create model
    model = HyperRes(meta_blocks=args.meta_blocks, sigmas=lvls_in, device=args.device, bn=args.bn, bias=args.bias,
                     gray=args.y_channel, norm_factor=args.norm_f).to(args.device)
    noise_model = loadNoiseNet(args.device)

    # Load weights
    if not os.path.isfile(args.weights):
        print("=> no checkpoint found at '{}'".format(args.weights))
        exit()
    print("=> loading weights '{}'".format(args.weights))
    checkpoint = torch.load(args.weights, map_location=args.device)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    orig_sigmas = checkpoint['sigmas']

    model.load_state_dict(checkpoint['state_dict'])

    CommonTools.set_random_seed(42)
    torch.backends.cudnn.benckmark = True

    # Data loading code
    val_loader = NoisyDataset(
        args.data,
        cor_lvls=args.sigmas,
        phase='test',
        interp=False,
        lr_prefix=args.data_type,
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    psnr_avg = validate(val_loader, model, noise_model)


def validate(val_loader, model: HyperRes, noise_model: NoiseNet):
    print("Validation:")

    sig_loss = [AverageMeter() for _ in range(len(args.sigmas))]
    ssim_loss = [AverageMeter() for _ in range(len(args.sigmas))]

    model.eval()
    noise_model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            target = data['target'].to(args.device)
            images = [x.to(args.device) for x in data['image']]

            print("=====================")
            print(i)
            print(args.sigmas)
            model.setLevel(args.sigmas)
            output = model(images)
            for j in range(len(args.sigmas)):
                # Measure PSNR
                for out_idx, out in enumerate(output[j]):
                    imgs = postProcessForStats([target[out_idx], out, images[j]])
                    trg, out, noise = imgs

                    plt.imsave("blindNoise/{}_gt.png".format(i), trg)
                    plt.imsave("blindNoise/{}_{}_noise.png".format(i, args.sigmas[j]), noise)
                    plt.imsave("blindNoise/{}_{}_semi.png".format(i, args.sigmas[j]), out)

                    psnr = calculate_psnr(out, trg, args.data_type == 'sr')
                    ssim = calculate_ssim(out, trg, args.data_type == 'sr')
                    sig_loss[j].update(psnr)
                    ssim_loss[j].update(ssim)
                    print("\t{:.3f}:\tPSNR:\t{:.3f} SSIM\t{:.3f}".format(args.sigmas[j], psnr, ssim))

            # Blind
            new_sigmas = [noise_model(image) for image in images]
            model.setLevel(new_sigmas)
            output = model(images)

            new_sigmas = [x.detach().cpu().numpy()[0][0] for x in new_sigmas]
            print(new_sigmas)
            for j in range(len(new_sigmas)):
                # Measure PSNR
                for out_idx, out in enumerate(output[j]):
                    imgs = postProcessForStats([target[out_idx], out])
                    trg, out = imgs

                    plt.imsave("blindNoise/{:}_{:.3f}.png".format(i, new_sigmas[j]), out)

                    psnr = calculate_psnr(out, trg, args.data_type == 'sr')
                    ssim = calculate_ssim(out, trg, args.data_type == 'sr')
                    sig_loss[j].update(psnr)
                    ssim_loss[j].update(ssim)
                    print("\t{:.3f}:\tPSNR:\t{:.3f} SSIM\t{:.3f}".format(new_sigmas[j], psnr, ssim))

    model.train()

    return {s: (t.avg, ssim.avg) for s, t, ssim in zip(args.sigmas, sig_loss, ssim_loss)}


if __name__ == '__main__':
    main()
