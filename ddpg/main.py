from __future__ import division
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from train import train
from test import test

parser = argparse.ArgumentParser(description='DDPG')
parser.add_argument('--env', default='UnrealArm-ContinuousMeasuredReal-v2', metavar='ENV', help='environment to train on')
parser.add_argument('--max-eps', type=int, default=1000000, metavar='me', help='max episodes')
parser.add_argument('--eval-eps', type=int, default=100, metavar='eval', help='evaluation episodes')
parser.add_argument('--buffer-size', type=int, default=200000, metavar='S', help='size of replay buffer')
parser.add_argument('--start-learning', type=int, default=10000, metavar='S', help='hom many steps to start learning')
parser.add_argument('--render', dest='render', action='store_true', help='if render')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model-dir', default=None, metavar='MD', help='path to saved models')
parser.add_argument('--steps-save', type=int, default=10000, metavar='ss', help='steps interval to save')
parser.add_argument('--batch-size', type=int, default=16, metavar='bs', help='batch size')
parser.add_argument('--lr-c', type=float, default=0.001, metavar='lrc', help='learning rate for critic')
parser.add_argument('--lr-a', type=float, default=0.0001, metavar='lra', help='learning rate for actor')
parser.add_argument('--gamma', type=float, default=0.95, metavar='g', help='param for Q value iteration')
parser.add_argument('--tau', type=float, default=0.001, metavar='t', help='param for soft update target network')
parser.add_argument('--decay-rate', type=float, default=1, metavar='dr', help='decay rate')
parser.add_argument('--eps-start', type=float, default=1, metavar='es', help='the start exploration rate')
parser.add_argument('--eps-end', type=float, default=0.1, metavar='ee', help='the end exploration rate')
parser.add_argument('--eps-decay', type=int, default=100000, metavar='ed',
                    help='how many episodes to decay the exploration rate')
parser.add_argument('--gpu-ids', type=int, default=-1, help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--model', default='none', metavar='h',
                    help='whether to combine heads of actor/critic net')
parser.add_argument('--pri', dest='pri', action='store_true', help='prioritized memory')
parser.add_argument('--obs', default='vector', metavar='UE', help='unreal env')
parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--normalize', dest='normalize', action='store_true', help='normalize image')
parser.add_argument('--stack-frames', type=int, default=4, metavar='SF', help='Choose number of observations to stack')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--sleep-time', type=int, default=10, metavar='LO', help='seconds')
parser.add_argument('--test', dest='test', action='store_true', help='only test not train')

if __name__ == '__main__':
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.gpu_ids == -1:
        device = torch.device('cpu')
        print('using cpu')
    else:
        gpu_idx = args.gpu_ids
        device = torch.device(gpu_idx)
        print('using gpu %d' % gpu_idx)

    if args.test:
        test(device, args)
    else:
        train(0, device, args)
