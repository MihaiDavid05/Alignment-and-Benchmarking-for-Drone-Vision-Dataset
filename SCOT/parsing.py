#!usr/bin/bash python

import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description='SCOT in pytorch')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument('--datapath', type=str, default='./Datasets_SCOT')
parser.add_argument('--dataset', type=str, default='pfpascal')
parser.add_argument('--backbone', type=str, default='resnet101')
parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--hyperpixel', type=str, default='')
parser.add_argument('--logpath', type=str, default='')
parser.add_argument('--split', type=str, default='test', help='trn,val.test')

# Algorithm parameters
parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
parser.add_argument('--classmap', type=int, default=1, help='class activation map: 0 for none, 1 for using CAM')
parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
