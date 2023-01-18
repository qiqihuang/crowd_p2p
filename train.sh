#!/usr/bin/env bash

cp models/p2pnet_pafpn.py models/p2pnet.py
python train.py --epochs 500 --warmup 3 --batch_size 2
mv ckpt/best_mae.pth ckpt/vgg_pafpn_p3p4best_mae.pth

cp models/p2pnet_bifpn.py models/p2pnet.py
python train.py --epochs 500 --warmup 3 --batch_size 2

mv ckpt/best_mae.pth ckpt/vgg_bifpn_p3p4best_mae.pth