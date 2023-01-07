#!/usr/bin/env bash

python train.py --backbone resnet50 --epochs 500 --warmup 5
python train.py --backbone cspresnet50 --epochs 500 --warmup 5
python train.py --backbone cspdarknet50 --epochs 500 --warmup 5