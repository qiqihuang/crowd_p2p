from collections import OrderedDict, namedtuple
import argparse
import datetime
import random
import time
from pathlib import Path
import torch.autograd.profiler as profiler

import torch
import torchvision.transforms as standard_transforms
import numpy as np
from torch import nn
import thop
from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='./',
                        help='path where to save')
    parser.add_argument('--weights', default='./ckpt/vgg16bn_best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--profile', action='store_true', help='profile gogo')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    return parser

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def profile(input, ops, n=10, device=None):
    results = []
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0
            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results

def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    if args.profile:
        model, _ = build_model(args, training=True)
        model = model.to(device)
    else:
        model = build_model(args).to(device)
    # move to GPU
    # load trained model
    if args.profile == False and args.weights is not None:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    img_path = "./vis/demo1.jpg"
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

    if args.profile:
        im = torch.zeros(1, 3, 768, 1280).to(device)
        model = model.train()
        profile(im, model, device=device)
        with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
            y = model(im)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))

        for _ in range(10):
            _ = model(im)
            
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(im)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
            print('Rep %d ' % repetitions + 'avg timing: %.3fms' % np.mean(timings))
        return 1
    else:
        im = torch.zeros(1, 3, new_height, new_width).to(device)
        for _ in range(5):
            y = model(im)
    
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0).repeat(2, 1, 1, 1)
    samples = samples.to(device)

    outputs = model(samples)
    outputs_scores = outputs['pred_logits'][0]
    outputs_points = outputs['pred_points'][0]

    threshold = 0.5

    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)