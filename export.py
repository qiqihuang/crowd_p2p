import argparse

from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def export_onnx(model, im, onnxfile, opset_version=13, dynamic=False, train=False):
    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        onnxfile,
        verbose=False,
        opset_version=opset_version,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=['images'],
        output_names=['pred_logits', 'pred_points'],
        dynamic_axes={
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,3,640,640)

            'pred_logits': {
                0: 'batch',
                1: 'points'},  # shape(1,25200,85)

            'pred_points': {
                0: 'batch',
                1: 'points'}  # shape(1,25200,85)
        } if dynamic else None)

    import onnx, onnxsim
    model_onnx = onnx.load(onnxfile) 
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, 'assert check failed'
    onnx.save(model_onnx, onnxfile)

def export_engine(onnx_file, save_engine, half=False):
    import tensorrt as trt
    dynamic = False
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f'failed to load ONNX file: {onnx_file}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    prefix=colorstr('TensorRT:')
    print(f'{prefix} Network Description:')
    for inp in inputs:
        print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
    
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    with builder.build_engine(network, config) as engine, open(save_engine, 'wb') as t:
        t.write(engine.serialize())

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='./',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./ckpt/vgg_best_mae.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--onnxfile', default=None, type=str, help='path where the onnx file saved')
    parser.add_argument('--enginefile', default=None, type=str, help='path where the onnx file saved')
    return parser

def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    model = build_model(args).to(device)
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()

    img_path = "./vis/demo1.jpg"
    img_raw = Image.open(img_path).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

    im = torch.zeros(1, 3, new_height, new_width).to(device)
    for _ in range(2):
        y = model(im)

    if args.onnxfile:
        export_onnx(model, im, '/'.join(['./onnx', args.onnxfile]), opset_version=13, dynamic=False, train=False)

    if args.onnxfile and args.enginefile:
        export_engine('/'.join(['./onnx', args.onnxfile]), '/'.join(['./tensorrt', args.enginefile]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Make Onnxfile Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)