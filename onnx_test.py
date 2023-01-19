from collections import OrderedDict, namedtuple
import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def softmax(x):
    max = np.max(x,axis=1,keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x,axis=1,keepdims=True)
    f_x = e_x / sum 
    return f_x

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='./test_folder',
                        help='path where to save')

    parser.add_argument('--weight_path', default='./ckpt/vgg_best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    
    parser.add_argument('--onnxfile', default=None, type=str, help='path where the onnx file saved')
    
    parser.add_argument('--differ_test', default=False, action='store_true', help='test different')
    return parser

def main(args, debug=False):
    if args.onnxfile is None:
        print('onnx file is None')
        return -1
        
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    model = build_model(args).to(device)
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cnt = 1
    while True:
        if os.path.isdir('/'.join([args.output_dir, 'onnx_run{}'.format(cnt)])) == False:
            os.mkdir('/'.join([args.output_dir, 'onnx_run{}'.format(cnt)]))
            save_folder = '/'.join([args.output_dir, 'onnx_run{}'.format(cnt)])
            break
        else:
            cnt += 1
        
    
    img_path = "./vis/demo1.jpg"
    img_raw = Image.open(img_path).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

    im = torch.zeros(1, 3, new_width, new_height).to(device)
    for _ in range(2):
        y = model(im)

    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)

    ''' Pytorch Model '''
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
    cv2.imwrite(os.path.join(save_folder, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

    ''' Onnx Model '''
    import onnxruntime
    
    session = onnxruntime.InferenceSession(args.onnxfile)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()
    output_name_1 = output_name[0].name
    output_name_2 = output_name[1].name
    data = to_numpy(samples)

    out = session.run([output_name_1, output_name_2], {input_name : data})

    if args.differ_test:
        try: 
            torch_out = np.array([to_numpy(outputs['pred_logits']), to_numpy(outputs['pred_points'])])
            np.testing.assert_allclose(torch_out, out, rtol=1e-03, atol=1e-05)
            print("no difference.")
        except AssertionError as msg:
            print("Error.")
            print(msg)

    onnx_outputs_scores = out[0]
    onnx_outputs_points = torch.from_numpy(out[1][0])[onnx_outputs_scores > threshold]
    onnx_predict_cnt = int((onnx_outputs_scores > threshold).sum())

    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in onnx_outputs_points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(save_folder, 'onnx_pred{}.jpg'.format(onnx_predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Make Onnxfile Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)