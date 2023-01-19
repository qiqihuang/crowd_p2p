from collections import OrderedDict, namedtuple
import argparse
import datetime
import random
import time
from pathlib import Path
from util.general import letterbox
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
import tensorrt as trt

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

    parser.add_argument('--weights', default='./ckpt/vgg_best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    
    parser.add_argument('--enginefile', default=None, type=str, help='path where the engine file saved')
    
    parser.add_argument('--differ_test', default=False, action='store_true', help='test different')
    return parser

def main(args, debug=False):
    if args.enginefile is None:
        print('engine file is None')
        return -1
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    model = build_model(args).to(device)
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cnt = 1
    while True:
        if os.path.isdir('/'.join([args.output_dir, 'trt_run{}'.format(cnt)])) == False:
            os.mkdir('/'.join([args.output_dir, 'trt_run{}'.format(cnt)]))
            save_folder = '/'.join([args.output_dir, 'trt_run{}'.format(cnt)])
            break
        else:
            cnt += 1
        
    
    img_path = "./vis/demo1.jpg"
    img_raw = Image.open(img_path).convert('RGB')
    width, height = img_raw.size
    new_width = 1280
    new_height = 768
    numpy_image=np.array(img_raw)
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    img_raw, _, _ = letterbox(opencv_image, (768, 1280), auto=False)
    color_coverted = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_raw=Image.fromarray(color_coverted)
    # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

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
    
    ''' TensorRT Engine '''
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    with open(args.enginefile, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()
    bindings = OrderedDict()
    dynamic = False
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        if model.binding_is_input(index):
            if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                dynamic = True
                context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
            if dtype == np.float16:
                fp16 = True
        shape = tuple(context.get_binding_shape(index))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
    
    if dynamic and samples.shape != bindings['images'].shape:
        i_in, i_out = (model.get_binding_index(x) for x in ('images', 'output'))
        context.set_binding_shape(i_in, samples.shape)  # reshape if dynamic
        bindings['images'] = bindings['images']._replace(shape=im.shape)
        bindings['output'].data.resize_(tuple(context.get_binding_shape(i_out)))

    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    batch_size = bindings['images'].shape[0]

    s = bindings['images'].shape

    assert samples.shape == s, f"input size {samples.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
    binding_addrs['images'] = int(samples.data_ptr())
    context.execute_v2(list(binding_addrs.values()))

    trt_pred_logits = bindings['pred_logits'].data
    trt_pred_points = bindings['pred_points'].data

    if isinstance(trt_pred_logits, np.ndarray):
        trt_pred_logits = torch.tensor(trt_pred_logits, device=device)
    if isinstance(trt_pred_points, np.ndarray):
        trt_pred_points = torch.tensor(trt_pred_points, device=device)

    if args.differ_test:
        try: 
            # torch_out = np.array([to_numpy(outputs['pred_logits'].squeeze(0).unsqueeze(1)), to_numpy(outputs['pred_points'])])
            # trt_out = np.array([to_numpy(trt_pred_logits.squeeze(0).unsqueeze(1)), to_numpy(trt_pred_points)])
            torch_logits_out = to_numpy(outputs['pred_logits'])
            trt_logits_out = to_numpy(trt_pred_logits)
            np.testing.assert_allclose(torch_logits_out, trt_logits_out, rtol=1e-03, atol=1e-05)
            print("logits is no difference.")

        except AssertionError as msg:
            print("Error.")
            print(msg)

    if args.differ_test:
        try:
            torch_points_out = to_numpy(outputs['pred_points'])
            trt_points_out = to_numpy(trt_pred_points)
            np.testing.assert_allclose(torch_points_out, trt_points_out, rtol=1e-03, atol=1e-05)
            print("points is no difference.")

        except AssertionError as msg:
            print("Error.")
            print(msg)
            
    trt_outputs_scores = trt_pred_logits[0]
    trt_outputs_points = trt_pred_points[0][trt_outputs_scores > threshold]
    trt_predict_cnt = int((trt_outputs_scores > threshold).sum())

    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in trt_outputs_points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(save_folder, 'trt_pred{}.jpg'.format(trt_predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet TensorRT Test Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)