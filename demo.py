import argparse
import datetime
import random
import time
from pathlib import Path
import torch
import torchvision.transforms as trans
import numpy as np

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
    parser.add_argument('--backbone', default='cspdarknet53', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--video_path', default='crowd_test_230120.mp4', type=str, help='asd')

    parser.add_argument('--img', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--folder', action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weights', default='ckpt/cspdarknet53_pafpn_p3p4_bestmae.pth', type=str,
                        help='path where the trained weights saved')

    parser.add_argument('--view-img', action='store_true',
                        help='view image')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def img2tensor(img):
    #the format of img needs to be bgr format

    img = img[..., ::-1]  #bgr2rgb
    img = img.transpose(2, 0, 1)  #(H, W, CH) -> (CH, H, W)
    img = np.ascontiguousarray(img)
    return img

def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    path = args.video_path
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # move to GPU

    model = model.to(device)
    # load trained model
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = trans.Compose([
        trans.ToTensor(), 
        trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    new_width = 1280
    new_height = 768
    out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (new_width, new_height))
            img_raw = frame.copy()
            #frame = img2tensor(frame)
            frame = transform(frame)
            samples = torch.Tensor(frame).unsqueeze(0)
            samples = samples.to(device)
            infer_start = time.time()
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = model(samples)
            infer_end = time.time()

            nms_start = time.time()
            outputs_scores = outputs['pred_logits'][0]
            outputs_points = outputs['pred_points'][0]
            threshold = 0.5
            # filter the predictions
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            nms_end = time.time()
            print('Inference time %.2f ms '%((infer_end - infer_start)*1000)+'NMS time %.2f ms'% ((nms_end - nms_start) * 1000))
            size = 2
            img_to_draw = img_raw

            crowd_count = len(points)
            for p in points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
                
            cv2.putText(img_to_draw, 'Count: %d'%crowd_count, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            out.write(img_to_draw)
            # if args.view_img:
            #     #cv2.resizeWindow('frame', 1080, 720)
            #     cv2.imshow('frame', img_to_draw)
            #     cv2.waitKey(1)
        else:
            print("Fail to read frame!")
            break
                
    cap.release()
    out.release()
        # cv2.destroyAllWindows()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
