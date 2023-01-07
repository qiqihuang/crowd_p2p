# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

import models.vgg_ as models
import models.resnet_ as resnet
from models.common import Conv
import timm

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool = True):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            self.body1 = nn.Sequential(*features[:33])
            self.body2 = nn.Sequential(*features[33:43])

    def forward(self, x):
        x = self.body1(x)
        return x, self.body2(x)


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool = True):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=False)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)

class BackboneBase_ResNet(nn.Module):
    def __init__(self, backbone: nn.Module, deepstem=False):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4
        x = self.layer1(x) # 1/4
        x = self.layer2(x) # 1/8

        return x, self.layer3(x) # 1/8, 1/16

class Backbone_ResNet(BackboneBase_ResNet):
    def __init__(self, name: str):
        if name == 'resnet50':
            backbone = resnet.resnet50(pretrained=True)
        
        super().__init__(backbone)

class BackboneBase_CSPResNet(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        self.stem = backbone.stem
        self.stage1 = backbone.stages[0]
        self.stage2 = backbone.stages[1]
        self.stage3 = backbone.stages[2]
        #self.stage4 = backbone.stages[3]
    
    def forward(self, x):
        x = self.stem(x) # 1/4
        x = self.stage1(x) # 1/4
        x = self.stage2(x) # 1/8
        # x = self.stage3(x) # 1/16
        # x = self.stage4(x) # 1/32

        return x, self.stage3(x)

class Backbone_CSPResNet(BackboneBase_CSPResNet):
    def __init__(self, name: str):
        if name == 'cspresnet50':
            backbone = timm.create_model('cspresnet50', pretrained=True)
        
        super().__init__(backbone)

class BackboneBase_CSPDarkNet(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        self.stem = backbone.stem
        self.stage1 = backbone.stages[0]
        self.stage2 = backbone.stages[1]
        self.stage3 = backbone.stages[2]
        self.stage4 = backbone.stages[3]
        #self.stage5 = backbone.stages[4]
    
    def forward(self, x):
        x = self.stem(x) # 1
        x = self.stage1(x) # 1/2
        x = self.stage2(x) # 1/4
        x = self.stage3(x) # 1/8
        # x = self.stage4(x) # 1/16
        # x = self.stage5(x) # 1/32

        return x, self.stage4(x)

class Backbone_CSPDarkNet(BackboneBase_CSPDarkNet):
    def __init__(self, name: str):
        if name == 'cspdarknet53':
            backbone = timm.create_model('cspdarknet53', pretrained=True)
        
        super().__init__(backbone)

def build_backbone(args):
    if 'vgg16_bn' == args.backbone:
        backbone = Backbone_VGG(args.backbone)
    elif 'resnet50' == args.backbone:
        backbone = Backbone_ResNet(args.backbone)
    elif 'cspresnet50' == args.backbone:
        backbone = Backbone_CSPResNet(args.backbone)
    elif 'cspdarknet53' == args.backbone:
        backbone = Backbone_CSPDarkNet(args.backbone)
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16_bn')