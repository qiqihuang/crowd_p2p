import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd
from .common import Conv
import numpy as np

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif t is nn.Parameter:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.output(x)
        bs, _, _, _ = x.shape
        x = x.permute(0, 2, 3, 1)
        return x.contiguous().view(bs, -1, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, _, width, height = x.shape

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, self.num_anchor_points, self.num_classes).contiguous()

        return x.view(batch_size, -1, self.num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(self.strides[idx], row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, 
                               padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=256, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = nn.LeakyReLU(negative_slope=0)
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = nn.LeakyReLU(negative_slope=0)
        self.upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_05x = nn.Upsample(scale_factor=0.5, mode='nearest')
        
        self._init_weights()

    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * self.upsample_2x(p7_td))        
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * self.upsample_2x(p6_td))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * self.upsample_2x(p5_td))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * self.upsample_2x(p4_td))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * self.upsample_05x(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * self.upsample_05x(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * self.upsample_05x(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * self.upsample_05x(p6_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
    def _init_weights(self):
        initialize_weights(self)

class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, num_layers=2, epsilon=0.0001):
        super(Decoder, self).__init__()
        self.p3 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        
        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        
        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, inputs):
        c3, c4, c5 = inputs
        
        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)        
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c5)
        p7_x = self.p7(p6_x)
        
        features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        return self.bifpn(features)

class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, name='vgg'):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        num_anchor_points = row * line

        # self.regression_P7 = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        # self.classification_P7 = ClassificationModel(num_features_in=256, \
        #                                     num_classes=self.num_classes, \
        #                                     num_anchor_points=num_anchor_points)

        # self.regression_P6 = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        # self.classification_P6 = ClassificationModel(num_features_in=256, \
        #                                     num_classes=self.num_classes, \
        #                                     num_anchor_points=num_anchor_points)

        # self.regression_P5 = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        # self.classification_P5 = ClassificationModel(num_features_in=256, \
        #                                     num_classes=self.num_classes, \
        #                                     num_anchor_points=num_anchor_points)

        self.regression_P4 = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification_P4 = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        # self.regression_P3 = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        # self.classification_P3 = ClassificationModel(num_features_in=256, \
        #                                     num_classes=self.num_classes, \
        #                                     num_anchor_points=num_anchor_points)
        self.concat = Concat()
        self.anchor_points = AnchorPoints(pyramid_levels=[3, 4, 5, 6], row=row, line=line)


        if name == 'resnet50':
            self.fpn = Decoder(512, 1024)
        elif name == 'cspresnet50':
            self.fpn = Decoder(256, 512)
        elif name == 'cspdarknet53':
            self.fpn = Decoder(256, 512)
        else:
            self.fpn = Decoder(256, 512, 512)
        # initialize_weights(self)

    def forward(self, samples: NestedTensor):
        batch_size = samples.shape[0]

        features = self.backbone(samples)
        fpn_features = self.fpn(features)

        # reg_P7 = self.regression_P7(fpn_features[4]) * 100 # 8x
        # cls_P7 = self.classification_P7(fpn_features[4])

        # reg_P6 = self.regression_P6(fpn_features[3]) * 100 # 8x
        # cls_P6 = self.classification_P6(fpn_features[3])

        # reg_P5 = self.regression_P5(fpn_features[2]) * 100 # 8x
        # cls_P5 = self.classification_P5(fpn_features[2])

        reg_P4 = self.regression_P4(fpn_features[1]) * 100 # 8x
        cls_P4 = self.classification_P4(fpn_features[1])

        # reg_P3 = self.regression_P3(P3) * 100 # 8x
        # cls_P3 = self.classification_P3(P3)

        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)

        #regression = regression.sigmoid()
        output_coord = self.concat([reg_P4]) + anchor_points
        output_class = self.concat([cls_P4])
        out = {'pred_logits': output_class, 'pred_points': output_coord}
       
        return out

class SetCriterion_Crowd(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

# create the P2PNet model
def build(args, training):
    # treats persons as a single class
    num_classes = 1

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line, args.backbone)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion