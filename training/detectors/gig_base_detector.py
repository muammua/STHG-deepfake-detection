
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv, GraphNorm

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='gig_base')
class GIGBaseDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.model = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        backbone = GIG_Model(num_class=2, num_segment=config['clip_size'], add_softmax=False)
        pretrained_path = config['pretrained']
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            state_dict = {k.replace("base_", "").replace("model.", ""): v for k, v in state_dict.items()}
            state_dict = {"base_model." + k: v for k, v in state_dict.items()}
            msg = backbone.load_state_dict(state_dict, False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{pretrained_path}'")
            torch.cuda.empty_cache()
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        bs, t, c, h, w = data_dict['image'].shape
        inputs = data_dict['image'].view(bs, t * c, h, w)
        pred, batch_out = self.model(inputs)
        return pred, batch_out

    def classifier(self, features: torch.tensor):
        pass

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].long()
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the prediction by backbone
        pred, batch_out = self.features(data_dict)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': batch_out}

        return pred_dict


class GIG_Model(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_segment=8,
                 add_softmax=False,
                 **kwargs):
        super().__init__()

        self.num_class = num_class
        self.num_segment = num_segment

        self.add_softmax = add_softmax

        self.build_model()

    def build_model(self):
        """
        Construct the model.
        """
        self.base_model = scnet50_v1d(self.num_segment, pretrained=True)

        fc_feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(fc_feature_dim, self.num_class)

        if self.add_softmax:
            self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.tensor): input tensor of shape (n, t*c, h, w). n is the batch_size, t is num_segment
        """
        # img channel default to 3
        img_channel = 3

        # x: [n, tc, h, w] -> [nt, c, h, w]
        # out: [nt, num_class]
        out, batch_out = self.base_model(
            x.view((-1, img_channel) + x.size()[2:])
        )

        out = out.view(-1, self.num_segment, self.num_class)  # [n, t, num_class]
        out = out.mean(1, keepdim=False)  # [n, num_class]

        if self.add_softmax:
            out = self.softmax_layer(out)

        return out, batch_out

    def set_segment(self, num_segment):
        """Change num_segment of the model.
        Useful when the train and test want to feed different number of frames.

        Args:
            num_segment (int): New number of segments.
        """
        self.num_segment = num_segment


model_urls = {
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
}



class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class SCBottleneck(nn.Module):
    """
    SCNet SCBottleneck. Variant for ResNet Bottlenect.
    """
    expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False),
            norm_layer(group_width),
        )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation

    def forward(self, x):
        """Forward func which splits the input into two branchs a and b.
        a: trace features
        b: spatial features
        """
        residual = x

        out_a = self.relu(self.bn1_a(self.conv1_a(x)))
        out_b = self.relu(self.bn1_b(self.conv1_b(x)))

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SCNet(nn.Module):
    def __init__(self, num_segments, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d):
        """SCNet, a variant based on ResNet.

        Args:
            num_segments (int):
                Number of input frames.
            block (class):
                Class for the residual block.
            layers (list):
                Number of layers in each block.
            num_classes (int, optional):
                Number of classification class.. Defaults to 1000.
            dilated (bool, optional):
                Whether to apply dilation conv. Defaults to False.
            dilation (int, optional):
                The dilation parameter in dilation conv. Defaults to 1.
            deep_stem (bool, optional):
                Whether to replace 7x7 conv in input stem with 3 3x3 conv. Defaults to False.
            stem_width (int, optional):
                Stem width in conv1 stem. Defaults to 64.
            avg_down (bool, optional):
                Whether to use AvgPool instead of stride conv when downsampling in the bottleneck. Defaults to False.
            avd (bool, optional):
                The avd parameter for the block Defaults to False.
            norm_layer (class, optional):
                Normalization layer. Defaults to nn.BatchNorm2d.
        """
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd
        self.num_segments = num_segments

        super(SCNet, self).__init__()
        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # from gigfake_detector import GIGFake
        self.gigfake = GIGFake(pixel_in_channels = [512, 256],
                               pixel_hidden_channels = [256, 128],
                               pixel_out_channels = [256, 128],
                               num_classes = 2,
                               edge_refine = False,
                               feature_dim = 2048,)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
        """
        Core function to build layers.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.num_segments, self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        feats = self.features(input)
        gig_log, batch_out = self.gigfake(feats)
        x = self.logits(feats) + gig_log
        return x, batch_out


def scnet50_v1d(num_segments, pretrained=False, **kwargs):
    """
    SCNet backbone, which is based on ResNet-50
    Args:
        num_segments (int):
            Number of input frames.
        pretrained (bool, optional):
            Whether to load pretrained weights.
    """
    model = SCNet(num_segments, SCBottleneck, [3, 4, 6, 3],
                  deep_stem=True, stem_width=32, avg_down=True,
                  avd=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50_v1d']), strict=False)

    return model


class GeneralGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GeneralGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        return x


class EdgeIndexGenerator(nn.Module):
    def __init__(self, in_channels):
        super(EdgeIndexGenerator, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, node_features, height=None, width=None):
        num_nodes = node_features.size(0)
        edge_index = []

        if height and width:
            for i in range(num_nodes):
                batch_idx = i // (height * width)
                pixel_idx = i % (height * width)
                x = pixel_idx % width
                y = pixel_idx // width
                edge_index.append([i, i])
                if x > 0:
                    left_idx = batch_idx * height * width + y * width + (x - 1)
                    edge_index.append([i, left_idx])
                    edge_index.append([left_idx, i])
                if x < width - 1:
                    right_idx = batch_idx * height * width + y * width + (x + 1)
                    edge_index.append([i, right_idx])
                    edge_index.append([right_idx, i])
                if y > 0:
                    up_idx = batch_idx * height * width + (y - 1) * width + x
                    edge_index.append([i, up_idx])
                    edge_index.append([up_idx, i])
                if y < height - 1:
                    down_idx = batch_idx * height * width + (y + 1) * width + x
                    edge_index.append([i, down_idx])
                    edge_index.append([down_idx, i])

            center_x = (width - 1) / 2
            center_y = (height - 1) / 2
            symmetric_x = int(2 * center_x - x)
            symmetric_y = int(2 * center_y - y)
            symmetric_idx = batch_idx * height * width + symmetric_y * width + symmetric_x
            if symmetric_idx != i:
                edge_index.append([i, symmetric_idx])
                edge_index.append([symmetric_idx, i])

        else:
            for i in range(num_nodes):
                edge_index.append([i, i])
                if i > 0:
                    edge_index.append([i, i - 1])
                    edge_index.append([i - 1, i])
                if i < num_nodes - 1:
                    edge_index.append([i, i + 1])
                    edge_index.append([i + 1, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index.to(node_features.device)

class GIGFake(nn.Module):
    def __init__(self, pixel_in_channels, pixel_hidden_channels, pixel_out_channels,
                 num_classes, edge_refine, feature_dim):
        super(GIGFake, self).__init__()

        self.first_convs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4 , pixel_in_channels[0], 3, 1, 1),
        )

        num_iterations = len(pixel_in_channels)

        batch_in_channels = [_ * 7 * 7 for _ in pixel_out_channels]
        batch_hidden_channels = [_ // 49 for _ in batch_in_channels]
        batch_out_channels = [_ * 7 * 7 for _ in pixel_out_channels]

        self.pixel_gcns = nn.ModuleList([
            GeneralGCN(pixel_in_channels[i], pixel_hidden_channels[i], pixel_out_channels[i])
            for i in range(num_iterations)
        ])

        self.batch_gcns = nn.ModuleList([
            GeneralGCN(batch_in_channels[i], batch_hidden_channels[i], batch_out_channels[i])
            for i in range(num_iterations)
        ])

        self.pixel_edge_generator = EdgeIndexGenerator(pixel_in_channels[0])
        self.batch_edge_generator = EdgeIndexGenerator(batch_in_channels[0])


        self.first_run = True
        self.pixel_edge_index = None
        self.batch_edge_index = None
        self.edge_refine = edge_refine
        self.classifier = nn.Sequential(
            nn.Linear(batch_out_channels[-1], num_classes),
        )

        self.num_iterations = num_iterations

    def forward(self, features):
        self.first_run = True
        if len(features.shape) == 5:
            main_batch_size, num, feature_channels, height, width = features.shape
            features = features.view(main_batch_size * num, feature_channels, height, width)
        else:
            main_batch_size = 1
            num, feature_channels, height, width = features.shape

        features = self.first_convs(features)

        _, channels, _, _ = features.shape

        pixel_features = features.permute(0, 2, 3, 1).reshape(-1,
                                                              channels)  # b*n, c, w, h -> b*n, w, h, c -> b*n*w*h, c

        for i in range(self.num_iterations):

            pixel_gcn = self.pixel_gcns[i]
            batch_gcn = self.batch_gcns[i]

            if self.first_run:
                self.pixel_edge_index = self.pixel_edge_generator(pixel_features, height, width)
            elif self.edge_refine:
                print("edge_refine is not available, please set edge_refine = False")

            pixel_output = pixel_gcn(pixel_features, self.pixel_edge_index)  # b*n*w*h, c
            batch_features = pixel_output.view(main_batch_size * num, -1)  # b*n, w*h*c

            if self.first_run:
                self.batch_edge_index = self.batch_edge_generator(batch_features)
                self.first_run = False
            elif self.edge_refine:
                print("edge_refine is not available, please set edge_refine = False")

            batch_output = batch_gcn(batch_features, self.batch_edge_index)  # b*n, w*h*c

            if i < self.num_iterations - 1:
                # batch -> pixel
                pixel_features = batch_output.view(main_batch_size * num * height * width, -1)  # b*n*w*h, c

        logits = self.classifier(batch_output)  # (bs * num, num_classes)
        return logits, batch_output