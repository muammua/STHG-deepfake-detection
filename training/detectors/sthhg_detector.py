
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
import logging
from networks import BACKBONE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch_geometric.data import Data
from hyper0 import encoders
from hyper0 import manifolds
import argparse
from torchvision.models import swin_v2_t
import torchvision.models as models
from types import SimpleNamespace

logger = logging.getLogger(__name__)

__all__ = ['SCNet', 'scnet50', 'scnet101', 'scnet50_v1d', 'scnet101_v1d']

model_urls = {
    'scnet50': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50-dc6a7e87.pth',
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
    'scnet101': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet101-44c5b751.pth',
    # 'scnet101_v1d': coming soon...
}


@DETECTOR.register_module(module_name='sthhg')
class STHHGDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)


        # 从YAML配置加载参数
        hyper_params = config.get('hypergraph_params', {})
        args0 = SimpleNamespace(**hyper_params.get('args0', {}))
        args1 = SimpleNamespace(**hyper_params.get('args1', {}))
        # 为每个尺度创建不同的encoder
        scales = config.get('scales')

        self.hyper_encoders = nn.ModuleList([HC(args0) for _ in scales])


        self.hyper_encoder6 = HC(args1)
        infeature_dim = config['infeature_dim']
        self.head = nn.Linear(infeature_dim, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        backbone_name = config['backbone_name']
        if backbone_name == 'swin_v2_t':
            backbone = swin_v2_t(pretrained=True)
            logger.info('Successfully downloaded and loaded pretrained swin_v2_b model.')
        elif backbone_name == 'scnet50':
            backbone = scnet50(pretrained=True)
        elif backbone_name == 'scnet50_v1d':
            backbone = scnet50_v1d(pretrained=True)
        elif backbone_name == 'scnet101':
            backbone = scnet101(pretrained=True)
        elif backbone_name == 'scnet101_v1d':
            backbone = scnet101_v1d(pretrained=True)
        elif backbone_name == 'mc3':
            backbone = models.video.mc3_18(pretrained=False)
            r3d_state = torch.load(config['pretrained'])
            backbone.load_state_dict(r3d_state)
            logger.info('Successfully downloaded and loaded pretrained mc3_18 model.')

        else:
            # 如果不是SCNet系列，使用原来的逻辑
            backbone_class = BACKBONE[backbone_name]
            model_config = config['backbone_config']
            backbone = backbone_class(model_config)
            pretrained = config.get('pretrained', False)
            if pretrained:
                state_dict = torch.load(config['pretrained'])
                for name, weights in state_dict.items():
                    if 'pointwise' in name:
                        state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
                state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
                backbone.load_state_dict(state_dict, False)
                logger.info('Load pretrained model successfully!')

        # 移除最后的全连接层
        if hasattr(backbone, 'fc'):
            backbone = nn.Sequential(*list(backbone.children())[:-3])


        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # 获取输入张量的形状
        b, t, c, h, w = data_dict['image'].shape
        # 将帧维度与批次维度合并

        # 通过backbone提取特征
        if self.config['backbone_name'] in ['scnet50', 'scnet50_v1d', 'scnet101', 'scnet101_v1d']:
            frame_input = data_dict['image'].reshape(-1, c, h, w)
            outputs = self.backbone(frame_input)
        elif self.config['backbone_name'] == 'swin_v2_t':
            frame_input = data_dict['image'].reshape(-1, c, h, w)
            outputs = self.backbone.features(frame_input)
            outputs = outputs.permute(0, 3, 1, 2)
        elif self.config['backbone_name'] == 'mc3':
            frame_input = data_dict['image'].permute(0, 2, 1, 3, 4)
            outputs = self.backbone(frame_input)
            outputs = outputs.permute(0, 2, 1, 3, 4)
            b, t, c, h, w = outputs.shape
            outputs = outputs.reshape(-1, c, h, w)
            # outputs = self.pooling(outputs)
        else:
            frame_input = data_dict['image'].reshape(-1, c, h, w)
            outputs = self.backbone.features(frame_input)

        # outputs = self.pooling(outputs)
        c_out = outputs.shape[1]
        h_out = outputs.shape[2]
        w_out = outputs.shape[3]
        outputs = outputs.reshape(b, t, c_out, h_out, w_out)

        return outputs

    def classifier(self, features: torch.tensor) -> torch.tensor:
        b, t, c = features.shape
        # 将特征展平为二维张量
        features = features.view(-1, c)
        pred = self.head(features)
        # 恢复形状为 (8, 8, 2)
        pred = pred.view(b, t, 2)
        return pred

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
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
        self.video_names = []
        return metric_batch_dict


    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)  # 8,8,2048,8,8
        b, t, c, h, w = features.shape

        scales = self.config.get('scales', [1, 2, 3, 4])
        scales_length = len(scales)
        hyper_ns = []

        # 构建超图并编码
        for i, scale in enumerate(scales):
            hypergraph = STHC(features, scale, 0.7)
            # 使用对应尺度的encoder
            hyper_n = self.hyper_encoders[i].encode(hypergraph)
            hyper_ns.append(hyper_n)


        # 处理编码后的特征
        feature_nodes = []
        for i, scale in enumerate(scales):
            c_out = hyper_ns[i].shape[1]
            hyper_n = hyper_ns[i].reshape(b, t, scale, scale, c_out)
            hyper_n = hyper_n.permute(0, 1, 4, 2, 3)
            feature_node = hyper_n.reshape(b * t, c_out, scale, scale)
            feature_node = F.adaptive_avg_pool2d(feature_node, (1, 1)).squeeze(-1)
            feature_nodes.append(feature_node)

        # 将三个特征拼接为 (b*t, c, 3)
        fused_features = torch.stack(feature_nodes, dim=2)
        fused_features = fused_features.reshape(b, t, c_out, scales_length)
        graph0 = MSHF(fused_features)
        hyper_n4 = self.hyper_encoder6.encode(graph0)

        c_out = hyper_n4.shape[1]
        hyper_n4 = hyper_n4.reshape(b, t, scales_length, c_out)
        hyper_n4 = hyper_n4.permute(0, 1, 3, 2)
        hyper_n4 = hyper_n4.mean(dim=-1).squeeze(-1)

        pred = self.classifier(hyper_n4)
        # get the probability of the pred
        pred = pred.mean(dim=1)
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': hyper_n4}

        return pred_dict


# 初始化并查集
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def _calculate_similarity_matrix(features):
    """计算并归一化相似度矩阵"""
    similarity_matrix = torch.mm(features, features.t())
    return (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())


def _build_hyperedges_by_union_find(node_indices, features, threshold, uf_class):
    """根据并查集构建超边"""
    similarity_matrix = _calculate_similarity_matrix(features)
    uf = uf_class(len(node_indices))

    # 合并相似节点
    similar_pairs = (similarity_matrix > threshold).nonzero(as_tuple=False)
    for pair in similar_pairs:
        uf.union(pair[0], pair[1])

    # 构建超边
    hyperedges = {}
    for i in range(len(node_indices)):
        root = uf.find(i)
        if root not in hyperedges:
            hyperedges[root] = []
        hyperedges[root].append(node_indices[i])

    return hyperedges


def STHC(features: torch.tensor, scale: int, similarity_threshold: float) -> torch.tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, C, h, w = features.shape


    # 确保 B, T, N 是整数类型
    B = int(B)
    T = int(T)
    H = W = int(scale)
    N = int(H * W)

    # 特征预处理
    features = features.reshape(B * T, C, h, w)
    pooled_features = nn.functional.adaptive_avg_pool2d(features, (scale, scale))
    pooled_features = pooled_features.reshape(B, T, C, scale, scale)
    pooled_features = pooled_features.reshape(B, T, C, -1)



    node_features = pooled_features.permute(0, 1, 3, 2).reshape(-1, C)
    batch = torch.tensor([b for b in range(B) for _ in range(T * N)], dtype=torch.long).to(device)

    hyperedge_indices = []
    similarity_threshold = similarity_threshold

    # 1. Temporal prior rule
    for b in range(B):
        for n in range(N):
            hyperedge = [b * T * N + t * N + n for t in range(T)]
            hyperedge_indices.append(hyperedge)

    # 2. Spatial prior rule
    for b in range(B):
        for t in range(T):
            # 对称性超边
            for h in range(H):
                for w in range(W // 2):
                    node_index = b * T * H * W + t * H * W + h * W + w
                    sym_node_index = b * T * H * W + t * H * W + h * W + (W - 1 - w)
                    hyperedge = sorted([node_index, sym_node_index])
                    if hyperedge not in hyperedge_indices:
                        hyperedge_indices.append(hyperedge)

            # 聚合性超边
            for h in range(H):
                for w in range(W):
                    node_index = b * T * H * W + t * H * W + h * W + w
                    neighbors = [b * T * H * W + t * H * W + (h + dh) * W + (w + dw)
                                 for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 if 0 <= (h + dh) < H and 0 <= (w + dw) < W]
                    if neighbors:
                        hyperedge = [node_index] + neighbors
                        hyperedge_indices.append(hyperedge)

    # 3. Adaptive rules
    # 3.1 Joint adaption
    hyperedges = _build_hyperedges_by_union_find(
        range(B * T * N), node_features, similarity_threshold, UnionFind)
    for root, nodes in hyperedges.items():
        if len(nodes) > 1 and nodes not in hyperedge_indices:
            hyperedge_indices.append(nodes)

    # 3.2 Spatial adaption
    for b in range(B):
        for t in range(T):
            node_indices = [b * T * N + t * N + n for n in range(N)]
            current_features = node_features[node_indices]
            hyperedges = _build_hyperedges_by_union_find(
                node_indices, current_features, similarity_threshold, UnionFind)
            for root, nodes in hyperedges.items():
                if len(nodes) > 1 and nodes not in hyperedge_indices:
                    hyperedge_indices.append(nodes)

    # 3.3 Temporal adaption
    for b in range(B):
        for n in range(N):
            node_indices = [b * T * N + t * N + n for t in range(T)]
            current_features = node_features[node_indices]
            hyperedges = _build_hyperedges_by_union_find(
                node_indices, current_features, similarity_threshold, UnionFind)
            for root, nodes in hyperedges.items():
                if len(nodes) > 1 and nodes not in hyperedge_indices:
                    hyperedge_indices.append(nodes)

    # 构建最终的超图
    index_node = []
    index_hyperedge = []
    for i, hyperedge in enumerate(hyperedge_indices):
        index_node.extend(hyperedge)
        index_hyperedge.extend([i] * len(hyperedge))

    hyperedge_index = torch.tensor([index_node, index_hyperedge], dtype=torch.long).to(device)
    return Data(x=node_features, hyperedge_index=hyperedge_index, batch=batch,
                hyperedge_attr=None, hyperedge_weights=None)


def MSHF(features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, C, N = features.shape
    # 将特征张量展平为 (B * T * N, C)
    node_features = features.permute(0, 1, 3, 2).reshape(-1, C)
    hyperedge_indices = []
    hyperedge_count = 0

    # 模拟生成 data.batch，每个批次内的节点有相同的 batch 索引
    batch = []
    for b in range(B):
        batch.extend([b] * (T * N))
    batch = torch.tensor(batch, dtype=torch.long).to(device)

    # 对于相同 T 不同 N 的节点用一个超边连接
    for b in range(B):
        for t in range(T):
            hyperedge = []
            for n in range(N):
                node_index = b * T * N + t * N + n
                hyperedge.append(node_index)
            hyperedge_indices.append(hyperedge)
            hyperedge_count += 1

    # 对于相同 N 不同 T 的节点用同一个超边连接
    for b in range(B):
        for n in range(N):
            hyperedge = []
            for t in range(T):
                node_index = b * T * N + t * N + n
                hyperedge.append(node_index)
            hyperedge_indices.append(hyperedge)
            hyperedge_count += 1

    index_node = []
    index_hyperedge = []

    for i, hyperedge in enumerate(hyperedge_indices):
        index_node.extend(hyperedge)
        index_hyperedge.extend([i] * len(hyperedge))

    hyperedge_index = torch.tensor([index_node, index_hyperedge], dtype=torch.long).to(device)


    # 构建 Data 对象，包含超边权重
    data = Data(x=node_features, hyperedge_index=hyperedge_index, batch=batch, hyperedge_attr=None,
                hyperedge_weights=None)

    return data


def build_hypergraph(features):
    """
    构建超图的函数

    :param features: 输入特征张量，形状为 (B, T, C, N)
    :return: 超边索引，形状为 (2, num_hyperedges) 的 torch.Tensor，
             节点特征，形状为 (B * T * N, C) 的 torch.Tensor，
             data.batch，形状为 (B * T * N,) 的 torch.Tensor
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, C, N = features.shape
    # 将特征张量展平为 (B * T * N, C)
    node_features = features.permute(0, 1, 3, 2).reshape(-1, C)
    hyperedge_indices = []
    hyperedge_count = 0

    # 模拟生成 data.batch，每个批次内的节点有相同的 batch 索引
    batch = []
    for b in range(B):
        batch.extend([b] * (T * N))
    batch = torch.tensor(batch, dtype=torch.long).to(device)

    # 对于相同 T 不同 N 的节点用一个超边连接
    for b in range(B):
        for t in range(T):
            hyperedge = []
            for n in range(N):
                node_index = b * T * N + t * N + n
                hyperedge.append(node_index)
            hyperedge_indices.append(hyperedge)
            hyperedge_count += 1

    # 对于相同 N 不同 T 的节点用同一个超边连接
    for b in range(B):
        for n in range(N):
            hyperedge = []
            for t in range(T):
                node_index = b * T * N + t * N + n
                hyperedge.append(node_index)
            hyperedge_indices.append(hyperedge)
            hyperedge_count += 1

    index_node = []
    index_hyperedge = []

    for i, hyperedge in enumerate(hyperedge_indices):
        index_node.extend(hyperedge)
        index_hyperedge.extend([i] * len(hyperedge))

    hyperedge_index = torch.tensor([index_node, index_hyperedge], dtype=torch.long).to(device)

    # 添加超边特征
    num_hyperedges = hyperedge_index[1].max() + 1
    hyperedge_attr = torch.zeros(num_hyperedges, features.size(-2)).to(device)

    # 使用超边连接的节点特征的平均值初始化超边特征
    for hyperedge_idx in range(num_hyperedges):
        node_indices = hyperedge_index[0][hyperedge_index[1] == hyperedge_idx]
        if len(node_indices) > 0:
            hyperedge_attr[hyperedge_idx] = node_features[node_indices].mean(dim=0)

    # 确保 Linear 层在 GPU 上
    linear_layer = torch.nn.Linear(hyperedge_attr.size(1), 1).to(device)
    hyperedge_weights = torch.sigmoid(linear_layer(hyperedge_attr)).squeeze()

    # 构建 Data 对象，包含超边权重
    data = Data(x=node_features, hyperedge_index=hyperedge_index, batch=batch, hyperedge_attr=hyperedge_attr,
                hyperedge_weights=hyperedge_weights)

    return data

    # # 构建类似原代码中的 data 变量
    # data = Data(x=node_features, hyperedge_index=hyperedge_index, batch=batch)
    # return data


class HC(nn.Module):
    def __init__(self, args):
        super(HC, self).__init__()
        self.embedding_dim = args.hidden_dim
        if args.readout == 'concat':
            self.embedding_dim *= args.num_layers

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        # self.c = torch.tensor([args.c]).to(device)
        self.c = torch.nn.Parameter(torch.tensor([args.c], device=device))
        self.manifold = getattr(manifolds, args.manifold)()
        self.encoder = getattr(encoders, args.model)(self.c, args)  # .to(args.device)
        self.num_layers = args.num_layers

    def encode(self, data):
        # 添加数值稳定性检查
        with torch.no_grad():
            self.c.data.clamp_(min=1e-3, max=1.0)  # 限制c的范围


        o = torch.zeros_like(data.x)
        data.x = torch.cat([o[:, 0:1], data.x], dim=1)
        data.x = self.manifold.expmap0(data.x, self.c)

        # o = torch.zeros_like(data.hyperedge_attr)
        # data.hyperedge_attr = torch.cat([o[:, 0:1], data.hyperedge_attr], dim=1)

        hBol_hGra_n_f = self.encoder.encode(
            data)

        return hBol_hGra_n_f




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
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
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
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

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
    """ SCNet Variants Definations
    Parameters
    ----------
    block : Block
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block.
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained SCNet yielding a stride-8 model.
    deep_stem : bool, default False
        Replace 7x7 conv in input stem with 3 3x3 conv.
    avg_down : bool, default False
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck.
    norm_layer : object
        Normalization layer used (default: :class:`torch.nn.BatchNorm2d`).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    def __init__(self, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd

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
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
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
        elif dilation==2:
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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
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
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def scnet50(pretrained=False, **kwargs):
    """Constructs a SCNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3],
                deep_stem=False, stem_width=32, avg_down=False,
                avd=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50']))
    return model

def scnet50_v1d(pretrained=False, **kwargs):
    """Constructs a SCNet-50_v1d model described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.
    `ResNeSt: Split-Attention Networks <https://arxiv.org/pdf/2004.08955.pdf>`_.

    Compared with default SCNet(SCNetv1b), SCNetv1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 3x3 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50_v1d']))
    return model

def scnet101(pretrained=False, **kwargs):
    """Constructs a SCNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 23, 3],
                deep_stem=False, stem_width=64, avg_down=False,
                avd=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet101']))
    return model

def scnet101_v1d(pretrained=False, **kwargs):
    """Constructs a SCNet-101_v1d model described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.
    `ResNeSt: Split-Attention Networks <https://arxiv.org/pdf/2004.08955.pdf>`_.

    Compared with default SCNet(SCNetv1b), SCNetv1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 3x3 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 23, 3],
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet101_v1d']))
    return model

