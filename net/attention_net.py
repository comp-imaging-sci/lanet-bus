# CBAM github: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
from re import S
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_2d(tensor):
    # log(sum(exp(x)))
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    # channel attention module
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        if 'avg' in pool_types:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if 'max' in pool_types:
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type=='lp':
                lp_pool = nn.LPPool2d(2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))(x)
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sig( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    # spatial attention module
    def __init__(self, kernel_size=3, num_conv=3):
        super(SpatialGate, self).__init__()
        kernel_size = kernel_size
        self.compress = ChannelPool()
        spatial = []
        init_inplane = 2
        inplane = init_inplane 
        expand = 4
        for i in range(num_conv-1):
            spatial.append(BasicConv(inplane, init_inplane * expand, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, relu=True))
            inplane = init_inplane * expand
        spatial.append(BasicConv(inplane, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False))
        self.spatial = nn.Sequential(*spatial)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sig(x_out) # broadcasting
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels=None, reduction_ratio=16, pool_types=['avg', 'max'], channel_att=True, spatial_att=True, attention_kernel_size=3, attention_num_conv=3):
        super(CBAM, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att 
        if channel_att:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        if spatial_att:
            self.SpatialGate = SpatialGate(kernel_size=attention_kernel_size, num_conv=attention_num_conv)

    def forward(self, x):
        if self.channel_att:
            c_scale = self.ChannelGate(x)
            x = x * c_scale
        if self.spatial_att:
            s_scale = self.SpatialGate(x)
            x = x * s_scale
        # return x, s_scale
        return x


class SaliencyNet(nn.Module):
    # upsmaple and concatenate intermediate features, then apply conv to predict the saliency map 
    def __init__(self, map_size, planes=None, cbam_param=None):
        # planes: channels of each features 
        super(SaliencyNet, self).__init__()
        # self._use_cbam = use_cbam
        # if self._use_cbam:
        self.cbams = nn.ModuleList()
        self.squeeze = nn.ModuleList()
        for plane in planes: 
            self.cbams.append(CBAM(plane, 
                            reduction_ratio=cbam_param.get("reduction_ratio", 16), 
                            channel_att=cbam_param.get("channel_att", True),
                            spatial_att=cbam_param.get("spatial_att", True),
                            attention_kernel_size=cbam_param.get("attention_kernel_size", 3),
                            attention_num_conv=cbam_param.get("attention_num_conv", 3)))
            self.squeeze.append(nn.Conv2d(plane, 1, kernel_size=3, stride=1, padding=1))
        self.upsample = nn.Upsample([map_size, map_size], mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # if self._use_cbam:
        # extract spatial feature map 
        s = []
        for i, f in enumerate(x):
            #_, si = self.cbams[i](f)
            si = self.cbams[i](f)
            si = self.squeeze[i](si)
            si = self.upsample(si)
            s.append(si)
        # concate
        x = torch.cat(s, axis=1) # unable to backprop 
        # average
        # x = torch.mean(x, 1).unsqueeze(1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.sig(x)
        return x

if __name__ == "__main__":
    import random
    random.seed(1)
    x1 = torch.rand(2, 16, 3, 3)
    random.seed(1)
    x2 = torch.rand(2, 16, 6, 6)
    random.seed(1)
    x3 = torch.rand(2, 16, 12, 12)
    random.seed(1)
    x4 = torch.rand(2, 16, 12, 12)
    x = [x1, x2, x3, x4]
    planes = [16, 16, 16, 16]
    cbam_param = {"channel_att": True, "spatial_att": True, "attention_kernel_size": 3, "attention_num_conv":3}
    sl = SaliencyNet(8, planes, cbam_param=cbam_param)
    # s_pretrain_state=torch.load("../model/best_model.pt", map_location="cpu")
    # cur_s_state = self.saliency.state_dict()
    # new_s_state_dict={k:v if v.size()==cur_s_state[k].size()  else  cur_s_state[k] for k,v in zip(cur_s_state.keys(), s_pretrain_state.values())}
    # sl.load_state_dict(s_pretrain_state, strict=False)
    y = sl(x)
    print(y)
