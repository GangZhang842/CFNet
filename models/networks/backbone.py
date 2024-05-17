import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class conv3x3_bn(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1):
        super(conv3x3_bn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv3x3_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1):
        super(conv3x3_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1):
        super(conv3x3_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class bn_conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1):
        super(bn_conv3x3_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, bias=False):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class conv1x1_bn(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super(conv1x1_bn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv1x1_relu(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super(conv1x1_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv1x1_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super(conv1x1_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class bn_conv1x1_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super(bn_conv1x1_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class DeConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, bias=False):
        super(DeConv, self).__init__()
        # kernel_size = stride + 2 * padding
        kernel_size = 4
        padding = None
        if isinstance(stride, int):
            assert stride in [2, 4], "stride must be 2 or 4, but got {}".format(stride)
            padding = (kernel_size - stride) // 2
        elif isinstance(stride, (list, tuple)):
            assert all([x in [1, 2, 4] for x in stride]), "stride must be 1, 2 or 4, but got {}".format(stride)
            padding = [(kernel_size-s) // 2 for s in stride]
        else:
            raise NotImplementedError
        
        self.mod = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
        padding=padding, output_padding=0, groups=groups, bias=bias, dilation=1, padding_mode='zeros')
    
    def forward(self, x):
        return self.mod(x)


class DownSample2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DownSample2D, self).__init__()
        self.conv_branch = conv3x3_bn(in_planes, out_planes, stride=stride, dilation=1)
        self.pool_branch = nn.Sequential(
            conv1x1_bn(in_planes, out_planes),
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1, dilation=1)
        )

        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_conv = self.conv_branch(x)
        x_pool = self.pool_branch(x)
        x_out = self.act(x_conv + x_pool)
        return x_out


class ChannelAtt(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAtt, self).__init__()
        self.cnet = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            conv1x1_relu(channels, channels // reduction),
            conv1x1(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        #channel wise
        ca_map = self.cnet(x)
        x = x * ca_map
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, reduction=1, dilation=1, identity=True, use_att=True):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            conv3x3_bn_relu(in_planes, in_planes // reduction, stride=1, dilation=1),
            conv3x3_bn(in_planes // reduction, in_planes, stride=1, dilation=dilation)
        )

        self.use_att = use_att
        if self.use_att:
            self.channel_att = ChannelAtt(channels=in_planes, reduction=4)
        
        self.identity = identity
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.layer(x)
        if self.use_att:
            out = self.channel_att(out)
        
        if self.identity:
            return self.act(out + x)
        else:
            return self.act(out)


class PredBranch(nn.Module):
    def __init__(self, cin, cout):
        super(PredBranch, self).__init__()
        self.pred_layer = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, dilation=1)
        )
    
    def forward(self, x):
        pred = self.pred_layer(x)
        return pred