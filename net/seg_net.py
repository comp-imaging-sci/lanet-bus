"""
Train a DeepLabV3 segmentation network
Ref: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
Paper: https://arxiv.org/pdf/1802.02611v3.pdf
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def initialize_weights(net):
    for l in net.modules():
        if isinstance(l, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(l.weight)
            # l.bias.data.fill_(0)
        if isinstance(l, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            nn.init.constant_(l.weight, 1.0)
            nn.init.constant_(l.bias, 0.0)

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepLabV3, self).__init__()
        # load model 
        self.model = models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT", progress=True)
        classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes-1)
        self.model.classifier = classifier
        # self.m = nn.LogSoftmax(dim=1)
        self.m = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.m(x['out'])
        return x

def deeplabv3(backbone="resnet50", num_classes=2, pretrained=True):
    if backbone == "resnet50":
        model = models.segmentation.deeplabv3_resnet50(pretrained_backbone=pretrained, num_classes=num_classes)
    elif backbone == "resnet101":
        model = models.segmentation.deeplabv3_resnet101(pretrained_backbone=pretrained, num_classes=num_classes)
    return model

# UNet: code link: https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class US_UNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(US_UNet, self).__init__()
        model = UNet(n_classes=2)
        if pretrained:   
            ckpt = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth"
            model.load_state_dict(torch.hub.load_state_dict_from_url(ckpt), progress=False)
        model.outc = (OutConv(64, num_classes-1)) 
        initialize_weights(model.outc)
        self.net = model
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.net(x)
        x = self.sig(x)
        return x

if __name__ == "__main__":
    model = DeepLabV3(2)
    # model = US_UNet(2, False)
    # state_dict=torch.load("unet_carvana_scale1.0_epoch2.pth")
    # model.load_state_dict(state_dict)

    # model = deeplabv3(num_classes=2)
    inputs = torch.rand(4,3,128,128)
    res = model(inputs)
    print(res.shape)
