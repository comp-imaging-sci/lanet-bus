import torch
import os
import torch.nn as nn
from torchvision import models
import timm

try:    
    from .seg_net import DeepLabV3, US_UNet
    from .resnet import resnet18_arch, resnet50_arch
    from .lanet import LANet
except:
    from seg_net import DeepLabV3, US_UNet
    from resnet import resnet18_arch, resnet50_arch
    from lanet import LANet


def initialize_weights(net):
    for l in net.modules():
        if isinstance(l, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(l.weight)
            # l.bias.data.fill_(0)
        if isinstance(l, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            nn.init.constant_(l.weight, 1.0)
            nn.init.constant_(l.bias, 0.0)

class LogitResnet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, use_pretrained=True, return_feature=False):
        super(LogitResnet, self).__init__()
        if model_name == "resnet50":
            model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        elif model_name == "resnet34":
            model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
        elif model_name == "resnet18":
            model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        else:
            print("unknown resnet model")
            exit()
        num_features = model.fc.in_features
        self.return_feature = return_feature
        self.net = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, inputs):
        f = self.net(inputs)
        x = self.avgpool(f)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.return_feature:
            return [x, f]
        return x

class LogitEfficientNet(nn.Module):
    def __init__(self, model_name, num_classes, use_pretrained=True):
        super(LogitEfficientNet, self).__init__()
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=use_pretrained)
        else:
            print("Only b0 is supported yet")
        num_features = model.classifier[1].in_features
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(num_features, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])

    def forward(self, inputs):
        x = self.net(inputs)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=2, image_size=384):
        super(ViT, self).__init__()
        # assert image_size in [384, 224], "Image size must be 384 or 224"
        if image_size == 384:
            pretrained_w = models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        elif image_size == 224:
            pretrained_w = models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        else: 
            pretrained_w = None
        model = models.vision_transformer.vit_b_16(weights=pretrained_w, image_size=image_size)

        last_layer = list(model.children())[-1][0]
        # rest_layer = list(model.children())[:-1]
        # self.net = nn.Sequential(
        #     *rest_layer,
        #     nn.Linear(in_features=last_layer.in_features, out_features=num_classes, bias=True),
        #     )
        model.heads[0] = nn.Linear(in_features=last_layer.in_features, out_features=num_classes, bias=True) 
        self.net = model
        
    def forward(self, inputs):
        x = self.net(inputs)
        return x

class TimmViT(nn.Module):
    def __init__(self, num_classes, image_size=256, use_pretrained=True):
        super(TimmViT, self).__init__()
        if image_size == 224:
            model_name = "vit_base_patch16_224"
        elif image_size == 384: 
            model_name = "vit_base_patch16_384"
        elif image_size == 256:
            model_name = "vit_base_patch16_gap_224"
        model = timm.create_model(model_name, pretrained=use_pretrained)
        n_features = model.head.in_features
        model.head = nn.Linear(n_features, num_classes)
        self.model = model 
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.model(inputs)
        return x

class LogitDensenet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, return_feature=False, use_pretrained=True):
        super(LogitDensenet, self).__init__()
        if model_name == "densenet161":
            model = models.densenet161(pretrained=use_pretrained)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=use_pretrained)
        else:
            print("unknown densenet structure")
        num_features = model.classifier.in_features
        self.return_feature = return_feature
        self.fc = nn.Linear(num_features, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, inputs):
        x = self.net(inputs)
        f = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.return_feature:
            return [x, f]
        return x

class ClassificationHead(nn.Module):
    def __init__(self, inchannels, num_classes):
        super(ClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inchannels, num_classes)
        initialize_weights(self)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class RasaeeUpsampleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, output_size):
        super(RasaeeUpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6(inplace=True)
        self.up = nn.Upsample([output_size, output_size], mode="bilinear", align_corners=True)
        self.bn = nn.BatchNorm2d(output_channel)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.bn(x)
        return x

class RasaeeMaskHead(nn.Module):
    """https://arxiv.org/abs/2108.04345"""
    def __init__(self, resnet_name, map_size=448):
        super(RasaeeMaskHead, self).__init__()
        if resnet_name == "resnet50":
            init_c = 2048
        elif resnet_name == "resnet18":
            init_c = 512
        self.block1 = RasaeeUpsampleBlock(init_c, 256, 16)
        self.block2 = RasaeeUpsampleBlock(256, 64, 112)
        self.block3 = RasaeeUpsampleBlock(64, 1, map_size)
        self.sig = nn.Sigmoid()
        initialize_weights(self)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.sig(x)
        return x 

class ResNetMask(nn.Module):
    """Resnet with mask attention module"""
    def __init__(self,  
                 model_name,
                 num_classes, 
                 use_pretrained=True, 
                 map_size=256,
                 return_mask=True):
        super(ResNetMask, self).__init__()
        model_info = model_name.split("_")
        resnet_name = model_info[0]
        self.net = LogitResnet(resnet_name, num_classes, return_feature=True, use_pretrained=use_pretrained)
        resnet_name = model_name.split("_")[0]
        assert model_name in ["resnet50_rasaee_mask", "resnet18_rasaee_mask"], "Model name must be resnet50_rasaee_mask or resnet18_rasaee_mask"
        self.mask_module = RasaeeMaskHead(resnet_name, map_size)
        if resnet_name == "resnet50":
            cls_c = 2048
        elif resnet_name == "resnet18":
            cls_c = 512
        self.c = ClassificationHead(cls_c, num_classes)
        self.return_mask = return_mask

    def forward(self, x):
        _, x = self.net(x)
        mask = self.mask_module(x)
        if self.attention:
            x = x + x * mask
        x = self.c(x)
        if self.return_mask:
            return [x, mask]
        else:
            return x


class ResNetCbam(nn.Module):
    def __init__(self, 
                 model_name,
                 num_classes, 
                 use_pretrained=True, 
                 map_size=256,
                 use_cam=True,
                 use_sam=True,
                 use_mam=True,
                 reduction_ratio=16, 
                 attention_kernel_size=3, 
                 attention_num_conv=3,
                 backbone_weights="",
                 lanet_weights="",
                 device="cuda:0",
                 return_mask=True):
        super(ResNetCbam, self).__init__()
        if model_name == "resnet18_cbam_mask":
            model = resnet18_arch
            planes = [64, 128, 256, 512]
        elif model_name == "resnet50_cbam_mask":
            model = resnet50_arch
            planes = [256, 512, 1024, 2048]
        else:
            raise ValueError("Model name must be resent18_cbam_mask or resnet50_cbam_mask")
        cbam_param = dict(sp_kernel_size=attention_kernel_size, 
                          sp_num_conv=attention_num_conv,
                          use_cam=use_cam,
                          use_sam=use_sam,
                          reduction_ratio=reduction_ratio,
                          )
        self.net = model(pretrained=use_pretrained, return_all_feature=True)
        if os.path.exists(backbone_weights):
            b_pretrain_state=torch.load(backbone_weights, map_location=device)
            # not load fc params
            # pretrain_state.pop("fc.weight")
            # pretrain_state.pop("fc.bias")
            cur_b_state = self.net.state_dict()
            # state.update(state_dict)
            new_b_state_dict={k:v if v.size()==cur_b_state[k].size()  else  cur_b_state[k] for k,v in zip(cur_b_state.keys(), b_pretrain_state.values())}
            self.net.load_state_dict(new_b_state_dict, strict=False)     
        num_features = planes[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        self.use_mam = use_mam
        self.return_mask = return_mask
        self.lanet = LANet(planes=planes, map_size=map_size, cbam_param=cbam_param).to(device)
        if os.path.exists(lanet_weights):
            s_pretrain_state=torch.load(lanet_weights, map_location=device)
            # cur_s_state = self.lanet.state_dict()
            # new_s_state_dict={k:v if v.size()==cur_s_state[k].size()  else  cur_s_state[k] for k,v in zip(cur_s_state.keys(), s_pretrain_state.values())}
            self.lanet.load_state_dict(s_pretrain_state, strict=False)
        else:
            initialize_weights(self.lanet)

    def forward(self, x):
        x, fs = self.net(x)
        mask = self.lanet(fs)
        if self.use_mam:
            x = x + mask * x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.return_mask:
            return [x, mask]
        else:
            return x


def get_model(model_name, 
              num_classes, 
              use_pretrained=True, 
              return_feature=False, 
              map_size=8,
              **kwargs):
    if model_name in ["resnet50", "resnet34", "resnet18"]:
        model = LogitResnet(model_name, num_classes, use_pretrained=use_pretrained, return_feature=return_feature)
    elif model_name == "vgg":
        model = models.vgg16_bn(pretrained=use_pretrained)
        in_features = 25088
        model.classifer = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif model_name in ["densenet121", "densenet161"]:
        model = LogitDensenet(model_name, num_classes, return_feature=return_feature, use_pretrained=use_pretrained)
    elif model_name == "efficientnet_b0":
        model = LogitEfficientNet(model_name, num_classes, use_pretrained=use_pretrained)
    elif model_name == "ViT":
        model = ViT(num_classes, image_size=kwargs.get("image_size", 256))
        # model = TimmViT(num_classes, image_size=224, use_pretrained=use_pretrained)
        # model.freeze()
    elif model_name == "deeplabv3":
        model = DeepLabV3(num_classes=num_classes, pretrained=use_pretrained)
    elif model_name == "unet":
        model = US_UNet(num_classes=num_classes, pretrained=use_pretrained)
    elif model_name in ["resnet50_rasaee_mask", "resnet18_rasaee_mask"]:
        model = ResNetMask(model_name, num_classes, use_pretrained=use_pretrained, map_size=map_size, return_mask=kwargs.get("return_mask", True))
    elif model_name in ["resnet18_cbam_mask", "resnet50_cbam_mask"]:
        model = ResNetCbam(model_name, num_classes, use_pretrained=use_pretrained, map_size=map_size,
                           use_cam=kwargs.get("use_cam", True),
                           use_sam=kwargs.get("use_sam", True),
                           use_mam=kwargs.get("use_mam", True),
                           reduction_ratio=kwargs.get("reduction_ratio", 16), 
                           attention_kernel_size=kwargs.get("attention_kernel_size", 3), 
                           attention_num_conv=kwargs.get("attention_num_conv", 3),
                           backbone_weights=kwargs.get("backbone_weights", ""),
                           lanet_weights=kwargs.get("lanet_weights", "",),
                           device=kwargs.get("device", "cuda:0"),
                           return_mask=kwargs.get("return_mask", True))

    else:
        print("unknown model name!")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    inputs = torch.rand(2, 3, 256, 256)
    # model = get_model("resnet18", 3, use_pretrained=True)
    model = get_model("resnet50_rasaee_mask", 3, use_pretrained=False, return_feature=True, device="cpu") 
    # model = ViT(2, 256) 

    res = model(inputs)
    try:
        print(res[0].shape, res[1].shape)
    except:
        print(res.shape)
    print(res)

    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t", name, param.shape)
    # print("-"*10)   
