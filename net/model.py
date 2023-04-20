import torch
import os
import torch.nn as nn
from torchvision import models
import timm

try:    
    from .seg_net import DeepLabV3, US_UNet
    from .resnet_attention import resnet18, resnet50 
    from .attention_net import SaliencyNet
except:
    from seg_net import DeepLabV3, US_UNet
    from resnet_attention import resnet18, resnet50
    from attention_net import SaliencyNet


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
    def __init__(self, model_name, num_classes, return_logit=False, use_pretrained=True, return_feature=False):
        super(LogitResnet, self).__init__()
        if model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=use_pretrained)
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=use_pretrained)
        else:
            print("unknown resnet model")
            exit()
        num_features = model.fc.in_features
        self.return_logit = return_logit
        self.return_feature = return_feature
        self.net = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, inputs):
        f = self.net(inputs)
        x = self.avgpool(f)
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit and self.return_feature:
            return [x, f, l]
        if self.return_logit:
            return [x, l]
        if self.return_feature:
            return [x, f]
        return [x]

class LogitEfficientNet(nn.Module):
    def __init__(self, model_name, num_classes, return_logit=False, use_pretrained=True):
        super(LogitEfficientNet, self).__init__()
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=use_pretrained)
        else:
            print("Only b0 is supported yet")
        num_features = model.classifier[1].in_features
        self.return_logit = return_logit
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(num_features, num_classes)
        # self.fc2 = nn.Linear(embedding_dim, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])

    def forward(self, inputs):
        x = self.net(inputs)
        x = torch.flatten(x, 1)
        l = self.dropout(x)
        x = self.fc(l)
        # x = self.fc2(l)
        if self.return_logit:
            return [x, l]
        else:
            return [x]

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
        return [x]

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
        return [x]

class LogitDensenet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, return_feature=False, return_logit=False, use_pretrained=True):
        super(LogitDensenet, self).__init__()
        if model_name == "densenet161":
            model = models.densenet161(pretrained=use_pretrained)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=use_pretrained)
        else:
            print("unknown densenet structure")
        num_features = model.classifier.in_features
        self.return_logit = return_logit
        self.return_feature = return_feature
        self.fc = nn.Linear(num_features, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, inputs):
        x = self.net(inputs)
        f = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit and self.return_feature:
            return [x, f, l]
        if self.return_feature:
            return [x, f]
        if self.return_logit:
            return [x, l]
        return [x]


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MaskAttentionNet(nn.Module):
    """
    Learn mask attention map supervised by real masks 
    Augment features by lambda*(sigma(f)*f)+f, where f is input feature, 
    sigma(x) simulates the real mask information, lambda is attention weight
    """
    def __init__(self, reduction='mean', attention_weight=0.25):
        """
        reduction: method to reduce C channels into 1, 'mean' or 'max'
        """
        super(MaskAttentionNet, self).__init__()
        # self.bn = nn.BatchNorm2d(1)
        # self.relu = nn.LeakyReLU(0.02)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.act = nn.Sigmoid()
        self.reduction = reduction
        # self.attention_weight = attention_weight
        initialize_weights(self)

    def forward(self, x):
        # pool input feature from (N, C, H, W) to (N, 1, H, W)
        if self.reduction == "mean":
            x_pool = torch.mean(x, 1, keepdim=True)
        elif self.reduction == "max":
            x_pool, _ = torch.max(x, 1, keepdim=True)
        # x_map = self.bn(x_pool)
        # x_map = self.relu(x_map)
        x_map = self.conv(x_pool)
        x_map = self.act(x_map)
        # aug_f = x + self.attention_weight * x_map * x
        return x_map

class MaskAttentionNet2(nn.Module):
    """
    Optimized attention mask with continuous attention blocks which shrink the channel from 2048 to 1
    """
    def __init__(self, num_blocks=4):
        """
        reduction: method to reduce C channels into 1, 'mean' or 'max'
        """
        super(MaskAttentionNet2, self).__init__()
        init_channels = 2048
        channel_list = [init_channels//2**i for i in range(num_blocks)]
        channel_list.append(1)
        att_blocks = []
        for i in range(len(channel_list)-1):
            att_blocks.append(ConvBlock(channel_list[i], channel_list[i+1]))
        self.net = nn.Sequential(*att_blocks)
        self.act = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x_map = self.net(x)
        x_map = self.act(x_map)
        return x_map


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
                 num_blocks=3,
                 reduction='mean', 
                 attention_weight=0.5,
                 map_size=448):
        super(ResNetMask, self).__init__()
        model_info = model_name.split("_")
        resnet_name = model_info[0]
        self.attention = model_info[1] == "attention"
        self.attention_weight = attention_weight
        self.net = LogitResnet(resnet_name, num_classes, return_logit=False, return_feature=True, use_pretrained=use_pretrained)
        resnet_name = model_name.split("_")[0]
        if model_name in ["resnet50_attention_mask"]:
            # self.mask_module = MaskAttentionNet(reduction=reduction, attention_weight=attention_weight)
            self.mask_module = MaskAttentionNet2(num_blocks)
        elif model_name in ["resnet50_rasaee_mask", "resnet18_rasaee_mask"]:
            self.mask_module = RasaeeMaskHead(resnet_name, map_size)
        if resnet_name == "resnet50":
            cls_c = 2048
        elif resnet_name == "resnet18":
            cls_c = 512
        self.c = ClassificationHead(cls_c, num_classes)

    def forward(self, x):
        _, x = self.net(x)
        mask = self.mask_module(x)
        if self.attention:
            # x = x + self.attention_weight * mask * x
            x = x + x * mask
        x = self.c(x)
        return [x, mask]


class ResNetCbam(nn.Module):
    def __init__(self, 
                 model_name,
                 num_classes, 
                 use_pretrained=True, 
                 map_size=448,
                 use_mask=True,
                 channel_att=True,
                 spatial_att=True,
                 final_att=True,
                 reduction_ratio=16, 
                 attention_kernel_size=3, 
                 attention_num_conv=3,
                 backbone_weights="",
                 saliency_weights="",
                 device="cuda:0"):
        super(ResNetCbam, self).__init__()
        if model_name in ["resnet18_cbam_mask", "resnet18_cbam"]:
            model = resnet18
            planes = [64, 128, 256, 512]
        elif model_name in ["resnet50_cbam_mask", "resnet50_cbam"]:
            model = resnet50
            planes = [256, 512, 1024, 2048]
        self._use_mask = use_mask
        cbam_param = dict(sp_kernel_size=attention_kernel_size, 
                          sp_num_conv=attention_num_conv,
                          channel_att=channel_att,
                          spatial_att=spatial_att,
                          reduction_ratio=reduction_ratio,
                          )
        self.net = model(pretrained=use_pretrained, 
                        use_cbam=False,
                        cbam_param=None,
                        return_all_feature=True)
        if os.path.exists(backbone_weights):
            b_pretrain_state=torch.load(backbone_weights, map_location=device)
            # not load fc params
            # pretrain_state.pop("fc.weight")
            # pretrain_state.pop("fc.bias")
            cur_b_state = self.net.state_dict()
            # state.update(state_dict)
            new_b_state_dict={k:v if v.size()==cur_b_state[k].size()  else  cur_b_state[k] for k,v in zip(cur_b_state.keys(), b_pretrain_state.values())}
            self.net.load_state_dict(new_b_state_dict, strict=False)
            # self.net.load_state_dict(pretrain_state, strict=False)        
        # self.net = model
        num_features = planes[-1]
        # self.net = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        self.final_att = final_att
        if use_mask:
            self.saliency = SaliencyNet(planes=planes, map_size=map_size, cbam_param=cbam_param).to(device)
            if os.path.exists(saliency_weights):
                s_pretrain_state=torch.load(saliency_weights, map_location=device)
                # cur_s_state = self.saliency.state_dict()
                # new_s_state_dict={k:v if v.size()==cur_s_state[k].size()  else  cur_s_state[k] for k,v in zip(cur_s_state.keys(), s_pretrain_state.values())}
                self.saliency.load_state_dict(s_pretrain_state, strict=False)
            else:
                initialize_weights(self.saliency)

    def forward(self, x):
        x, fs = self.net(x)
        if self._use_mask:
            # print(self.saliency)
            # for param in self.saliency.parameters():
            #     print(param.data)
            mask = self.saliency(fs)
            # apply attention on the logit feature
            # if x.shape[0] == mask.shape[0]:
            if self.final_att:
                x = x + mask * x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self._use_mask:
            return [x, mask]
        else:
            return [x]


def get_model(model_name, 
              num_classes, 
              use_pretrained=True, 
              return_logit=False, 
              return_feature=False, 
              reduction='mean', 
              attention_weight=0.25,
              map_size=8,
              num_blocks=3,
              **kwargs):
    if model_name in ["resnet50", "resnet34", "resnet18"]:
        model = LogitResnet(model_name, num_classes, return_logit=return_logit, use_pretrained=use_pretrained, return_feature=return_feature)
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
        model = LogitDensenet(model_name, num_classes, return_logit=return_logit, return_feature=return_feature, use_pretrained=use_pretrained)
    elif model_name == "inception_v3":
        print("Warning: Inception V3 input size must be larger than 300x300")
        if use_pretrained:
            model = models.inception_v3(pretrained=True, aux_logits=False)
        else:
            model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "deeplabv3":
        # model = DeepLabV3(num_classes=num_classes, pretrained=use_pretrained)
        model = DeepLabV3(num_classes=num_classes, pretrained=False)
    elif model_name == "unet":
        model = US_UNet(num_classes=num_classes, pretrained=use_pretrained)
    elif model_name == "resnet50_attention_mask":
        model = ResNetMask(model_name, num_classes, use_pretrained=use_pretrained, reduction=reduction, attention_weight=attention_weight, num_blocks=num_blocks)
    elif model_name in ["resnet50_rasaee_mask", "resnet18_rasaee_mask"]:
        model = ResNetMask(model_name, num_classes, use_pretrained=use_pretrained, map_size=map_size)
    elif model_name in ["resnet18_cbam_mask", "resnet18_cbam", "resnet50_cbam_mask", "resnet50_cbam"]:
        model = ResNetCbam(model_name, num_classes, use_pretrained=use_pretrained, map_size=map_size,
                           use_mask=kwargs.get("use_mask", True),
                           channel_att=kwargs.get("channel_att", True),
                           spatial_att=kwargs.get("spatial_att", True),
                           final_att=kwargs.get("final_att", True),
                           reduction_ratio=kwargs.get("reduction_ratio", 16), 
                           attention_kernel_size=kwargs.get("attention_kernel_size", 3), 
                           attention_num_conv=kwargs.get("attention_num_conv", 3),
                           backbone_weights=kwargs.get("backbone_weights", ""),
                           saliency_weights=kwargs.get("saliency_weights", "",),
                           device=kwargs.get("device", "cuda:0"))
    elif model_name == "efficientnet_b0":
        model = LogitEfficientNet(model_name, num_classes, return_logit=return_logit, use_pretrained=use_pretrained)
    elif model_name == "ViT":
        # model = ViT(num_classes, image_size=256)
        # model = ViT(num_classes, image_size=224)
        model = TimmViT(num_classes, image_size=224, use_pretrained=use_pretrained)
        # model.freeze()
    else:
        print("unknown model name!")
    return model

# def decoder()

if __name__ == "__main__":
    torch.manual_seed(0)
    inputs = torch.rand(2, 3, 256, 256)
    # inputs = torch.ones(2, 3, 256, 256) * 0.5
    # inputs = torch.rand(2, 3, 32, 32) 
    # model = get_model("resnet18", 3, use_pretrained=True, return_logit=True)
    # model = models.densenet161(pretrained=False, num_classes=3) 
    # model = get_model("resnet50_attention_mask", 3, use_pretrained=False, return_logit=True, return_feature=True) 
    # print(list(model.children())[:-1])
    # model = nn.Sequential(*list(model.children())[:-1])
    backbone_w = "/Users/zongfan/Downloads/backbone_w.pt"
    # backbone_w = None
    # model = get_model(model_name="resnet50_cbam_mask", use_pretrained=True, image_size=256, num_classes=3, use_mask=True, channel_att=True, attention_kernel_size=3, attention_num_conv=3, backbone_weight=backbone_w, map_size=8)
    # res = model(inputs)
    # try:
    #     print(res[0].shape, res[1].shape)
    # except:
    #     print(res.shape)
    # print(res)
    model = ViT(2, 256) 
    # print(list(model.children())[:-1])
    # model = nn.Sequential(*list(model.children())[:-1])
    res = model(inputs)
    print(res)

    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t", name, param.shape)
    # print("-"*10)   
