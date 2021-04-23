import torch
import torch.nn as nn
from torchvision import models
from seg_net import DeepLabV3

class LogitResnet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, return_logit=False, use_pretrained=True):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, inputs):
        f = self.net(inputs)
        x = self.avgpool(f)
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit:
            return [x, l]
        return [x]

class LogitDensenet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, return_last_f=False, return_logit=False, use_pretrained=True):
        super(LogitDensenet, self).__init__()
        if model_name == "densenet161":
            model = models.densenet161(pretrained=use_pretrained)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=use_pretrained)
        else:
            print("unknown densenet structure")
        num_features = model.classifier.in_features
        self.return_logit = return_logit
        self.return_last_f = return_last_f
        self.fc = nn.Linear(num_features, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, inputs):
        x = self.net(inputs)
        f = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit and self.return_last_f:
            return [x, f, l]
        if self.return_last_f:
            return [x, f]
        if self.return_logit:
            return [x, l]
        return [x]

def get_model(model_name, num_classes, use_pretrained=True, return_logit=False, return_last_f=False):
    if model_name in ["resnet50", "resnet34", "resnet18"]:
        model = LogitResnet(model_name, num_classes, return_logit=return_logit, use_pretrained=use_pretrained)
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
        model = LogitDensenet(model_name, num_classes, return_logit, use_pretrained)
    elif model_name == "inception_v3":
        print("Warning: Inception V3 input size must be larger than 300x300")
        if use_pretrained:
            model = models.inception_v3(pretrained=True, aux_logits=False)
        else:
            model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "deeplabv3":
        model = DeepLabV3(num_classes=num_classes, pretrained=use_pretrained)
    else:
        print("unknown model name!")
    return model

# def decoder()

if __name__ == "__main__":
    inputs = torch.rand(2, 3, 224, 224)
    inputs = torch.rand(2, 3, 32, 32) 
    # model = get_model("resnet50", 3, use_pretrained=False, return_logit=True)
    # model = models.densenet161(pretrained=False, num_classes=3) 
    model = get_model("resnet50", 3, use_pretrained=False, return_logit=True) 
    # print(list(model.children())[:-1])
    # model = nn.Sequential(*list(model.children())[:-1])
    print(model(inputs)[1].size())
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t", name, param.shape)
    # print("-"*10)   
