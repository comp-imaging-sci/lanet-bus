import torch 
try:
    from net.model import get_model
except:
    from .net.model import get_model
import fire
import numpy as np
from scipy import stats
from collections import OrderedDict
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

BUSI_LABELS = ["normal", "malignant", "benign"]
BUSI_LABELS_BINARY = ["malignant", "benign"]
ORIG_LABELS = ["malignant", "benign"]
MAYO_LABELS = ["Malignant", "Benign"]

def read_image_tensor(image_path, image_size):
    # read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    # normalize
    img = img / 127.5 - 1 
    img = np.expand_dims(img, 0)
    # to tensor
    img = torch.tensor(img).type(torch.float32)
    # NHWC to NCHW
    img = img.permute(0, 3, 1, 2)
    return img


def mean_confidence_interval(x, confidence=0.95):
    # get CI with 0.95 confidence following normal gaussian distribution
    n = len(x)
    m, se = np.mean(x), stats.sem(x)
    ci = stats.t.ppf((1 + confidence) / 2., n-1) * se
    # ci = 1.96 * se  # assume gaussian distribution
    return m, ci


class Eval():
    def __init__(self, model_name, 
                 num_classes, 
                 model_weights,  
                 image_size=224, 
                 device="cpu",
                 dataset="covidx",
                 multi_gpu=False,
                 use_mask=True,
                 channel_att=False,
                 reduction_ratio=16, 
                 attention_num_conv=3, 
                 attention_kernel_size=3,
                 map_size=14,):
        super(Eval, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model_weights = model_weights
        self.image_size = image_size
        self.device = device
        self.dataset = dataset
        self.multi_gpu = multi_gpu
        self.use_mask = use_mask
        self.channel_att = channel_att
        self.reduction_ratio = reduction_ratio
        self.attention_num_conv = attention_num_conv
        self.attention_kernel_size = attention_kernel_size
        self.map_size = map_size
        self.load_model()
    
    def load_model(self):
        if self.use_mask:
            cbam_param = dict(channel_att=self.channel_att, 
                          reduction_ratio=self.reduction_ratio, 
                          attention_num_conv=self.attention_num_conv, 
                          attention_kernel_size=self.attention_kernel_size,
                          device=self.device,
                          backbone_weights="")
        else:
            cbam_param = {}
        self.model = get_model(model_name=self.model_name, 
                          num_classes=self.num_classes, 
                          use_pretrained=True, 
                          return_logit=False,
                          use_mask=self.use_mask,
                          map_size=self.map_size,
                          **cbam_param).to(self.device)
        state_dict=torch.load(self.model_weights, map_location=torch.device(self.device))
        if self.multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        

    def saliency(self, image_path, target_category=None, saliency_file=None, method="grad-cam"):
        image_tensor = read_image_tensor(image_path, self.image_size)
        try:
            # print(self.model.net.layer4[-1])
            # target_layers = [self.model.net.layer4[-1][-1]]
            # print("1", self.model.avgpool) 
            target_layers = [self.model.net.layer4[-1]]
        except:
            # print("2", self.model.net[-1][-1])
            target_layers = [self.model.net[-1][-1]]
        if method == "grad-cam":
            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=False)
        # target_category = [int(target_category)]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=image_tensor, target_category=target_category)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False)
        cv2.imwrite(saliency_file, visualization)
        print("Draw saliency map with {} done! Save in {}".format(method, saliency_file))

    
if __name__ == "__main__":
    # fire.Fire(Eval)
    model_weights = "test/res50_mask_256_busi.pt"
    model_name = "resnet50_cbam_mask"
    img_size = 256
    num_classes = 2
    dataset = "MAYO"
    map_size = img_size // 32
    use_mask = True
    channel_att = True
    mask_thres = 0.56
    multi_gpu = False
    image_path = "test/IM00033 annotated.png"
    saliency_file = "test/test_saliency_2.png"
    evaluator = Eval(model_name=model_name, 
                 num_classes=num_classes, 
                 model_weights=model_weights,  
                 image_size=img_size, 
                 device="cpu",
                 dataset=dataset,
                 multi_gpu=multi_gpu,
                 use_mask=use_mask,
                 channel_att=channel_att,
                 reduction_ratio=16, 
                 attention_num_conv=3, 
                 attention_kernel_size=3,
                 map_size=map_size)

    evaluator.saliency(image_path, saliency_file=saliency_file)
