import torch 
try:
    from net.model import get_model
except:
    from .net.model import get_model
from data import prepare_data
import fire
import numpy as np
from scipy import stats
from collections import OrderedDict
from util import batch_iou, read_image_tensor, draw_segmentation_mask, get_image_mask, show_mask_on_image
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

BUSI_LABELS = ["normal", "malignant", "benign"]
BUSI_LABELS_BINARY = ["malignant", "benign"]
ORIG_LABELS = ["malignant", "benign"]
MAYO_LABELS = ["Malignant", "Benign"]


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
    
    def image2mask(self, 
                   seg_image_list=None, 
                   mask_save_file=None, 
                   mask_thres=0.3,
                   #binary_mask=True
                  ):
        # load images in the seg_image_list if exists
        # draw mask instead of computing the IOU values or other metrics
        image_df = pd.read_csv(seg_image_list, header=None)
        images = image_df.iloc[:, 0]
        if self.dataset == "BUSI":
            mask_coord = None
        elif self.dataset in ["MAYO", "MAYO_bbox"]:
            mask_str = image_df.iloc[:, -1].tolist()
            mask_coord = np.array([x.split(":") for x in mask_str], dtype=int)
        image_list = []
        real_mask_list = []
        for image in images:
            image_tensor = read_image_tensor(image, self.image_size)
            mask = get_image_mask(image, self.image_size, dataset=self.dataset, mask_coord=mask_coord)
            # mask = mask / 255
            mask = np.expand_dims(mask, 0)
            mask = torch.tensor(mask)
            real_mask_list.append(mask)
            image_list.append(image_tensor)
        image_tensor = torch.stack(image_list).to(self.device)
        real_mask_tensor = torch.stack(real_mask_list)
        image_tensor = image_tensor.squeeze(1)
        outputs = self.model(image_tensor)
        if self.num_classes == 1:
            if self.model_name == "deeplabv3":
                prob = torch.nn.Sigmoid()(outputs)
                mask_pred = (prob>0.5).type(torch.int)
        else:
            if self.model_name in ["resnet50_mask", "resnet18_cbam_mask", "resnet50_cbam_mask"]:
                # interpolate mask to original size
                prob = torch.nn.functional.interpolate(outputs[1], size=(self.image_size, self.image_size), mode="bilinear", align_corners=True)
                mask_pred = torch.where(prob>mask_thres, 1, 0)
            #else:
            #    _, prob = torch.max(outputs, 1, keepdim=True)
        draw_segmentation_mask(image_tensor, real_mask_tensor, mask_pred, mask_save_file) 
        #if binary_mask:
        #    pred_mask_tensor = (prob>0.5).type(torch.int)
        #    draw_segmentation_mask(image_tensor, real_mask_tensor, pred_mask_tensor, mask_save_file) 
        #else:
        #    pred_mask_tensor = prob[0] # use first image
        #    img = (image_tensor[0]+1)/2 # scale to 0-1
        #    img = img.numpy().transpose([1, 2, 0])
        #    mask = pred_mask_tensor[0].cpu().detach().numpy()
        #    # mask = mask / np.max(mask)
        #    show_mask_on_image(img, mask, mask_save_file, use_rgb=False)
        
    def accuracy(self, test_file=None, binary_class=True):
        if test_file is None:
            if self.dataset == "BUSI":
                train_file = "data/busi_train_binary.txt"
                test_file = "data/busi_test_binary.txt"
            elif self.dataset == "MAYO":
                train_file = "data/mayo_train_mask_v2.txt"
                test_file  = "data/mayo_test_mask_v2.txt"
            elif self.dataset == "MAYO_bbox":
                train_file = "data/mayo_train_bbox.txt"
                test_file = "data/mayo_test_bbox.txt"
                self.dataset = "MAYO"
            elif self.dataset == "test_BUSI":
                train_file = "example/debug_BUSI.txt"
                test_file  = "example/debug_BUSI.txt"
                self.dataset = "BUSI"
            elif self.dataset == "test_MAYO":
                train_file = "example/debug_MAYO_mask.txt"
                test_file  = "example/debug_MAYO_mask.txt"
                self.dataset = "MAYO"
            elif self.dataset == "All":
                train_file = ["data/mayo_train_mask_v2.txt", "data/busi_train_binary.txt"]
                test_file = ["data/mayo_test_mask_v2.txt", "data/busi_test_binary.txt"]
                self.dataset = "All"
        else:
            train_file = test_file
        config = {"image_size": self.image_size, 
                  "train": train_file, 
                  "test": test_file, 
                  "dataset": self.dataset,
                  "mask": self.model_name in ["deeplabv3", "resnet50_mask", "resnet18_cbam_mask", "resnet50_cbam_mask"],
                  "dilute_mask": 0,
                 }
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        if self.dataset == "BUSI":
            if binary_class:
                result_matrics = np.zeros((2, 2))  
            else:
                result_matrics = np.zeros((3, 3)) 
        elif self.dataset in ["MAYO", "MAYO_bbox"]:
            result_matrics = np.zeros((2, 2)) 
        else:
            result_matrics = np.zeros((2, 2))
        with torch.no_grad():
            for data in dataloader:
                inputs = data["image"].to(self.device)
                labels = data["label"].to(self.device)
                tag = labels.cpu().numpy()[0]
                outputs = self.model(inputs)
                _, pred = torch.max(outputs[0], 1)
                # score = outputs[0].numpy()
                pred = int(pred.item())
                result_matrics[tag][pred] += 1
            # precision: TP / (TP + FP)
            print("result matrics: ", result_matrics)
            # res_acc = [result_matrics[i, i]/np.sum(result_matrics[:,i]) for i in range(num_classes)]
            res_acc = []
            # sensitivity: TP / (TP + FN)
            res_sens = []
            # res_sens = [result_matrics[i, i]/np.sum(result_matrics[i,:]) for i in range(num_classes)]
            # specificity: TN / (TN+FP)
            res_speci = []
            # f1 score: 2TP/(2TP+FP+FN)
            f1_score = []
            for i in range(self.num_classes):
                TP = result_matrics[i,i]
                FN = np.sum(result_matrics[i,:])-TP
                spe_matrics = np.delete(result_matrics, i, 0)
                FP = np.sum(spe_matrics[:, i])
                TN = np.sum(spe_matrics) - FP
                acc = TP/(TP+FP)
                sens = TP/(TP+FN)
                speci = TN/(TN+FP)
                f1 = 2*TP/(2*TP+FP+FN)
                res_acc.append(acc)
                res_speci.append(speci)
                res_sens.append(sens)
                f1_score.append(f1)      
        print('Precision: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_acc[0],res_acc[1], np.mean(res_acc)))
        print('Sensitivity: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_sens[0], res_sens[1], np.mean(res_sens)))
        print('Specificity: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_speci[0],res_speci[1], np.mean(res_speci)))
        print('F1 score: w/o: {0:.3f}, with: {1:.3f}, avg{2:.3f}'.format(f1_score[0],f1_score[1], np.mean(f1_score)))
    
    def iou(self, test_file=None, mask_thres=0.2):
        if test_file is None:
            if self.dataset == "BUSI":
                train_file = "data/busi_train_binary.txt"
                test_file = "data/busi_test_binary.txt"
            elif self.dataset == "MAYO":
                train_file = "data/mayo_train_mask_v2.txt"
                test_file = "data/mayo_test_mask_v2.txt"
            elif self.dataset == "MAYO_bbox":
                train_file = "data/mayo_train_bbox.txt"
                test_file = "data/mayo_test_bbox.txt"
                self.dataset = "MAYO"
            elif self.dataset == "test_BUSI":
                train_file = "example/debug_BUSI.txt"
                test_file = "example/debug_BUSI.txt"
                self.dataset = "BUSI"
            elif self.dataset == "test_MAYO":
                train_file = "example/debug_MAYO.txt"
                test_file = "example/debug_MAYO.txt"
                self.dataset = "MAYO"
        else:
            train_file = test_file
        config = {"image_size": self.image_size, 
                "train": train_file, 
                "test": test_file, 
                "dataset": self.dataset, 
                "mask": True}
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        result_matrics = []
        with torch.no_grad():
            for data in dataloader:
                img = data["image"].to(self.device)
                outputs = self.model(img)
                mask = data["mask"].to(self.device)
                if self.num_classes == 1:
                    prob = torch.nn.Sigmoid()(outputs)
                    mask_pred = (prob>0.5).type(torch.int)
                else:
                    # _, pred_mask_tensor = torch.max(outputs, 1, keepdim=True)
                    # print(torch.max(pred_mask_tensor), torch.max(outputs), outputs)
                    # pred_mask_tensor = (pred_mask_tensor>0).type(torch.int)
                    mask_size = mask.shape[-1]
                    mask_pred = torch.nn.functional.interpolate(outputs[1], size=(mask_size, mask_size), mode="bilinear", align_corners=True)
                    mask_pred = torch.where(mask_pred>mask_thres, 1, 0)
                iou = batch_iou(mask_pred, mask, 2)
                result_matrics.append(iou[0])
        print("Segmentation IOU: ", np.mean(result_matrics))

    def saliency(self, image_path, target_category=None, saliency_file=None, method="grad-cam"):
        image_tensor = read_image_tensor(image_path, self.image_size)
        try:
            # print(self.model.net.layer4[-1])
            # target_layers = [self.model.net.layer4[-1][-1]]
            target_layers = [self.model.net.layer4[-1]]
        except:
            # print(self.model.net[-1][-1])
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
    # model_weights = "/Users/zongfan/Downloads/res50_256_mask.pt"
    # model_weights = "/Users/zongfan/Downloads/deeplab_448.pt" 
    # model_weights = "/Users/zongfan/Downloads/res50_256_mayo.pt"
    model_weights = "/Users/zongfan/Downloads/res50_mask_256_mayo.pt" 
    seg_image_file = "test/busi_sample_binary.txt"
    mask_save_file = "test/busi_sample_mask_deeplab.png"
    # model_name = "resnet50_cbam_mask"
    # model_name = "deeplabv3"
    model_name = "resnet50_cbam_mask"
    img_size = 256
    num_classes = 2
    dataset = "BUSI"
    map_size = img_size // 32
    use_mask = True 
    channel_att = True
    mask_thres = 0.3
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

    # evaluator.image2mask(seg_image_file, mask_save_file, mask_thres=mask_thres)
    # evaluator.iou(test_file=seg_image_file, mask_thres=0.56)
    evaluator.saliency(image_path, saliency_file=saliency_file)
