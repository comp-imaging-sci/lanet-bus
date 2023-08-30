import torch 
try:
    from net.model import get_model
except:
    from .net.model import get_model
from data import prepare_data
import fire
import numpy as np
from collections import OrderedDict
from util import batch_iou, dice_score, read_image_tensor, draw_segmentation_mask, get_image_mask, show_mask_on_image
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from sklearn.metrics import roc_auc_score, auc, roc_curve
from torchattacks import PGD
import os

BUSI_LABELS_BINARY = ["malignant", "benign"]
MAYO_LABELS = ["Malignant", "Benign"]


def calculate_auc(pred, label, num_classes=3):
    fpr, tpr, roc_auc = {}, {}, {}
    # label = label_binarize(label, classes=list(range(num_classes)))
    label = np.eye(num_classes)[label]
    if num_classes > 1:
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        fpr[0], tpr[0], _ = roc_curve(label, pred)
        roc_auc[0] = auc(fpr[0], tpr[0])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):    
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr = mean_tpr / num_classes
    tpr["macro"] = mean_tpr
    fpr["macro"] = all_fpr
    macro_auc = auc(all_fpr, mean_tpr)
    roc_auc["macro_auc"] = macro_auc
    return roc_auc, tpr, fpr


class Eval():
    def __init__(self, model_name, 
                 num_classes, 
                 model_weights,  
                 image_size=256, 
                 device="cuda:0",
                 dataset="BUSI",
                 multi_gpu=False,
                 use_cam=False,
                 use_sam=True,
                 use_mam=True,
                 reduction_ratio=16, 
                 attention_num_conv=3, 
                 attention_kernel_size=3,
                 map_size=14,
                 adv_attack=False,
                 return_mask=True):
        super(Eval, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model_weights = model_weights
        self.image_size = image_size
        self.device = device
        self.dataset = dataset
        self.multi_gpu = multi_gpu
        self.use_cam = use_cam
        self.use_sam = use_sam
        self.use_mam = use_mam
        self.reduction_ratio = reduction_ratio
        self.attention_num_conv = attention_num_conv
        self.attention_kernel_size = attention_kernel_size
        self.map_size = map_size
        self.return_mask = return_mask
        self.adv_attack = adv_attack
        self.load_model()
        if self.adv_attack:
            self.atk = PGD(self.model, eps=8/255, alpha=2/255, steps=4, random_start=True)
            self.atk.set_normalization_used(mean=[0.5], std=[0.5])
    
    def load_model(self):
        cbam_param = dict(use_cam=self.use_cam, 
                        use_sam=self.use_sam,
                        use_mam=self.use_mam,
                        reduction_ratio=self.reduction_ratio, 
                        attention_num_conv=self.attention_num_conv, 
                        attention_kernel_size=self.attention_kernel_size,
                        device=self.device,
                        backbone_weights="")
        self.model = get_model(model_name=self.model_name, 
                          num_classes=self.num_classes, 
                          use_pretrained=True, 
                          map_size=self.map_size,
                          return_mask=self.return_mask,
                          image_size=self.image_size,
                          **cbam_param).to(self.device)
        state_dict=torch.load(self.model_weights, map_location=torch.device(self.device))
        if self.multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' due to data_parallel
                new_state_dict[name]=v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def image2mask(self, 
                   seg_image_list=None, 
                   mask_save_file=None, 
                   mask_thres=0.5,
                  ):
        # load images in the seg_image_list if exists
        # draw mask instead of computing the IOU values or other metrics
        image_df = pd.read_csv(seg_image_list, header=None)
        images = image_df.iloc[:, 0]
        if self.dataset == "BUSI":
            mask_coord = [None] * len(images)
        elif self.dataset in ["MAYO", "MAYO_bbox"]:
            mask_str = image_df.iloc[:, -1].tolist()
            mask_coord = np.array([x.split(":") for x in mask_str], dtype=int)
        image_list = []
        real_mask_list = []
        for i, image in enumerate(images):
            image_tensor = read_image_tensor(image, self.image_size)
            mask = get_image_mask(image, self.image_size, dataset=self.dataset, mask_coord=mask_coord[i])
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
            if self.model_name in ["resnet50_rmtl_mask", "resnet18_rmtl_mask", "resnet18_cbam_mask", "resnet50_cbam_mask"]:
                # interpolate mask to original size
                prob = torch.nn.functional.interpolate(outputs[1], size=(self.image_size, self.image_size), mode="bilinear", align_corners=True)
                mask_pred = torch.where(prob>mask_thres, 1, 0)
        draw_segmentation_mask(image_tensor, real_mask_tensor, mask_pred, mask_save_file) 
    
        
    def accuracy(self, test_file, return_auc=True):
        train_file = test_file
        config = {"image_size": self.image_size, 
                  "train": train_file, 
                  "test": test_file, 
                  "dataset": self.dataset,
                  "mask": self.model_name in ["deeplabv3", "unet", "resnet50_rmtl_mask", "resnet18_rmtl_mask", "resnet18_cbam_mask", "resnet50_cbam_mask"],
                  "dilute_mask": 0,
                 }
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        result_matrics = np.zeros((2, 2))
        pred_list, label_list = [], []
        
        for data in dataloader:
            inputs = data["image"].to(self.device)
            labels = data["label"].to(self.device)
            tag = labels.cpu().numpy()[0]
            if self.adv_attack:
                inputs = self.atk(inputs, labels, return_logit=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                if self.model_name in ["resnet50_rmtl_mask", "resnet18_rmtl_mask", "resnet18_cbam_mask", "resnet50_cbam_mask"]:
                    outputs = outputs[0]
                _, pred = torch.max(outputs, 1)
                score = outputs.cpu().numpy()
                pred = int(pred.item())
                result_matrics[tag][pred] += 1
                if return_auc:
                    if len(pred_list) == 0:
                        pred_list = score
                    else:
                        pred_list = np.concatenate([pred_list, score], axis=0)
                    label_list.append(tag)
        # precision: TP / (TP + FP)
        print("result matrics: ", result_matrics)
        # res_acc = [result_matrics[i, i]/np.sum(result_matrics[:,i]) for i in range(num_classes)]
        res_pre = []
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
            pre = TP/(TP+FP)
            acc = (TP+TN) / (TP+TN+FP+FN)
            sens = TP/(TP+FN)
            speci = TN/(TN+FP)
            f1 = 2*TP/(2*TP+FP+FN)
            res_pre.append(pre)
            res_acc.append(acc)
            res_speci.append(speci)
            res_sens.append(sens)
            f1_score.append(f1)      
        print('Accuracy: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_acc[0],res_acc[1], np.mean(res_acc)))
        print('Precision: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_pre[0],res_pre[1], np.mean(res_pre)))
        print('Sensitivity: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_sens[0], res_sens[1], np.mean(res_sens)))
        print('Specificity: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_speci[0],res_speci[1], np.mean(res_speci)))
        print('F1 score: w/o: {0:.3f}, with: {1:.3f}, avg{2:.3f}'.format(f1_score[0],f1_score[1], np.mean(f1_score)))
        if return_auc:
            auc_v, tpr, fpr = calculate_auc(pred_list, label_list, num_classes=self.num_classes)
            print("AUC: ", auc_v)
            # print("FPR: ", fpr)
            # print("TRP", tpr)
    
    def iou(self, test_file, mask_thres=0.5):
        train_file = test_file
        config = {"image_size": self.image_size, 
                "train": train_file, 
                "test": test_file, 
                "dataset": self.dataset, 
                "mask": True}
        image_datasets, data_sizes = prepare_data(config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False) 

        iou_res = []
        dice_res = []
        with torch.no_grad():
            for data in dataloader:
                if data["mask_exist"] == 0:
                    continue
                img = data["image"].to(self.device)
                outputs = self.model(img)
                mask = data["mask"].to(self.device)
                mask_size = mask.shape[-1]
                if self.model_name in ["unet", "deeplabv3"]:
                    mask_pred = outputs
                elif self.model_name in ["resnet50_rmtl_mask", "resnet18_rmtl_mask", "resnet18_cbam_mask", "resnet50_cbam_mask"]:
                    mask_pred = outputs[1]
                mask_pred = torch.nn.functional.interpolate(mask_pred, size=(mask_size, mask_size), mode="bilinear", align_corners=True)
                mask_pred = torch.where(mask_pred>mask_thres, 1, 0)
                
                iou = batch_iou(mask_pred, mask, 2)
                dice = dice_score(mask_pred, mask)
                iou_res.append(iou[0])
                dice_res.append(dice)

        print("Localization IOU: ", np.mean(iou_res))
        print("Localization Dice: ", np.mean(dice_res))

    def saliency(self, image_path, target_category=None, saliency_path=None, method="grad-cam"):
        image_tensor = read_image_tensor(image_path, self.image_size).to(self.device)
        try:
            # target_layers = [self.model.net.layer4[-1][-1]]
            target_layers = [self.model.net.layer4[-1]]
        except:
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
        cv2.imwrite(saliency_path, visualization)
        print("Draw saliency map with {} done! Save in {}".format(method, saliency_path))

    
if __name__ == "__main__":
    fire.Fire(Eval)