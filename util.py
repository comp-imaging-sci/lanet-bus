import torch 
import numpy as np
from sklearn.metrics import jaccard_score
import cv2 
import os, re
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms.functional as TF
from typing import Sequence
import random

# evaluate segmentation performance via IOU
def batch_iou(pred, mask, num_classes):
    """Get batched IOU value of the class indicated in the mask"""
    pred = pred.data.type(torch.int).cpu().numpy()
    mask = mask.data.type(torch.int).cpu().numpy()
    labels = list(range(num_classes))
    ious = []
    for i in range(len(pred)):
        iou = jaccard_score(pred[i].ravel(), mask[i].ravel(), labels=labels, average=None)
        # print(iou)
        # print(np.max(pred))
        mask_label = np.max(mask[i])
        if mask_label > 0:
            target_iou = iou[mask_label]
            ious.append(target_iou)
        else:
            ious.append(0)
    return ious

def dice_score(pred, mask):
    # flatten the true and predicted masks
    mask = mask.data.type(torch.int).cpu().numpy().flatten()
    pred = pred.data.type(torch.int).cpu().numpy().flatten()
    # calculate the intersection and union between the masks

    intersection = np.sum(mask * pred)
    union = np.sum(mask) + np.sum(pred)

    # calculate the dice score
    dice = (2 * intersection) / union

    return dice


class CustomRandomRotate:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def draw_segmentation_mask(image_tensor, real_mask_tensor, pred_mask_tensor, save_file):
    """
    image_tensor: input image tensor for predicting masks, 4D (N, C, H, W)
    real_mask_tensor: real mask tensor of input image tensor
    pred_mask_tensor: mask tensor generated, 4D (N, 1, H, W)
    """
    figure = np.array([])
    for i in range(len(image_tensor)):
        img = image_tensor[i].cpu().numpy()
        # restore
        img = (img + 1.) / 2. * 255
        pred_mask = pred_mask_tensor[i].cpu().numpy()
        pred_mask = pred_mask.repeat(3, axis=0)
        pred_mask = pred_mask * 255
        real_mask = real_mask_tensor[i].cpu().numpy()
        real_mask = real_mask.repeat(3, axis=0) 
        img = np.concatenate([img, real_mask, pred_mask], axis=2)
        img_pair = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        img_pair = cv2.cvtColor(img_pair, cv2.COLOR_RGB2BGR)
        if figure.size == 0:
            figure = img_pair
        else:
            figure = np.concatenate([figure, img_pair], axis=0)
    cv2.imwrite(save_file, figure)


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


def get_image_mask(image_name, image_size=None, dataset="BUSI", mask_coord=None):
    """
    Get image mask by giving image name
    dataset: the name of the dataset
    mask_coord: list of mask box coordinates, xyxy
    """
    image_dir = os.path.dirname(image_name)
    image_base_name = os.path.basename(image_name).replace(".png", "")
    if dataset == "BUSI":
        # BUSI dataset has corresponding mask image file 
        mask_name = os.path.join(image_dir, image_base_name+"_mask.png")
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # load all mask if exist
        count = 1
        while True:
            mask_name = os.path.join(image_dir, image_base_name+"_mask_{}.png".format(count))
            if os.path.exists(mask_name):
                aux_mask = cv2.imread(mask_name)
                aux_mask = cv2.cvtColor(aux_mask, cv2.COLOR_BGR2GRAY) 
                mask = mask + aux_mask
                count += 1
            else:
                break
    elif dataset == "MAYO":
        # MAYO dataset has bounding box[left x, left y, right x, right y] as rough mask information
        box = mask_coord
        image = cv2.imread(image_name)
        img_h, img_w, _ = image.shape
        # set mask region as 255
        mask = np.zeros((img_h, img_w))
        mask[box[1]:box[3], box[0]:box[2]] = 255
    mask = mask.astype(np.uint8)
    if image_size:
        mask = cv2.resize(mask, (image_size, image_size))
    return mask


def dilute_mask(mask, dilute_distance=0):
    """Expand mask regions with given distance
    mask: image with dim (H, W, C)
    """
    if dilute_distance == 0: # no dilution
        return mask
    background = np.zeros_like(mask)
    # left, right, up, down dilute
    l_dilute = background.copy() 
    r_dilute = background.copy()
    u_dilute = background.copy()
    d_dilute = background.copy()
    l_dilute[:, 0:-dilute_distance] = mask[:, dilute_distance:]
    r_dilute[:, dilute_distance:] = mask[:, 0:-dilute_distance]
    u_dilute[0:-dilute_distance, :] = mask[dilute_distance:, :]
    d_dilute[dilute_distance:, :] = mask[0:-dilute_distance, :]
    diluted = l_dilute | r_dilute | u_dilute | d_dilute
    return diluted


def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_mask_on_image(img, mask, mask_save_file, use_rgb=False, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    result =  np.uint8(255 * cam)
    cv2.imwrite(mask_save_file, result)
    

if __name__ == "__main__":
    pred = torch.tensor([[[0 ,0 ,0],[0, 1, 1], [0,0,1]]])
    mask = torch.tensor([[[0, 0, 0],[0, 1, 0], [0,0,0]]])
    res = batch_iou(pred, mask, 2)
    print(res)