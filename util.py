import torch 
import numpy as np
from sklearn.metrics import jaccard_score
import cv2 
import os


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
        target_iou = iou[mask_label]
        ious.append(target_iou)
    return ious

# TODO: lose gradient after logical calculation 
def ssl(pred, mask, sens_w=0.5):
    """Sensitivity specificity loss"""
    eps=1e-6
    confusing_matrix = (pred==1)&(mask==1)
    # confusing_matrix.requires_grad= True
    tp = torch.sum(confusing_matrix, axis=(1,2,3))
    fn = torch.sum((pred==0)&(mask==1), axis=(1,2,3))
    sens = tp/(tp+fn+eps)
    tn = torch.sum((pred==0)&(mask==0), axis=(1,2,3))
    fp = torch.sum((pred==1)&(mask==0), axis=(1,2,3))
    speci = tn/(tn+fp+eps)
    print(sens, speci)
    ssl = torch.sum(sens_w * sens + (1-sens_w) * speci)
    return ssl


def draw_segmentation_mask(image_tensor, real_mask_tensor, pred_mask_tensor, save_file):
    """
    image_tensor: input image tensor for predicting masks, 4D (N, C, H, W)
    real_mask_tensor: real mask tensor of input image tensor
    pred_mask_tensor: mask tensor generated, 4D (N, 1, H, W)
    """
    figure = np.array([])
    for i in range(len(image_tensor)):
        img = image_tensor[i].numpy()
        # restore
        img = (img + 1.) / 2. * 255
        pred_mask = pred_mask_tensor[i].numpy()
        pred_mask = pred_mask.repeat(3, axis=0)
        pred_mask = pred_mask * 255
        real_mask = real_mask_tensor[i].numpy()
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


def get_image_mask(image_name, image_size=None):
    """
    Get image mask by giving image name
    """
    image_dir = os.path.dirname(image_name)
    image_base_name = os.path.basename(image_name).replace(".png", "")
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
    # mask = mask / 255
    # mask = np.expand_dims(mask, 0).astype(np.uint8) 
    # mask = np.expand_dims(mask, 0).transpose(0, 3, 1, 2)
    # mask = torch.tensor(mask)
    mask = mask.astype(np.uint8)
    if image_size:
        mask = cv2.resize(mask, (image_size, image_size))
    return mask
    