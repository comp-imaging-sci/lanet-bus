import torch
from torchvision import transforms, datasets
from PIL import Image
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import glob, os 
import pandas as pd 
import numpy as np 
import random
from util import get_image_mask, dilute_mask, CustomRandomRotate


BUSI_LABELS = ["malignant", "benign"]
MAYO_LABELS = ["Malignant", "Benign"] 

class BUSI_dataset(Dataset):
    def __init__(self, csv_file, transform=None, mask=False, mask_transform=None, mask_dilute=0):
        """
        csv_file: csv file containing image file path and corresponding label
        transform: transform for image
        mask: return mask or not
        mask_transform: transformation for mask
        mask_dilute: dilute mask with given distance in all directions
        """
        df = pd.read_csv(csv_file, sep=",", header=None)
        if len(df.columns) == 3:
            df.columns = ["img", "label", "bbox"]
            # bbox coordinates are in the format of "top letf x: top left y: bottom right x: bottom right y. e.g.: 340:177:493:294"
            self._img_bbox = df["bbox"].tolist()
        else:
            df.columns = ["img", "label"]
        self._img_files = df["img"].tolist()
        self._img_labels = df["label"].tolist()
        self._transform = transform
        self._mask = mask
        self._mask_transform = mask_transform
        self._mask_dilute = mask_dilute

    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        image_name = self._img_files[idx]
        assert os.path.exists(image_name), "Image file not found!"
        # load image
        img = Image.open(image_name)
        img = img.convert("RGB")
        label = self._img_labels[idx]
        label_id = BUSI_LABELS.index(label)
        # get the identical random seed for both image and mask
        seed = random.randint(0, 2147483647)
        if self._transform:
            random.seed(seed)
            torch.manual_seed(seed)
            img = self._transform(img)
        # load mask
        mask = []
        mask_exist = 0
        if self._mask:
           mask = get_image_mask(image_name, dataset="BUSI")
           mask = dilute_mask(mask, dilute_distance=self._mask_dilute)
           # assign class label
           mask = Image.fromarray(mask)
           mask_exist = 1
           if self._mask_transform:
               random.seed(seed)
               torch.manual_seed(seed)
               # torch.set_rng_state(state)
               mask = self._mask_transform(mask)
               mask = mask.type(torch.float)
               # mask = mask * label_id  # normal case is identical to backaground
           try:
               if self._img_bbox[idx] == "0:0:0:0":
                   mask = torch.zeros_like(img, dtype=torch.float)
                   mask_exist = 0
           except:
               pass
        return {"image": img, "label": label_id, "mask": mask, "mask_exist": mask_exist}


# input image width/height ratio
BUSI_IMAGE_RATIO = 570/460
# skip padding for MAYO video data
MAYO_VIDEO_SKIP_PADDING = [75, 75, 25, 25]


class MAYO_dataset(Dataset):
    """Mayo dataset preprocessor"""
    def __init__(self, csv_file, crop_image_ratio=None, skip_padding=[75, 75, 25, 25], transform=None, mask=False, mask_transform=None, mask_dilute=0):
        """
        csv_file: files conting images paths for training or testing
        crop_image_ratio: if given, crop a window from raw image first and the window should have this width/height ratio
        skip_padding: number of pixels to skip along each side: left, right, top, bottom
        """
        df = pd.read_csv(csv_file, sep=",", header=None)
        if len(df.columns == 3):
            df.columns = ["img", "label", "annotate"]
        elif len(df.columns) == 3:
            df.columns = ["img", "label"]
        self._img_files = df["img"].tolist()
        self._img_labels = df["label"].tolist()
        self._transform = transform
        self._mask = mask and ("annotate" in df.columns)
        self._skip_padding = skip_padding
        self._crop_image_ratio = crop_image_ratio
        if mask:
            self._mask_transform = mask_transform
            self._mask_dilute = mask_dilute
            mask_str = df["annotate"].tolist()
            self._mask_coord = np.array([x.split(":") for x in mask_str], dtype=int)
    
    def __len__(self):
        return len(self._img_files)
    
    def crop(self, frame):
        # remove padding first
        img_w, img_h = frame.size
        frame = frame.crop([self._skip_padding[0], self._skip_padding[2], img_w-self._skip_padding[1], img_h-self._skip_padding[3]])
        img_w, img_h = frame.size
        # get rescaled w and h as given ratio
        crop_h = img_h
        crop_w = int(self._crop_image_ratio * crop_h)
        # determine random crop image's center
        lp = random.randint(0, img_w-crop_w)
        # crop coord: top left + bottom right 
        crop_coords = [lp, 0, lp+crop_w, crop_h]
        orig_crop_coords = [crop_coords[0]+self._skip_padding[0],
                            crop_coords[1]+self._skip_padding[2], 
                            crop_coords[2]+self._skip_padding[0],
                            crop_coords[3]+self._skip_padding[2]]
        return frame.crop(crop_coords), orig_crop_coords
    
    def __getitem__(self, idx):
        image_name = self._img_files[idx]
        assert os.path.exists(image_name), "Image file not found!"
        # load image
        img = Image.open(image_name)
        img = img.convert("RGB")
        label = self._img_labels[idx]
        label_id = MAYO_LABELS.index(label)
        # crop image
        img, crop_coords = self.crop(img)
        # get the identical random seed for both image and mask
        seed = random.randint(0, 2147483647)
        if self._transform:
            # state = torch.get_rng_state()
            random.seed(seed)
            torch.manual_seed(seed)
            img = self._transform(img) 
        mask = []
        mask_exist = 0
        if self._mask:
            if self._mask_coord[idx][2]-self._mask_coord[idx][0]+self._mask_coord[idx][3]-self._mask_coord[idx][1]>40:
                mask_exist = 1
            mask = get_image_mask(image_name, dataset="MAYO", mask_coord=self._mask_coord[idx])
            # crop mask as croping frame from input image
            mask = mask[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
            mask = dilute_mask(mask, dilute_distance=self._mask_dilute)
            # assign class label
            mask = Image.fromarray(mask)
            if self._mask_transform:
                random.seed(seed)
                torch.manual_seed(seed)
                # torch.set_rng_state(state)
                mask = self._mask_transform(mask)
                mask = mask.type(torch.float)
                # mask = mask * label_id  # normal case is identical to backaground
        return {"image": img, "label": label_id, "mask": mask, "mask_exist": mask_exist}


def prepare_data(config):
    """
    config: 
        image_size: size of input images
        train: train img file list
        test: test img file list
        dataset: name of dataset to use: BUSI
    """
    data_transforms = {
        'train_image': transforms.Compose([   
            CustomRandomRotate([90, 180, 270]),
            transforms.RandomResizedCrop(config["image_size"], scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.12, 0.12, 0.08, 0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'train_mask': transforms.Compose([   
            CustomRandomRotate([90, 180, 270]),
            transforms.RandomResizedCrop(config["image_size"], scale=(0.75, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(), # mask only has binary values 0 or 1
        ]), 
        'test_image': transforms.Compose([
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test_mask': transforms.Compose([
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
        ]), 
    }
    mask = config.get("mask", None)
    if config["dataset"] == "BUSI":
        image_datasets = {x: BUSI_dataset(config[x], 
                                          transform=data_transforms[x+"_image"], 
                                          mask=mask, 
                                          mask_transform=data_transforms[x+"_mask"], 
                                          mask_dilute=config.get("mask_dilute", 0)) for x in ["train", "test"]}
    elif config["dataset"] == "MAYO":
        image_datasets = {x: MAYO_dataset(config[x], 
                                          crop_image_ratio=BUSI_IMAGE_RATIO,
                                          skip_padding=MAYO_VIDEO_SKIP_PADDING,
                                          transform=data_transforms[x+"_image"],
                                          mask=mask,
                                          mask_transform=data_transforms[x+"_mask"], 
                                          mask_dilute=config.get("mask_dilute", 0),
                                          ) for x in ["train", "test"]}
    else:
        print("Unknown dataset")
    data_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    return image_datasets, data_sizes



if __name__ == "__main__":
     import cv2

     config = {"image_size": 256, "train": "train_sample_v1.txt", "test": "test_sample_v1.txt", "dataset": "BUSI", "mask": True}
     ds, _ = prepare_data(config)
     batch_size = 2
     dataloader = torch.utils.data.DataLoader(ds["train"], batch_size=batch_size)
     for data in dataloader:
        imgs = data['image']
        masks = data["mask"]
        for i in range(len(imgs)):
            img = imgs[i].numpy()
            img = (img + 1.) / 2. * 255
            mask = masks[i].numpy()
            mask = mask.repeat(3, axis=0)
            mask = mask * 255
            img = np.concatenate([img, mask], axis=2)
            x = np.transpose(img, (1, 2, 0))
            x = x.astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("test/text.png", x)
            # break
            cv2.imshow("test", x)
            if cv2.waitKey(0) == ord("q"):
                exit()
