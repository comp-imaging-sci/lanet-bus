import torch
from torchvision import transforms, datasets
from PIL import Image
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import glob, os 
import pandas as pd 
# from skimage import io 
import numpy as np 
import re
import random


BUSI_LABELS = ["normal", "malignant", "benign"] # breast cancer label

class BUSI_dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file, sep=",", header=None)
        df.columns = ["img", "label"]
        self._img_files = df["img"].tolist()
        self._img_labels = df["label"].tolist()
        self._transform = transform

    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        image_name = self._img_files[idx]
        assert os.path.exists(image_name), "Image file not found!"
        img = Image.open(image_name)
        img = img.convert("RGB")
        # print("img size: ", img.size)
        label = self._img_labels[idx]
        label_id = BUSI_LABELS.index(label)
        #onehot_id = torch.nn.functional.one_hot(torch.Tensor(label_id), len(LABELS))
        if self._transform:
            img = self._transform(img)
        return img, label_id
    

def prepare_data(config):
    """
    config: 
        input_size,
        train: train img file list
        test: test img file list
        dataset: name of dataset to use: BUSI
    """
    data_transforms = {
        'train': transforms.Compose([   
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            # transforms.Resize(480),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(config["input_size"], scale=(0.85, 1.0), ratio=(0.85, 1.15)),
            transforms.ColorJitter(0.12, 0.12, 0.08, 0.08),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            transforms.Resize(config["input_size"]),
            transforms.CenterCrop(config["input_size"]),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    if config["dataset"] == "BUSI":
        image_datasets = {x: BUSI_dataset(config[x], data_transforms[x]) for x in ["train", "test"]}
    else:
        print("Unknown dataset")
    # class_names = image_datasets["train"].classes 
    data_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    return image_datasets, data_sizes


def generate_image_list(img_dir, save_dir, test_sample_size=40):
    global BUSI_LABELS
    image_list = glob.glob(img_dir+"/**/*.png", recursive=True)
    train_sample_file = os.path.join(save_dir, "train_sample.txt")
    test_sample_file = os.path.join(save_dir, "test_sample.txt")
    random.shuffle(image_list)
    train_f = open(train_sample_file, "w")
    test_f = open(test_sample_file, "w")
    counter = {x: 0 for x in BUSI_LABELS}
    for img in image_list:
        class_name = os.path.basename(os.path.dirname(img))
        # ignore mask images
        if re.search("mask", img):
            continue
        if counter[class_name] > test_sample_size:
            write_f = train_f 
        else:
            write_f = test_f 
        write_f.write("{},{}\n".format(img, class_name))
        counter[class_name] += 1
    train_f.close()
    test_f.close()


if __name__ == "__main__":
    import cv2
    image_dir = "/Users/zongfan/Projects/data/chest_xray"
    image_dir = "/Users/zongfan/Projects/data/Dataset_BUSI_with_GT"
    # config = {"input_size": 448}
    # ds, _ = prepare_data(image_dir, config)
    # batch_size = 2
    # dataloader = torch.utils.data.DataLoader(ds["train"], batch_size=batch_size)
    # for imgs, label in dataloader:
    #     print(imgs.shape)
    #     img_shape = imgs.shape
    #     imgs = imgs.view(img_shape[0]*img_shape[1], img_shape[2], img_shape[3], img_shape[4])
    #     print(imgs.shape)
    #     for img in imgs:
    #         x = np.transpose(img.numpy(), (1, 2, 0))
    #         x = (x + 1.) / 2. * 255
    #         x = x.astype(np.uint8)
    #         x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    #         cv2.imshow("test", x)
    #         if cv2.waitKey(0) == ord("q"):
    #             exit()
    generate_image_list(image_dir, ".", test_sample_size=40)

    # test_image = "/Users/zongfan/Projects/data/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg"
    # img = Image.open(test_image)
    # print(img.size)
    # img = img.convert("RGB")
    # img = transforms.ToTensor()(img)
    # print(img.shape)
    # p = PatchGenerator(n=16, patch_size=448, style="grid")
    # res = p(img)
    # print(len(res), res[0])
    # for im in res:
    #     im.show()
