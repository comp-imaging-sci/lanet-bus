# link: https://discuss.pytorch.org/t/extracting-bounding-box-coordinates-from-mask/61179/6

import os
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops


def mask2bbox(mask):
    labels = label(mask, background=0, connectivity=2)
    props = regionprops(labels)
    return props

def busi_mask2bbox(busi_file, save_file):
    df = pd.read_csv(busi_file, sep=",", header=None)
    df.columns = ["file", "label"]
    fs = df["file"]
    lbl = df["label"]
    mask_fs = [x.replace(".png", "_mask.png") for x in fs]
    sf = open(save_file, "w")
    for i, mask_f in enumerate(mask_fs):
        mask = cv2.imread(mask_f)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        props = mask2bbox(mask)
        for prop in props:
            bbox = [prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]] # x, y, x, y
            bbox_str = ":".join([str(x) for x in bbox])
            sf.write(fs[i]+","+lbl[i]+","+bbox_str+"\n")
            # cv2.rectangle(mask, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
        # cv2.imshow("test", mask)
        # key = cv2.waitKey(0)
        # if key == ord("q"):
        #     break
    sf.close()


if __name__ == "__main__":
    input_file = "busi_test_binary.txt"
    save_file = "busi_test_binary_bbox.txt"
    busi_mask2bbox(input_file, save_file)
