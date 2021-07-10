import json
import os
import glob
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np


def _write_list_to_file(images, image_types, mask_boxes, save_file):
    f = open(save_file, "w")
    for i, t, b in zip(images, image_types, mask_boxes):
        coord = ":".join([str(k) for k in b])
        f.write("{},{},{}\n".format(i, t, coord))
    f.close()


def decode_video_json(json_file):
    video_info = json.load(open(json_file, "r"))
    video_info.pop("__comment__")
    formated_info = {}
    for pid in video_info:
        formated_info[pid] = {}
        for v_info in video_info[pid]:
            formated_info[pid][v_info["id"]] = v_info
    return formated_info

def parse_mayo_mask_box(patient_mask_file, box_anno_dir):
    """Parse patient annotation information to get the rough mask region coordinates"""
    df = pd.read_csv(patient_mask_file)
    df = df.fillna("")
    box_coord = {}
    masks = df["annotate"].tolist()
    for idx, pid in enumerate(df["patient"].tolist()):
        if masks[idx]:
            pid_masks = masks[idx].split(":")
            pbox = []
            # read all mask images and get boxes
            for pid_mask in pid_masks:
                mask_file = os.path.join(box_anno_dir, "{}_{} annotated.png_bbox.txt".format(pid, pid_mask))
                with open(mask_file, "r") as f:
                    box_str = f.readline().strip()
                    box_str_list = box_str.split(",")
                    box = [int(c) for c in box_str_list] 
                    pbox.append(box)
                f.close()
            # merge box to get the union
            pbox = np.array(pbox)
            union_xl = pbox[:, 0].min()
            union_yl = pbox[:, 1].min()
            union_xr = pbox[:, 2].max()
            union_yr = pbox[:, 3].max()
            box_coord[pid] = [union_xl, union_yl, union_xr, union_yr]
        else:
            box_coord[pid] = [0, 0, 854, 500]
    return box_coord

def generate_dataset_files(image_dir, save_dir, patient_anno_file, video_anno_file, test_ratio=0.15, seed=None, skip_keywords_list=None, only_keep_conf=True, only_keep_square_image=True, mask_annotate_dir=None):
    """
    image_dir: path to save the images
    patient_anno_file: patient type file with patient ID and corresponding class labels. Header: patient, type
    test_size: the ratio of the testing images w.r.t the total dataset
    seed: seed to randomly select the images from the whole dataset
    video_anno_file: file storing video information which is important to decide whether a certain frame would be kept for training
    only_keep_conf: when processing video data, we only keep the frames with conf as 1
    only_keep_square_image: If true, keep image when the scanning imaging region is square rather than sector
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_file = os.path.join(save_dir, "mayo_train_mask.txt")
    test_file = os.path.join(save_dir, "mayo_test_mask.txt")
    # get all images
    images = glob.glob(image_dir+"/**/*.png", recursive=True)
    print("Total images: {}".format(len(images)))
    valid_images = []
    valid_image_types = []
    valid_mask_box = []
    # get annotation information
    df = pd.read_csv(patient_anno_file, dtype=str)
    info = {i:j for i, j in zip(df["patient"].tolist(), df["type"].tolist())}
    # get skip words list
    if skip_keywords_list is None:
        skip_keywords_list = []
    # get video info
    video_info = decode_video_json(video_anno_file)
    # get mask info
    mask_info = parse_mayo_mask_box(patient_anno_file, mask_annotate_dir)
    for image in images:
        # get image patient id and disease type
        filename = os.path.basename(image)
        patient_id = re.search("([0-9\-]+)_IM.*\.png", filename).group(1)
        patient_type = info[patient_id]
        box = mask_info[patient_id]
        if skip_keywords_list:
            skip = False
            for keyword in skip_keywords_list:
                # filename containing the skip keyword
                if re.search(keyword, filename): 
                    skip = True
                    break 
        else:
            skip = False 
        # check if the frames is kept when it is from a video
        if re.search("video", filename):
            match = re.search(".+_(IM\d+)\s+video_(\d+)\.png", filename)
            video_id = match.group(1)
            frame_id = match.group(2)
            # if video confidence is 0, skip 
            # print(video_id, patient_id)
            cur_video_info = video_info[patient_id][video_id]
            if only_keep_conf:
                if not cur_video_info["conf"]:
                    skip = True 
            # if video imaging is sector, skip 
            if only_keep_square_image:
                if not cur_video_info["shape"] == "square":
                    skip = True
            # if video is not in ROI region, skip
            if cur_video_info["start"] < int(frame_id) < cur_video_info["end"]:
                skip = True
        if not skip: 
            valid_images.append(image)
            valid_image_types.append(patient_type)
            valid_mask_box.append(box)
    # split image to train and test 
    train_images, test_images, train_type, test_type, train_box, test_box = train_test_split(valid_images, valid_image_types, valid_mask_box, test_size=test_ratio, random_state=seed)
    _write_list_to_file(train_images, train_type, train_box, train_file)
    _write_list_to_file(test_images, test_type, test_box, test_file)
    print("Image names are written done. Total {} training image, {} testing images".format(len(train_images), len(test_images)))

def json2csv(json_file, csv_file):
    data = decode_video_json(json_file)
    data_dict = defaultdict(list)
    for pid in data:
        for vid in data[pid]:
            v_info = data[pid][vid]
            data_dict["Patient ID"].append(pid)
            data_dict["Video ID"].append(vid)
            v_info.pop("id")
            for k, v in v_info.items():
                data_dict[k].append(v)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    image_dir = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/images"
    mask_anno_dir = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/annotate"
    seed = 1 
    save_dir = "/Users/zongfan/Downloads/cancer_ultrasound_project/breast_cancer_project/data"
    anno_file = "mayo_patient_info.csv"
    skip_keywords_list = ["color", "annotated"]
    video_json = "video_info.json"
    only_keep_square_image = True
    only_keep_conf = True
    generate_dataset_files(image_dir, save_dir, anno_file, video_json, seed=seed, skip_keywords_list=skip_keywords_list,
    only_keep_conf=only_keep_conf, only_keep_square_image=only_keep_square_image, mask_annotate_dir=mask_anno_dir)
    # video_csv = "video_info.csv"
    # json2csv(video_json, video_csv)
