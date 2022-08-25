import random
import os 
import pandas as pd


def random_remove_mask(input_file, output_file, keep_ratio=0.25):
    df = pd.read_csv(input_file, header=None, sep=",")
    df.columns = ["img", "label", "bbox"]
    of = open(output_file, "w")
    for i in range(len(df.index)):
        img = df["img"][i]
        label = df["label"][i]
        if random.random() <= keep_ratio:  
            bbox = df["bbox"][i]
        else:
            bbox = "0:0:0:0"
        of.write("{},{},{}\n".format(img, label, bbox))
    of.close()


if __name__ == "__main__":
    input_file = "busi_train_binary_bbox.txt"
    keep_ratio = 0.75
    output_file = input_file.replace(".txt", "_{}.txt".format(keep_ratio))
    random_remove_mask(input_file, output_file, keep_ratio)
