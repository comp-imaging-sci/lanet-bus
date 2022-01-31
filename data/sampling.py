import os
import pandas as pd
import cv2
#import shutil
import glob

def extract_image_from_file(input_file, save_dir, num_images=500, image_size=512):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(input_file, sep=",", header=None)
    df.columns = ["img", "label"]
    labels = set(list(df['label']))
    for label in labels:
        cate_dir = os.path.join(save_dir, label)
        os.makedirs(cate_dir, exist_ok=True)
        images = df[df["img"] == label]
        images = df["img"]
        print(len(images))
        #if label in ["COVID-19", "normal"]:
        #    continue
        #if label == "COVID-19":
        #    select_count = 500
        #else:
        #    select_count = num_images
        select_count = num_images
        count = 1
        for sample in images:
            sample_name = os.path.basename(sample)
            out = os.path.join(cate_dir, sample_name)
            #shutil.copy(inp, out)
            #assert os.path.exists(inp), "{}".format(inp)
            try:
                im = cv2.imread(sample)
                im = cv2.resize(im, (image_size, image_size))
                cv2.imwrite(out, im)
                count += 1
            except:
                print("invalid image")
            if count > select_count:
                break
    print("copy done!")

def extract_image_from_dir(image_dir, save_dir, num_images=200, image_size=512):
    images = glob.glob(image_dir+"/**/*.png", recursive=True)
    print(len(images))
    count = 1
    for image in images:
        items = image.split("/")
        label = items[-2]
        fname = items[-1]
        outdir = os.path.join(save_dir, label)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        out = os.path.join(outdir, fname)
        try:
            im = cv2.imread(image)
            im = cv2.resize(im, (image_size, image_size))
            cv2.imwrite(out, im)
            count += 1
        except:
            print("invalid image")
        if count > num_images:
            break
    print("extract done")

train_txt = "busi_train_sample.txt"
test_txt = "busi_test_sample.txt"
image_dir = "/Users/zongfan/Projects/data/breas_cancer_us/Dataset_BUSI_with_GT"
save_dir = "/Users/zongfan/Downloads/BUSI"
extract_image_from_dir(image_dir, save_dir, num_images=1e4)
