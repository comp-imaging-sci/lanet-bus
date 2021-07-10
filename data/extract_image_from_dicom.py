from pydicom import dcmread
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import os 
import glob 
import re


def extract_image_from_dicom(fpath, save_dir, image_size=None):
    """Extract and save images from a dicom file"""
    ds = dcmread(fpath)
    data = ds.pixel_array
    # print(data.shape)
    dshape = data.shape
    # summarize data value range
    # plt.hist(data.flatten(), bins=100, color='c')
    # print("max value: {}, min value: {}".format(np.max(data), np.min(data)))
    # plt.show()
    # extract images and save path 
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    elif isinstance(image_size, list or tuple):
        assert len(image_size) == 2, "Input image size must be 2-integer list/tuple, or an integer"
    img_name = os.path.basename(fpath)
    pname = os.path.basename(os.path.dirname(fpath))
    if len(dshape) == 3 and re.search("color", img_name):
        frame = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if image_size is not None:
            frame = cv2.resize(frame, image_size)
        save_name = os.path.join(save_dir, "{}_{}.png".format(pname, img_name))
        cv2.imwrite(save_name, frame)
        return 
    if len(dshape) == 2:
        frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        if image_size is not None:
            frame = cv2.resize(frame, image_size)
        save_name = os.path.join(save_dir, "{}_{}.png".format(pname, img_name))
        cv2.imwrite(save_name, frame)
        return
    if len(dshape) == 3 and re.search('video', img_name):
        for i in range(dshape[0]):
            frame = data[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if image_size is not None:
                frame = cv2.resize(frame, image_size)
            save_name = os.path.join(save_dir, "{}_{}_{}.png".format(pname, img_name, i))
            # cv2.imshow("img", frame)
            # if cv2.waitKey() == ord("q"):
            #     break
            cv2.imwrite(save_name, frame)
        return


def run(data_dir, save_dir, image_size=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images = glob.glob(data_dir+"/**/IM*", recursive=True)
    for image in images:
        extract_image_from_dicom(image, save_dir, image_size)
        image_name = os.path.basename(image)
        print("{} processed!".format(image_name))

# get valid frames containing lesion in the video


if __name__ == '__main__':
    # fpath = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/001-050/001/IM00028"
    # fpath = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/001-050/001/IM00033 annotated"
    # fpath = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/001-050/001/IM00027 video"
    # fpath = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/001-050/009/IM00009 color" 
    # save_dir = "test"
    # save_dir = "/Users/zongfan/Downloads/dicom_test"
    data_dir = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/001-050"
    save_dir = "/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/images"
    # extract_image_from_dicom(fpath, save_dir)
    run(data_dir, save_dir)



