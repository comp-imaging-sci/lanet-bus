import cv2
import numpy as np
import glob
import os
import re

MIN_MATCH_SCORE = 3e6

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def search_template(target, template):
    """Search template image on target and return matched coordinates"""
    # Load template, convert to grayscale, perform canny edge detection
    # temp = cv2.imread(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 100, 200)
    (tH, tW) = template.shape[:2]

    # Load original image, convert to grayscale
    # frame = cv2.imread(target)
    # tarH, tarW = target.shape[:2]
    # frame = frame[50:tarH-50, 50:tarW-50]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    best1, best2 = None, None

    # Dynamically rescale image for better template matching
    for scale in np.linspace(0.1, 2, 20)[::-1]:
        # Resize image to scale and keep track of ratio
        resized = maintain_aspect_ratio_resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # Stop if template image size is larger than resized image
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # Detect edges in resized image and apply template matching
        canny = cv2.Canny(resized, 100, 200)
        detected = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF)

        # get max value
        (_, max_val, _, max_loc) = cv2.minMaxLoc(detected)
        # get second max value
        # print(max_loc, detected.shape)
        detected_clone = detected.copy()
        detected_clone[max_loc[1]-tH//2:max_loc[1]+tH,max_loc[0]-tW//2:max_loc[0]+tW] = 0
        # print(detected_clone[max_loc[1],max_loc[0]])
        (_, sub_max_val, _, sub_max_loc) = cv2.minMaxLoc(detected_clone) 
        # print(max_loc, sub_max_loc)
        # Keep track of correlation value
        # Higher correlation means better match
        if best1 is None or max_val > best1[0]:
            best1 = (max_val, max_loc, r)
        if best2 is None or sub_max_val > best2[0]:
            best2 = (sub_max_val, sub_max_loc, r)

    # # Compute coordinates of bounding box
    match = [best1, best2]
    box_coords = []
    for coord in match:
        (max_v, loc, r) = coord
        # print(coord)
        (start_x, start_y) = (int(loc[0] * r), int(loc[1] * r))
        (end_x, end_y) = (int((loc[0] + tW) * r), int((loc[1] + tH) * r))
        # check if max_v is above matching threshold
        if max_v >= MIN_MATCH_SCORE:
            box_coords.append((start_x, start_y, end_x, end_y))

    # # Draw bounding box on ROI
    # for box in box_coords:
    #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
    #     cv2.imwrite("test/test.png", frame)

    # cv2.imshow('detected', original_image)
    # cv2.imwrite('detected.png', original_image)
    # cv2.waitKey(0)
    return box_coords

def get_lesion_location(target, temp_images, strip_padding=[50, 50, 50, 50], draw_box=False):
    """Get location of breast cancer lesion in the tissue
    strip_padding: pixels to remove on each side of the image, [left, right, top, bottom]
    """
    target_frame = cv2.imread(target)
    tH, tW = target_frame.shape[:2]
    # crop image
    target_frame = target_frame[strip_padding[2]:tH-strip_padding[3], strip_padding[0]:tW-strip_padding[1]]
    colors = [(0, 255, 0), (0, 0, 255)]
    box_pt = []
    for i, temp in enumerate(temp_images):
        template_frame = cv2.imread(temp)
        # get coordinates of given template
        temp_box = search_template(target_frame, template_frame)
        # draw bot around template marks
        # for box in temp_box:
        #     cv2.rectangle(target_frame, (box[0], box[1]), (box[2], box[3]), colors[i], 2)
        # get mark center
        if temp_box:
            for box in temp_box:
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                box_pt.append([center_x, center_y])
    # get coordinates of top left and bottom right points
    if box_pt:
        box_pt = np.asarray(box_pt)
        # if 4 points are detected, use bounding box of the 4 points
        # if box_pt.shape[0] == 4:
        # top left x, y
        lx, ly = np.min(box_pt, axis=0)
        rx, ry = np.max(box_pt, axis=0)
        # if only 2 points are detected, crop a square window at the center of the 2 points
        if box_pt.shape[0] == 2: 
            window_w = rx - lx
            window_h = ry - ly
            window_size = max(window_h, window_w) 
            window_cent_x = (lx + rx) // 2
            window_cent_y = (ly + ry) // 2
            lx = max(window_cent_x - window_size // 2, 0)
            ly = max(window_cent_y - window_size // 2, 0)
            rx = min(window_cent_x + window_size // 2, tW)
            ry = min(window_cent_y + window_size // 2, tH)
        if draw_box:
            cv2.rectangle(target_frame, (lx, ly), (rx, ry), (0, 255, 0), 2)
            # cv2.imwrite("test/test.png", target_frame)
        # revert to original coord
        lx += strip_padding[0]
        ly += strip_padding[2]
        rx += strip_padding[0]
        ry += strip_padding[2]
        return (lx, ly, rx, ry), target_frame
    else:
        return None, None


def run(data_dir, save_dir, temp_images, draw_box=False):
    """Get lesion location from all the annotated images in given data directory"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get all annotated images
    images = glob.glob(data_dir+"/**/*annotated.png", recursive=True)
    print(len(images))
    for image in images:
        loc, frame = get_lesion_location(image, temp_images, draw_box=draw_box)
        if not loc: 
            continue
        img_name = os.path.basename(image)
        loc_save_name = os.path.join(save_dir, img_name+"_bbox.txt")
        img_save_name = os.path.join(save_dir, img_name+"_bbox.png")
        with open(loc_save_name, "w") as f:
            f.write(",".join(map(str, loc)))
        f.close()
        if draw_box:
            cv2.imwrite(img_save_name, frame)
        print("{} processed!".format(img_name))

if __name__ == "__main__":
    target ="/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/images/018_IM00003 annotated.png" 
    temp_images = ['temp1.png', 'temp2.png']
    # coord, frame = get_lesion_location(target, temp_images, draw_box=True)
    data_dir = "/shared/radon/TOP/breast_cancer_us/MAYO/101-150/images"
    anno_save_dir = "/shared/radon/TOP/breast_cancer_us/MAYO/101-150/annotate"
    # data_dir = "/Users/zongfan/Downloads/annotate"
    # anno_save_dir = "/Users/zongfan/Downloads/annotate"
    # image_save_dir = anno_save_dir
    run(data_dir, anno_save_dir, temp_images, draw_box=True)