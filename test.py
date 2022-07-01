import cv2
from cv2 import THRESH_BINARY
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib 

file = "/Users/zongfan/Downloads/mag_sos_wn.nii"
# frame = cv2.imread("/Users/zongfan/Downloads/r2t_test.png")

data = nib.load(file)
data_np = data.get_fdata()
print(data_np.shape)
frame = data_np[:, :, 143]
# for i in range(data_np.shape[-1]):
#     f = data_np[:, :, i]
#     # f[f>6000] = 6000
#     f = cv2.normalize(f, None,  0, 255, cv2.NORM_MINMAX)
#     f = f.astype(np.uint8)
#     print("idx:", i)
#     print(np.max(f), np.min(f)) 
#     cv2.imshow("test", f)
#     key = cv2.waitKey(0)
#     if key == ord("q"):
#         break

fm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
fm_cp = fm.copy()

# plt.imshow(fm_cp, cmap="gray")
# plt.show()
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(np.max(fm), np.min(fm))
_, binary = cv2.threshold(fm, 60, 255, cv2.THRESH_BINARY_INV)

# show_img = np.hstack([fm, binary])
# plt.imshow(show_img, cmap="gray")
# plt.show()


contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt_area = []
fil_cnts = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    cnt_area.append(area)
    if area < 100:
        fil_cnts.append(cnt)
cnt_area = np.array(cnt_area)
print(cnt_area)
# fil_cnts = contours[cnt_area > 1000]
# draw all contours
vis = cv2.cvtColor(fm_cp.astype(np.uint8), cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(vis, fil_cnts, -1, (0, 255, 0), 4)

# img = np.hstack([img, binary_img])

plt.imshow(img, cmap="gray")
plt.show()