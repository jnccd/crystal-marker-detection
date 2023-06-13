import math
import cv2
import numpy as np
from pathlib import Path

filename = "DSC_3741.JPG"
root_dir = Path(__file__).resolve().parent
test_img_path = root_dir / filename

blur_kernel_size = 7

img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
target_hist = np.asarray([65818, 53459, 40828, 28699, 18714, 11198,  6051,  8348, 12378,  6724,  7668, 11688, 19081, 29440, 43767, 57100, 63234], dtype=np.int32)
#img = cv2.resize(img, (1920, 1080))

print('Computing gy...')
kernel = np.asarray([-255,0,255], dtype=np.float32)
gy_img = cv2.filter2D(img, -1, kernel)
gy_img = cv2.GaussianBlur(gy_img,(blur_kernel_size,blur_kernel_size),0)
cv2.imwrite(str(root_dir / (filename+'kernelled_y_img.png')), gy_img)

print('Computing gx...')
kernel = cv2.transpose(kernel)
gx_img = cv2.filter2D(img, -1, kernel)
gx_img = cv2.GaussianBlur(gx_img,(blur_kernel_size,blur_kernel_size),0)
cv2.imwrite(str(root_dir / (filename+'kernelled_x_img.png')), gx_img)

print('Computing atan array...')
rows,cols = img.shape
atan_array = np.zeros(shape=(rows,cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        ang = math.degrees(math.atan2(gy_img[i,j], gx_img[i,j]))
        atan_array[i,j] = int(ang / 5)

print('max min',np.max(atan_array),np.min(atan_array))
cv2.imwrite(str(root_dir / (filename+'atan.png')), atan_array)

stepwidth = 50

print('Computing histogram similarities...')
wh, ww = 700, 700
ih, iw = int((rows-wh) / stepwidth), int((cols-ww) / stepwidth)
out_img = np.zeros((ih, iw), dtype=np.int32)
for swi in range(0,ih):
    for swj in range(0,iw):
        
        wi = swi * stepwidth
        wj = swj * stepwidth
        
        window = atan_array[wi:wi+wh, wj:wj+ww]
        hist, bin_edges = np.histogram(window, bins=range(18))
        
        max_i = np.argmax(hist)
        rot_hist = np.roll(hist, max_i)
        
        hist_diff = np.sum(np.abs(rot_hist-target_hist))
        #print(wi, wj, rot_hist, hist_diff)
        out_img[swi, swj] = hist_diff
        
out_img = np.interp(out_img, (out_img.min(), out_img.max()), (0, 255))
cv2.imwrite(str(root_dir / (filename+'marked.png')), out_img)
