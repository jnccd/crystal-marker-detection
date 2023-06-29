import math
import cv2
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent
test_img_path = root_dir / 'DSC_3741.JPG'

test_img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)

# Gen img pyramid
pyr_imgs = [test_img]
while pyr_imgs[-1].shape[0] > 64:
    pyr_h, pyr_w = pyr_imgs[-1].shape[:2]
    pyr_imgs.append(cv2.pyrDown(pyr_imgs[-1], dstsize=(int(pyr_w/2), int(pyr_h/2))))
print(len(pyr_imgs))

# Skip the 2 larges images
for img in pyr_imgs[4:]:
    img_h, img_w = img.shape[:2]
    block_size = 50#int(img_w/600*50)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    img_t = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,block_size,3)
    cv2.imshow('pyr',img_t)
    cv2.waitKey(0)
    
    # # Pixel neighborhood
    # img_draw = np.copy(img)
    # img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    # window_size_div2 = 32
    # total_pixs_in_window = (window_size_div2*2)**2
    # for x in range(window_size_div2, img_w-window_size_div2, 2):
    #     for y in range(window_size_div2, img_h-window_size_div2, 2):
    #         window = img_t[y-window_size_div2:y+window_size_div2, x-window_size_div2:x+window_size_div2]
    #         #print(window.shape)
    #         white_pixs = np.sum(window == 255)
    #         white_ratio = white_pixs / total_pixs_in_window
    #         #print('white_ratio', white_ratio)
    #         white_ratio = 255 - abs(white_ratio * 255 - 185)
    #         if white_ratio > 230:
    #             cv2.rectangle(img_draw, (x,y,1,1), (0,0,white_ratio))
    # cv2.imshow('pyr',img_draw)
    # cv2.waitKey(0)
    
    # Hough Transform
    img_edges = cv2.Canny(img, 50, 300, None, 5)
    cv2.imshow('pyr',img_edges)
    cv2.waitKey(0)
    linesP = cv2.HoughLinesP(img_edges, 0.5, np.pi / 360, 10, None, 3, 2)
    linesP_atan2 = [(l, math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in linesP]
    img_draw = np.copy(img)
    if linesP is not None:
        print(len(linesP))
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_draw, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow('pyr',img_draw)
    cv2.waitKey(0)
    
    window_size_div2 = 32
    for x in range(window_size_div2, img_w-window_size_div2, 2):
        for y in range(window_size_div2, img_h-window_size_div2, 2):
            y_min = y-window_size_div2
            y_max = y+window_size_div2
            x_min = x-window_size_div2
            x_max = x+window_size_div2
            
            window_lines = []
            for l, a in linesP_atan2:
                if l[0][0] > x_min and l[0][0] > x_max and l[0][1] > y_min and l[0][1] > y_max:
                    window_lines.append((l, a))
            #window_lines = filter(lambda l: l[0][0] > x_min and l[0][0] > x_max and l[0][1] > y_min and l[0][1] > y_max, linesP_atan2)
            window_lines_atan2 = [x[1] * 360 / math.pi for x in window_lines]
            #print(window_lines_atan2)
            hist, bin_edges = np.histogram(window_lines_atan2, bins=range(18))
            if len(window_lines_atan2) > 0:
                print(x, y, hist, 'mm', np.max(window_lines_atan2), np.min(window_lines_atan2))
            