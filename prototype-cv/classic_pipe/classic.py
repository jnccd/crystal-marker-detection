import math
import sys
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

# Skip the larges images
for img in pyr_imgs[3:]:
    img_h, img_w = img.shape[:2]
    cv2.imwrite(str(root_dir / f'{pyr_imgs.index(img)}_pyr_img.png'), img)
    
    block_size = 50#int(img_w/600*50)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    img_t = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,block_size,3)
    # cv2.imshow('pyr',img_t)
    # cv2.waitKey(0)
    
    promising_points = []
    
    # --------------------------------------------------------------------------------------------------------------------------------------
    
    # # Pixel neighborhood
    img_draw = np.copy(img)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    window_size_div2 = 32
    total_pixs_in_window = (window_size_div2*2)**2
    for x in range(window_size_div2, img_w-window_size_div2, 2):
        for y in range(window_size_div2, img_h-window_size_div2, 2):
            window = img_t[y-window_size_div2:y+window_size_div2, x-window_size_div2:x+window_size_div2]
            #print(window.shape)
            white_pixs = np.sum(window == 255)
            white_ratio = white_pixs / total_pixs_in_window
            #print('white_ratio', white_ratio)
            white_ratio = 255 - abs(white_ratio * 255 - 150)
            if white_ratio > 220:
                cv2.rectangle(img_draw, (x,y,1,1), (0,0,white_ratio))
                promising_points.append((x,y))
    cv2.imwrite(str(root_dir / f'{pyr_imgs.index(img)}_pyr_img_pixel_neighbors.png'),img_draw)
    cv2.waitKey(0)
    # Too many candidates!
    
    # --------------------------------------------------------------------------------------------------------------------------------------
    
    # # Hough Transform
    # img_edges = cv2.Canny(img, 50, 300, None, 5)
    # # cv2.imshow('pyr',img_edges)
    # # cv2.waitKey(0)
    # linesP = cv2.HoughLinesP(img_edges, 0.5, np.pi / 360, 10, None, 3, 2)
    # #print(linesP)
    # linesP_atan2 = [(l, math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in linesP]
    # img_draw = np.copy(img)
    # if linesP is not None:
    #     print(len(linesP))
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(img_draw, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    # # cv2.imshow('pyr',img_draw)
    # # cv2.waitKey(0)
    
    # img_draw = np.copy(img)
    # img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    # window_size_div2 = 32
    # for x in range(window_size_div2, img_w-window_size_div2, 2):
    #     for y in range(window_size_div2, img_h-window_size_div2, 2):
    #         y_min = y-window_size_div2
    #         y_max = y+window_size_div2
    #         x_min = x-window_size_div2
    #         x_max = x+window_size_div2
            
    #         window_lines = []
    #         for l, a in linesP_atan2:
    #             if l[0][0] > x_min and l[0][0] > x_max and l[0][1] > y_min and l[0][1] > y_max:
    #                 window_lines.append((l, a))
                    
    #         #window_lines = filter(lambda l: l[0][0] > x_min and l[0][0] > x_max and l[0][1] > y_min and l[0][1] > y_max, linesP_atan2)
    #         #window_lines_atan2 = [x[1] * 360 / math.pi for x in window_lines]
    #         #print(window_lines_atan2)
    #         #hist, bin_edges = np.histogram(window_lines_atan2, bins=range(18))
    #         #if len(window_lines_atan2) > 0:
    #         #    print(x, y, hist, 'mm', np.max(window_lines_atan2), np.min(window_lines_atan2))
            
    #         hist_min = -math.pi / 2
    #         hist_max = math.pi / 2
    #         hist_range = hist_max - hist_min
    #         num_bins = 20
    #         bin_borders = []
    #         bins = np.zeros((num_bins))
    #         for i in range(num_bins):
    #             bin_borders.append((i / num_bins) * hist_range + hist_min)
    #         bin_borders.append(hist_max)
    #         #print('borders',bin_borders)
            
    #         window_lines_atan2 = [x[1] for x in window_lines]
    #         #print(min(window_lines_atan2))
    #         #print(max(window_lines_atan2))
    #         for wline in window_lines:
    #             for i in range(num_bins):
    #                 if wline[1] <= bin_borders[i]:
    #                     l = wline[0]
    #                     bins[i] += math.sqrt((l[0][3] - l[0][1])**2 + (l[0][2] - l[0][0])**2)
    #                     break
            
    #         #print('Bins in:', x_min, y_min, x_max, y_max, 'pyr_index:', pyr_imgs.index(img), 'bins:', bins)
            
    #         sorted_bins = np.sort(bins, axis=None)
    #         first_two_peaks_diff = abs(sorted_bins[-1] - sorted_bins[-2])
    #         first_peak_height = sorted_bins[-1]
    #         third_peak_height = sorted_bins[-3]
            
    #         markeryness = first_peak_height / 4 - first_two_peaks_diff - third_peak_height
    #         print('markeryness', markeryness, x_min, y_min, 'single vals', first_peak_height, third_peak_height, first_two_peaks_diff)
    #         cv2.rectangle(img_draw, (x,y,2,2), (0,0,int(max(0, markeryness))))
            
    #         # print([x[1] for x in window_lines])
    #         # sys.exit(0)
    # cv2.imwrite(str(root_dir / f'{pyr_imgs.index(img)}_pyr_img_hough_markeryness.png'), img_draw)
    
    # --------------------------------------------------------------------------------------------------------------------------------------