import math
import sys
import cv2
import numpy as np
from pathlib import Path

img_img_filename = 'DSC_3741.JPG'
root_dir = Path(__file__).resolve().parent
test_img_path = root_dir / img_img_filename
test_img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)

# Gen img pyramid
pyr_imgs = [test_img]
while pyr_imgs[-1].shape[0] > 128:
    pyr_h, pyr_w = pyr_imgs[-1].shape[:2]
    pyr_imgs.append(cv2.pyrDown(pyr_imgs[-1], dstsize=(int(pyr_w/2), int(pyr_h/2))))
print(len(pyr_imgs))

# Skip the larges images
for img in pyr_imgs[3:]:
    img_i = img_img_filename + '_' + str(pyr_imgs.index(img)) # Make index an identifier
    img_h, img_w = img.shape[:2]
    rows,cols = img.shape[:2]
    cv2.imwrite(str(root_dir / f'{img_i}_pyr_img.png'), img)
    
    block_size = 50#int(img_w/600*50)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    img_t = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,block_size,3)
    # cv2.imshow('pyr',img_t)
    # cv2.waitKey(0)
    
    promising_points = []
    
    # --------------------------------------------------------------------------------------------------------------------------------------
    
    # Pixel neighborhood
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
    cv2.imwrite(str(root_dir / f'{img_i}_pyr_img_pixel_neighbors.png'),img_draw)
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
            
    #         #print('Bins in:', x_min, y_min, x_max, y_max, 'pyr_index:', img_i, 'bins:', bins)
            
    #         sorted_bins = np.sort(bins, axis=None)
    #         first_two_peaks_diff = abs(sorted_bins[-1] - sorted_bins[-2])
    #         first_peak_height = sorted_bins[-1]
    #         third_peak_height = sorted_bins[-3]
            
    #         markeryness = first_peak_height / 4 - first_two_peaks_diff - third_peak_height
    #         print('markeryness', markeryness, x_min, y_min, 'single vals', first_peak_height, third_peak_height, first_two_peaks_diff)
    #         cv2.rectangle(img_draw, (x,y,2,2), (0,0,int(max(0, markeryness))))
            
    #         # print([x[1] for x in window_lines])
    #         # sys.exit(0)
    # cv2.imwrite(str(root_dir / f'{img_i}_pyr_img_hough_markeryness.png'), img_draw)
    
    # --------------------------------------------------------------------------------------------------------------------------------------
    
    # Gradient hist
    blur_kernel_size = 5
    
    print('Computing gy...')
    kernel = np.asarray([-255,0,255], dtype=np.float32)
    gy_img = cv2.filter2D(img, -1, kernel)
    gy_img = cv2.GaussianBlur(gy_img,(blur_kernel_size,blur_kernel_size),0)
    ret, gy_img = cv2.threshold(gy_img, 60, 255, cv2.THRESH_TOZERO)
    cv2.imwrite(str(root_dir / f'{img_i}_kernelled_y_img.png'), gy_img)
    gy_img = gy_img.astype(np.float32)
    gy_img = gy_img / 128 - 1

    print('Computing gx...')
    kernel = cv2.transpose(kernel)
    gx_img = cv2.filter2D(img, -1, kernel)
    gx_img = cv2.GaussianBlur(gx_img,(blur_kernel_size,blur_kernel_size),0)
    ret, gx_img = cv2.threshold(gx_img, 60, 255, cv2.THRESH_TOZERO)
    cv2.imwrite(str(root_dir / f'{img_i}_kernelled_x_img.png'), gx_img)
    gx_img = gx_img.astype(np.float32)
    gx_img = gx_img / 128 - 1
    
    #print('maxmin', np.min(gx_img), np.max(gx_img))
    
    print('Computing ang array...')
    ang_array = np.zeros(shape=(rows,cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            ang_array[i,j] = math.atan2(gy_img[i,j], gx_img[i,j])
        
    #print('maxmin', np.min(ang_array), np.max(ang_array))
            
    print('Computing mag array...')
    mag_array = np.zeros(shape=(rows,cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            mag_array[i,j] = math.sqrt(gy_img[i,j]**2 + gx_img[i,j]**2)
    
    #print(promising_points)
    markerynesses = []
    for ppoint in promising_points:
        x_min = ppoint[0] - window_size_div2
        x_max = ppoint[0] + window_size_div2
        y_min = ppoint[1] - window_size_div2
        y_max = ppoint[1] + window_size_div2
        
        window_size_div2 = 32
        img_window = img[ppoint[1] - window_size_div2:ppoint[1] + window_size_div2, ppoint[0] - window_size_div2: ppoint[0] + window_size_div2]
        gy_img_window = gy_img[ppoint[1] - window_size_div2:ppoint[1] + window_size_div2, ppoint[0] - window_size_div2: ppoint[0] + window_size_div2]
        gx_img_window = gx_img[ppoint[1] - window_size_div2:ppoint[1] + window_size_div2, ppoint[0] - window_size_div2: ppoint[0] + window_size_div2]
        ang_array_window = ang_array[ppoint[1] - window_size_div2:ppoint[1] + window_size_div2, ppoint[0] - window_size_div2: ppoint[0] + window_size_div2]
        mag_array_window = mag_array[ppoint[1] - window_size_div2:ppoint[1] + window_size_div2, ppoint[0] - window_size_div2: ppoint[0] + window_size_div2]
        
        # hist_min = -math.pi / 2
        # hist_max = math.pi / 2
        # hist_range = hist_max - hist_min
        # num_bins = 20
        # bin_borders = []
        # bins = np.zeros((num_bins))
        # for i in range(num_bins):
        #     bin_borders.append((i / num_bins) * hist_range + hist_min)
        # bin_borders.append(hist_max)
        # #print('borders',bin_borders)
        # print(np.min(ang_array_window))
        # print(np.max(ang_array_window))
        # for x in range(img_window.shape[1]):
        #     for y in range(img_window.shape[0]):
        #         for i in range(num_bins):
        #             if ang_array_window[x, y] <= bin_borders[i]:
        #                 bins[i] += mag_array_window[x, y]
        #                 break
        hist, bin_edges = np.histogram(ang_array_window, bins=20, range=(-math.pi, math.pi))#, weights=mag_array_window)
        #print(f'img_id {img_i}, ppoint {ppoint}, hist {hist}, bin_edges {bin_edges}, bbox, {(x_min, y_min, x_max, y_max)}')
        
        peaks = []
        hist[2] = 0
        for i in range(len(hist)):
            lower_bound = i-2 if i-2 >= 0 else 0
            higher_bound = i+2 if i+2 < len(hist) else len(hist) - 1
            hist_window = hist[lower_bound:higher_bound]
            #print(i, hist_window, hist)
            if max(hist_window) != hist[i]:
                continue
            peaks.append((i, hist[i]))
        peaks.sort(key=lambda x: -x[1])
        
        peak_heights = [x[1] for x in peaks]
        markeryness = sum(peak_heights[:2]) - sum(peak_heights[2:]) * 4 - abs(peak_heights[0] - peak_heights[1] * 2) * 2
        #print(peak_heights[:2], peak_heights[2:], peak_heights)
        
        #print(ppoint, peak_heights)
        
        markerynesses.append((ppoint, markeryness))
        
    split_markerynesses = [x[1] for x in markerynesses]
    max_markeryness = max(split_markerynesses)
    min_markeryness = min(split_markerynesses)
    mm_diff_markeryness = max_markeryness - min_markeryness
    
    #print(min_markeryness, max_markeryness)
    
    img_draw = np.copy(img)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    for m in markerynesses:
        cv2.rectangle(img_draw, (m[0][0],m[0][1],1,1), (0,0,((m[1] - min_markeryness) / mm_diff_markeryness) * 255 ))
    cv2.imwrite(str(root_dir / f'{img_i}_markerynesses.png'), img_draw)