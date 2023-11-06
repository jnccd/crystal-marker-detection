import math
import random
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
for img in pyr_imgs[2:]:
    img_i = img_img_filename + '_' + str(pyr_imgs.index(img)) # Make index an identifier
    img_h, img_w = img.shape[:2]
    rows,cols = img.shape[:2]
    
    # block_size = 50#int(img_w/600*50)
    # block_size = block_size if block_size % 2 == 1 else block_size + 1
    # img_t = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,block_size,3)
    # cv2.imshow('pyr',img_t)
    # cv2.waitKey(0)
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(img, threshold1=10, threshold2=50)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # to demonstrate the impact of contour approximation, let's loop
    # over a number of epsilon sizes
    for eps in [0.01]:#np.linspace(0.001, 0.05, 10):
        # draw the approximated contour on the image
        output = img.copy()
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        
        # approximate the contour
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            print(approx.shape)
            # if approx.shape[0] != 4:
            #     continue
            cv2.drawContours(output, [approx], -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
        
        text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
        # cv2.putText(output, text, (0, 0 - 15), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.9, (0, 255, 0), 2)
        # show the approximated contour image
        print("[INFO] {}".format(text))
        cv2.imshow("Approximated Contour", output)
        cv2.waitKey(0)

        cv2.imwrite(str(root_dir / f'{img_i}_pyr_img.png'), output)

    # Draw the contours on the original image
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Display the result
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()