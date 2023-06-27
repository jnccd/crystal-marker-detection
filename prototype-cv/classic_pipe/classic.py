import cv2
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent
test_img_path = root_dir / 'DSC_3741.JPG'

test_img = cv2.imread(str(test_img_path))
pyr_imgs = [test_img]
while pyr_imgs[-1].shape[0] > 64:
    pyr_h, pyr_w = pyr_imgs[-1].shape[:2]
    pyr_imgs.append(cv2.pyrDown(pyr_imgs[-1], dstsize=(int(pyr_w/2), int(pyr_h/2))))
    cv2.imshow('pyr',pyr_imgs[-1])
    cv2.waitKey(0)

