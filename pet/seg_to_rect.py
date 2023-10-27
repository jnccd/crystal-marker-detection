from pathlib import Path
import random
import cv2
import numpy as np

from utils import *

def main():
    root_dir = Path(__file__).resolve().parent
    output_folder = create_dir_if_not_exists(root_dir / 'output/pt-seg')
    eval_folder = create_dir_if_not_exists(output_folder / 'eval')
    to_rect_output_folder = create_dir_if_not_exists(root_dir / 'output/to-rect')
    
    pred_img_paths = [Path(x) for x in get_files_from_folders_with_ending([eval_folder], '_pred.png')]
    for pred_img_path in pred_img_paths:
        img = cv2.imread(str(pred_img_path))
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_pyr_img.png'), img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        dst = cv2.cornerHarris(gray,17,9,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        print('--------------')
        for i in range(1, len(corners)):
            print(corners[i])
        img[dst>0.1*dst.max()]=[0,0,255]
        cv2.imshow('image', img)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    main()