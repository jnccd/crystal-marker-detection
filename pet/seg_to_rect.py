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
    in_img_paths = [Path(x) for x in get_files_from_folders_with_ending([eval_folder], '_in.png')]
    for pred_img_path, in_img_path in zip(pred_img_paths, in_img_paths):
        in_img = cv2.imread(str(in_img_path))
        in_img_h, in_img_w = in_img.shape[:2]
        pred_img = cv2.imread(str(pred_img_path))
        pred_img_h, pred_img_w = pred_img.shape[:2]
        
        gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        # Get corners
        dst = cv2.cornerHarris(gray,17,9,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners: np.ndarray = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        pred_img[dst>0.1*dst.max()]=[0,0,255]
        
        if len(corners) != 5:
            continue
        
        # Swap last 2 corners and delete first entry
        print(corners.shape)
        corners = corners[1:, :]
        swap = corners[2].copy()
        corners[2] = corners[3].copy()
        corners[3] = swap
        # print corners
        print('--------------')
        for c in corners:
            print(c)
        
        # Thresh in img
        in_image_gray = cv2.imread(str(in_img_path),0)
        block_size = 90#int(img_w/600*50)
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        in_image_t = cv2.adaptiveThreshold(in_image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,block_size,3)
        
        # Homography
        src_rect  = np.array([[0, in_img_h, 1], [0, 0, 1], [in_img_w, 0, 1], [in_img_w, in_img_h, 1]])
        dest_rect = corners
        h, status = cv2.findHomography(src_rect, dest_rect)
        hi, status = cv2.findHomography(dest_rect, src_rect)
        marker_img = cv2.warpPerspective(in_image_t, hi, (in_img_w, in_img_h))
        cv2.imshow('image', marker_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        # Fix point order
        
        
        # Write 
        in_image_grgb = cv2.cvtColor(in_image_t, cv2.COLOR_GRAY2RGB) #cv2.imread(str(in_img_path),0)
        pts = np.array([(int(point[0]), int(point[1])) for point in corners], dtype=np.int32)
        in_image_grgb = cv2.polylines(in_image_grgb, pts=[pts], isClosed=True, color=(0,255,0))
        for i, pt in enumerate(pts):
            in_image_grgb = cv2.putText(in_image_grgb, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                  (0,0,255), 2, cv2.LINE_AA, False)
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_rect.png'), in_image_grgb)
        cv2.imshow('image', in_image_grgb)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    main()