from pathlib import Path
import random
import cv2
import numpy as np

from utils import *

def main():
    root_dir = Path(__file__).resolve().parent
    output_folder = create_dir_if_not_exists(root_dir / 'output/pt-seg')
    eval_folder = create_dir_if_not_exists(output_folder / 'eval')
    to_rect_output_folder = create_dir_if_not_exists(root_dir / 'output/to-rect-2')
    marker_img_path = root_dir / 'assets/in-img-marker.png'
    
    marker_img = cv2.imread(str(marker_img_path),0)
    
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
        
        cv2.imshow('image', pred_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
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
        hi, status = cv2.findHomography(dest_rect, src_rect)
        marker_area_img = cv2.warpPerspective(in_image_t, hi, (in_img_w, in_img_h))
        cv2.imshow('image', marker_area_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        # Compare markers
        res_marker_img = cv2.resize(marker_img, marker_area_img.shape[:2], interpolation=cv2.INTER_NEAREST_EXACT)
        inv_res_marker_img = cv2.bitwise_not(res_marker_img)
        min_diff = 9999999
        min_i = 0
        for i, marker_rot in enumerate([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
            rot_marker_area_img = cv2.rotate(marker_area_img, marker_rot) if marker_rot is not None else marker_area_img
            
            marker_img_diff = np.abs(rot_marker_area_img.astype('int32') - res_marker_img.astype('int32')).astype('uint8')
            marker_img_diff_sum = np.sum(marker_img_diff)
            inv_marker_img_diff = np.abs(rot_marker_area_img.astype('int32') - inv_res_marker_img.astype('int32')).astype('uint8')
            inv_marker_img_diff_sum = np.sum(inv_marker_img_diff)
            
            mdiff = min(marker_img_diff_sum, inv_marker_img_diff_sum)
            if mdiff < min_diff:
                min_diff = mdiff
                min_i = i
            
            print(mdiff, i, min_i, marker_img_diff_sum, inv_marker_img_diff_sum)
            cv2.imshow('image', cv2.hconcat([rot_marker_area_img, res_marker_img, marker_img_diff, inv_marker_img_diff]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            
        # Fix point order
        corners = np.roll(corners, min_i, axis=0)
        print('---roll-----------')
        for c in corners:
            print(c)
        
        # Write 
        in_image_grgb = cv2.imread(str(in_img_path))#cv2.cvtColor(in_image_t, cv2.COLOR_GRAY2RGB) #
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