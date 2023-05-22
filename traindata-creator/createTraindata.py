import os
import sys
import time
import cv2
from pathlib import Path

import numpy as np

window_name = "Window"

top_left_corner=[]
bottom_right_corner=[]
new_top_left = None
cur_m_pos = None

def mouseEvent(action, x, y, flags, *userdata):
    global window_name, top_left_corner, bottom_right_corner, new_top_left, cur_m_pos
  
    cur_m_pos = (x,y)
    if action == cv2.EVENT_LBUTTONDOWN:
        new_top_left = (x,y)
    elif action == cv2.EVENT_LBUTTONUP:
        top_left_corner.append(new_top_left)
        bottom_right_corner.append((x,y))
        new_top_left = None
        
def aruco_transform_and_display(corners, ids, rejected, image):
    
    centers = []
    out_corners = []
    
    if len(corners) > 0:
		
        ids = ids.flatten()
        
        for (markerCorner, markerID) in zip(corners, ids):
			
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
			
            out_corners.append([topLeft, topRight, bottomRight, bottomLeft])
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            cv2.rectangle(image, topLeft, (topLeft[0]+1, topLeft[1]+1), (0, 0, 255), 5)
			
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            centers.append((cX, cY))
			
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
			
    return image, centers, out_corners

def find_inner_rect(cornerss, ccx, ccy):
        in_between_rect = [None, None, None, None]
        for corners in cornerss:
            
            min_i = sys.maxsize
            min_v = sys.maxsize
            for i in range(len(corners)):
                v = corners[i]
                cv = abs(v[0] - ccx) + abs(v[1] - ccy)
                if cv <= min_v:
                    min_i = i
                    min_v = cv
                    
            min_vert = corners[min_i]
            while in_between_rect[min_i] is not None:
                min_i += 1
                min_i = min_i % 4
            in_between_rect[min_i] = (int(min_vert[0]), int(min_vert[1]))
            
        return in_between_rect
    
def get_opencv_aruco_detector(dict):
    dictionary = cv2.aruco.getPredefinedDictionary(dict)
    parameters =  cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)
    
def find_homography_from_aruco(img, detector, width, height):
    corners, ids, rejected = detector.detectMarkers(img)
    marked_img, centers, cornerss = aruco_transform_and_display(corners, ids, rejected, img.copy())
    
    # Found paper
    if len(corners) == 4:
        # Find center
        ccx = int(np.mean([c[0] for c in centers]))
        ccy = int(np.mean([c[1] for c in centers]))
        cv2.rectangle(marked_img, (ccx, ccy), (ccx+1, ccy+1), (0, 0, 255), 5)
        
        # Find inner rectangle
        in_between_rect = find_inner_rect(cornerss, ccx, ccy)
        for i in range(0,4):
            j = (i+1)%4
            cv2.line(marked_img, in_between_rect[i], in_between_rect[j], (0, 0, 255), 2)
        
        # Find Homography
        src_rect  = np.array([[0, height, 1], [0, 0, 1], [width, 0, 1], [width, height, 1]])
        dest_rect = np.array([[x,y,1] for (x,y) in in_between_rect])
        h, status = cv2.findHomography(src_rect, dest_rect)
        hi, status = cv2.findHomography(dest_rect, src_rect)
        return h, hi, marked_img
    else:
        return None, None, marked_img
    
def main():
    global window_name, top_left_corner, bottom_right_corner, new_top_left, cur_m_pos
    
    # Prepare paths
    root_dir = Path(__file__).resolve().parent
    input_dir = root_dir / 'images'
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )
    marked_dir = root_dir / 'images_marked'
    if not os.path.exists(marked_dir):
        os.makedirs(marked_dir)
    
    # Load first image and preprocess
    img = cv2.imread(input_img_paths[0])
    img_h, img_w = img.shape[:2]
    print(img.shape[:2])
    detector = get_opencv_aruco_detector(cv2.aruco.DICT_6X6_50)
    oh, hi, marked_img = find_homography_from_aruco(img, detector, img_w, img_h)
    if hi is None:
        print("Didn't find the aruco frame in base img :/")
        return
    warped_img = cv2.warpPerspective(img, hi, (img_w, img_h))
    # Show marked_img for troubleshooting
    cv2.imshow(window_name, marked_img)
    cv2.waitKey(0)
    
    # Load window and hook events
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseEvent)
    
    while True:
        # Control logic at 60 FPS
        k = cv2.waitKey(16) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            top_left_corner=[]
            bottom_right_corner=[]
            new_top_left = None
        elif k == ord('z'):
            top_left_corner.pop()
            bottom_right_corner.pop()
        elif k == ord(' '):
            print("Building...")
            warped_inner_rect_cornerss = []
            for i in range(0, len(bottom_right_corner)):
                # Clockwise corner point lists starting at top left for all marked rects
                warped_inner_rect_cornerss.append([top_left_corner[i], (bottom_right_corner[i][0], top_left_corner[i][1]), bottom_right_corner[i], (top_left_corner[i][0], bottom_right_corner[i][1])])
            # Flatten
            warped_inner_rect_corners = [item for sublist in warped_inner_rect_cornerss for item in sublist]
            # Iterate through other images
            for other_img_path in input_img_paths[1:]:
                print(f"Building {other_img_path}...")
                other_img = cv2.imread(other_img_path)
                h, hi, marked_img = find_homography_from_aruco(other_img, detector, img_w, img_h)
                cv2.imshow(window_name, marked_img)
                if h is None:
                    print("Didn't find the aruco frame :/")
                    continue
                hircs = [h @ (p[0], p[1], 1) for p in warped_inner_rect_corners] # homogeneous_in_other_img_inner_rect_corners
                ircs = [(int(p[0] / p[2]), int(p[1] / p[2])) for p in hircs]
                
                print("hircs", hircs)
                print("ircs", ircs)
                
                draw_other_img = other_img.copy()
                for i in range(0, len(ircs)):
                    if i % 4 == 3:
                        cv2.line(draw_other_img, ircs[i], ircs[i-3], (0,0,255), 2)
                    else:
                        cv2.line(draw_other_img, ircs[i], ircs[i+1], (0,0,255), 2)
                cv2.imshow(window_name, draw_other_img)
                cv2.waitKey(32)
                cv2.imwrite(str(marked_dir / (Path(other_img_path).stem + ".png")), draw_other_img)
                

        # Draw
        display_img = warped_img.copy()
        for i in range(0, len(bottom_right_corner)):
            cv2.rectangle(display_img, top_left_corner[i], bottom_right_corner[i], (0,255,0), 1, 8)
        if new_top_left is not None:
            cv2.rectangle(display_img, new_top_left, cur_m_pos, (0,255,0), 1, 8)
        cv2.imshow(window_name, display_img)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()