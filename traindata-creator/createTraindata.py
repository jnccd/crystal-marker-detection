import os
import sys
import cv2
from pathlib import Path

window_name = "Window"

top_left_corner=[]
bottom_right_corner=[]
new_top_left = None
cur_m_pos = None

def mouseEvent(action, x, y, flags, *userdata):
    global top_left_corner, bottom_right_corner, new_top_left, cur_m_pos
  
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

root_dir = Path(__file__).resolve().parent
input_dir = root_dir / 'images'
input_img_paths = sorted(
    [
         os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)

image = cv2.imread(input_img_paths[0])
height, width = image.shape[:2]
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

    # Draw
    display_image = image.copy()
    for i in range(0, len(bottom_right_corner)):
        cv2.rectangle(display_image, top_left_corner[i], bottom_right_corner[i], (0,255,0), 1, 8)
    if new_top_left is not None:
        cv2.rectangle(display_image, new_top_left, cur_m_pos, (0,255,0), 1, 8)
    cv2.imshow(window_name, display_image)
 
cv2.destroyAllWindows()