import cv2
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent
marker_img_path = root_dir / '..' / 'sift' / 'sift-base' / 'border-marker-scaled.png'

marker_img = cv2.imread(str(marker_img_path))
marker_img = cv2.resize(marker_img, (200, 200), interpolation=cv2.INTER_CUBIC)

color1 = (0, 0, 255)
color2 = (255, 255, 0)

window_name = 'window_name'
padding = 100
img_h, img_w = (marker_img.shape[0] + padding * 2, marker_img.shape[1] + padding * 2)
cur_m_pos = (0,0)
drawn_rect_size = 25

# Detector code
def get_opencv_aruco_detector(dict):
    dictionary = cv2.aruco.getPredefinedDictionary(dict)
    parameters =  cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)
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

            cv2.line(image, topLeft, topRight, color1, 2)
            cv2.line(image, topRight, bottomRight, color1, 2)
            cv2.line(image, bottomRight, bottomLeft, color1, 2)
            cv2.line(image, bottomLeft, topLeft, color1, 2)
            cv2.rectangle(image, topLeft, (topLeft[0]+1, topLeft[1]+1), color2, 5)
			
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, color2, -1)
            centers.append((cX, cY))
			
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color1, 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
			
    return image, centers, out_corners

def mouseEvent(action, x, y, flags, *userdata):
    global cur_m_pos
    
    cur_m_pos = (x,y)
    #print(cur_m_pos)

# Load window and hook events
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouseEvent)
detector = get_opencv_aruco_detector(cv2.aruco.DICT_6X6_50)

while True:
    # Control logic at 60 FPS
    k = cv2.waitKey(16) & 0xFF
    if k == ord('q'):
        break
    elif k == ord(' '):
        cv2.imwrite(str(root_dir / 'out.png'), drawn_img)

    # Draw
    drawn_img = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    drawn_img.fill(255)
    drawn_img[padding:img_h-padding, padding:img_w-padding] = marker_img
    cv2.rectangle(drawn_img, (cur_m_pos[0] - drawn_rect_size, cur_m_pos[1] - drawn_rect_size, drawn_rect_size*2, drawn_rect_size*2), (255,255,255), -1)
    corners, ids, rejected = detector.detectMarkers(drawn_img)
    drawn_img, centers, cornerss = aruco_transform_and_display(corners, ids, rejected, drawn_img)
    cv2.imshow(window_name, drawn_img)
