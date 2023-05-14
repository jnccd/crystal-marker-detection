import sys
import numpy as np
import time
import cv2

def aruco_display(corners, ids, rejected, image):
    
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

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while cap.isOpened():
    
    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width*(h/w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
 
    corners, ids, rejected = detector.detectMarkers(img)
    marked_img, centers, vertss = aruco_display(corners, ids, rejected, img)
    
    # Found paper
    if len(corners) == 4:
        ccx = int(np.mean([c[0] for c in centers]))
        ccy = int(np.mean([c[1] for c in centers]))
        print("Center", ccx, " ", ccy)
        cv2.rectangle(marked_img, (ccx, ccy), (ccx+1, ccy+1), (0, 0, 255), 5)
        
        # Find inner rectangle
        in_between_rect = [None, None, None, None]
        for verts in vertss:
            
            min_i = sys.maxsize
            min_v = sys.maxsize
            for i in range(len(verts)):
                v = verts[i]
                cv = abs(v[0] - ccx) + abs(v[1] - ccy)
                if cv <= min_v:
                    min_i = i
                    min_v = cv
                    
            min_vert = verts[min_i]
            in_between_rect[min_i] = (int(min_vert[0]), int(min_vert[1]))
        # Draw inner rectangle
        print("in_between_rect", in_between_rect)
        for i in range(0,4):
            j = (i+1)%4
            cv2.line(marked_img, in_between_rect[i], in_between_rect[j], (0, 0, 255), 2)
            
        if in_between_rect.__contains__(None):
            print('continuing...')
            continue
        print('doing stuff')
        
        # Find Homography
        src_rect  = np.array([[0, height, 1], [0, 0, 1], [width, 0, 1], [width, height, 1]])
        dest_rect = np.array([[x,y,1] for (x,y) in in_between_rect])
        print("src_rect", src_rect)
        print("dest_rect", dest_rect)
        h, status = cv2.findHomography(src_rect, dest_rect)
        print("status", status)
        
        # Get points
        pattern_image = cv2.imread('pattern.png',0)
        rows,cols = pattern_image.shape
        pattern_points = []
        for i in range(rows):
            for j in range(cols):
                if pattern_image[i,j] < 255:
                    pattern_points.append(np.array([i*width/rows,j*height/cols,1]))
        print("pattern_points", pattern_points)
        
        # Transform points using Homography
        for p in pattern_points:
            p_in_img = h @ p
            print("p_in_img", p_in_img)
            ix = int(p_in_img[0] / p_in_img[2])
            iy = int(p_in_img[1] / p_in_img[2])
            cv2.rectangle(marked_img, (ix, iy), (ix+1, iy+1), (255, 0, 0), 5)

    cv2.imshow("Image", marked_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()