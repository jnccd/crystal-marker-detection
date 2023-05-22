import os
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