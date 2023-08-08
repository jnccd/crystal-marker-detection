import cv2

# --- Imgs -------------------------------------------------------------------------------------------------------------------------

def set_img_width(img, max_width):
    img_h, img_w = img.shape[:2]
    resize_factor = float(max_width) / img_w
    target_size = (max_width, int(img_h * resize_factor))
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def set_img_height(img, max_height):
    img_h, img_w = img.shape[:2]
    resize_factor = float(max_height) / img_h
    target_size = (int(img_w * resize_factor), max_height)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def resize_img_by_factor(img, factor):
    img_h, img_w = img.shape[:2]
    target_size = (int(img_w * factor), int(img_h * factor))
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def keep_image_size_in_check(img, max_img_width=1920, max_img_height=1080):
    img_h, img_w = img.shape[:2]
    if img_w > max_img_width:
        img = set_img_width(img, max_img_width)
    if img_h > max_img_height:
        img = set_img_height(img, max_img_height)
    return img