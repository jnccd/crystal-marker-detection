import os
from pathlib import Path
import shutil
from math import isnan
import cv2
import numpy as np

# --- Textfiles -------------------------------------------------------------------------------------------------------------------------

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)
        
# --- Other -------------------------------------------------------------------------------------------------------------------------

def get_files_from_folders_with_ending(folders, ending):
    paths = []
    for folder in folders:
        paths.extend(sorted(
            [
                os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(ending)
            ]
        ))
    return paths

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

def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def flatten(list):
    return [item for sublist in list for item in sublist]

def overlay_transparent_fore_alpha(background_img, foreground_img, x = 0, y = 0):
    if foreground_img.shape[0] != background_img.shape[0] or \
        foreground_img.shape[1] != background_img.shape[1]:
        #print('reshaping...')
        new_foreground_img = np.zeros(background_img.shape[:2] + (foreground_img.shape[2],))
        new_foreground_img[y:y+foreground_img.shape[0], x:x+foreground_img.shape[1]] = foreground_img
        #print(new_foreground_img.shape, foreground_img.shape, background_img.shape)
        
        foreground_img = new_foreground_img
    
    # Create weights
    bg_channels = background_img.shape[2]
    weights = foreground_img[:,:,3].astype('float32') / 255
    weights = np.repeat(weights[:,:,np.newaxis], bg_channels, axis=2)
    
    weighted_bg = background_img * (1 - weights)
    weighted_fg = foreground_img[:,:,:bg_channels] * weights
    mixed_img = weighted_bg + weighted_fg
    return mixed_img.astype('uint8')

# --- Other -------------------------------------------------------------------------------------------------------------------------

def handle_model_out(
    i,
    boxes, 
    img_w, 
    img_h, 
    out_testdata_path,
    label_path,
    confidence_threshold, 
    border_ignore_size, 
    squareness_threshold,
    build_debug_output = False,
    mask = None,
    thesis_output_imgs = False,
    input_img = None,
    ):
    
    # Draw _raw_detections image
    raw_draw_img = input_img.copy()
    if thesis_output_imgs:
        draw_img = input_img.copy()
        for box in boxes:
            cv2.rectangle(draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        for box in boxes:
            cv2.rectangle(raw_draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 0), thickness=2)
        cv2.imwrite(str(out_testdata_path / f'{i}_raw_detections.png'), draw_img)
    
    # Filter for confidence
    boxes = list(filter(lambda box: box[4] > confidence_threshold, boxes))
    
    # Draw _confidence_filtered_detections image
    confidence_f_draw_img = input_img.copy()
    if thesis_output_imgs:
        for box in boxes:
            cv2.rectangle(raw_draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(out_testdata_path / f'{i}_confidence_filtered_detections.png'), raw_draw_img)
        
        for box in boxes:
            cv2.rectangle(confidence_f_draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 0), thickness=2)
    
    # Filter for border
    if border_ignore_size > 0:
        boxes = list(filter(lambda box: #xmin, ymin, xmax, ymax, conf: 
            box[0] / img_w > border_ignore_size and 
            box[1] / img_h > border_ignore_size and
            1 - (box[2] / img_w) > border_ignore_size and 
            1 - (box[3] / img_h) > border_ignore_size, boxes))
    
    # Draw BIS change image
    bis_f_draw_img = input_img.copy()
    if thesis_output_imgs:
        for box in boxes:
            cv2.rectangle(confidence_f_draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        mask_image = np.zeros((img_h, img_w) + (4,), dtype = np.uint8)
        cv2.rectangle(mask_image, pt1=(0, 0), pt2=(img_w, int(img_h * border_ignore_size)), color=(0, 0, 255, 50), thickness=-1)
        cv2.rectangle(mask_image, pt1=(0, 0), pt2=(int(img_w * border_ignore_size), img_h), color=(0, 0, 255, 50), thickness=-1)
        cv2.rectangle(mask_image, pt1=(int(img_w * (1 - border_ignore_size)), 0), pt2=(img_w, img_h), color=(0, 0, 255, 50), thickness=-1)
        cv2.rectangle(mask_image, pt1=(0, int(img_h * (1 - border_ignore_size))), pt2=(img_w, img_h), color=(0, 0, 255, 50), thickness=-1)
        confidence_f_draw_img = overlay_transparent_fore_alpha(confidence_f_draw_img, mask_image)
        cv2.imwrite(str(out_testdata_path / f'{i}_bis_filtered_detections.png'), confidence_f_draw_img)
        
        for box in boxes:
            cv2.rectangle(bis_f_draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 0), thickness=2)
    
    # Filter SQT and mask
    if squareness_threshold > 0:
        boxes = list(filter(lambda box: ((box[2] - box[0]) / (box[3] - box[1]) if (box[2] - box[0]) / (box[3] - box[1]) < 1 else 1 / ((box[2] - box[0]) / (box[3] - box[1]))) > squareness_threshold, boxes))
    if mask is not None:
        # print(boxes)
        box_windows = [mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in boxes]
        box_windows = [(wind, box) for wind, box in zip(box_windows, boxes) 
                       if wind.shape[0] != 0 and wind.shape[1] != 0]
        # print(box_windows)
        # print(mask.shape, [x[0].shape for x in box_windows], [np.max(x[0]) for x in box_windows])
        boxes = [box for wind, box in box_windows 
                    if np.max(wind) > 80]
        all_boxes = [box for wind, box in box_windows 
                        if True]
        # print(boxes)
        sanity_check_image = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
        
        # write mask
        cv2.imwrite(str(out_testdata_path / f'{i}_mask.png'), mask)
        
        # write mask diff in pred boxes
        for box in all_boxes:
            if box in boxes:
                box_color = (0, 255, 0)
            else:
                box_color = (0, 0, 255)
            int_box = [int(x) for x in box]
            print(box[4])
            cv2.rectangle(sanity_check_image, int_box[:2], int_box[2:4], box_color)
            cv2.putText(sanity_check_image, str(round(box[4], 2)), int_box[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, box_color)
        cv2.imwrite(str(out_testdata_path / f'{i}_mask_dropout_check.png'), sanity_check_image)
    
    # Draw filtered detections
    if thesis_output_imgs:
        draw_img = input_img.copy()
        for box in boxes:
            cv2.rectangle(draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(out_testdata_path / f'{i}_filtered_detections.png'), draw_img)
    
    # Rasterize Segmentation image
    if build_debug_output:
        sanity_check_image = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
        for min_x, min_y, max_x, max_y, conf in boxes:
            verts = np.array([(int(min_x), int(min_y)), (int(min_x), int(max_y)), (int(max_x), int(max_y)), (int(max_x), int(min_y)), (int(min_x), int(min_y))])
            cv2.fillPoly(sanity_check_image, pts=[verts], color=(255, 255, 255))
        cv2.imwrite(str(out_testdata_path / f'{i}_network_output.png'), sanity_check_image)
    # Write model out
    with open(out_testdata_path / f'{i}_network_output.txt', "w") as text_file:
        for xmin, ymin, xmax, ymax, conf in boxes:
            text_file.write(f"{xmin} {ymin} {xmax} {ymax} {conf}\n")
        
    # Write labels
    # Rasterize Segmentation image
    sanity_check_image = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    with open(label_path, 'r') as file:
        vd_bbox_lines = file.read().split('\n')
    vd_bbox_lines_og = vd_bbox_lines
    vd_bbox_lines = list(filter(lambda s: s and not s.isspace(), vd_bbox_lines)) # Filter whitespace lines away
    target_output_path = out_testdata_path / f'{i}_target_output.txt'
    ground_truth_boxes = []
    with open(target_output_path, "w") as text_file:
        for line in vd_bbox_lines:
            sc, sx, sy, sw, sh = line.split(' ')
            
            if any(isnan(float(x)) for x in [sx, sy, sw, sh]):
                print(f'Encountered NaN output in {label_path}', list(vd_bbox_lines), vd_bbox_lines_og, sx, sy, sw, sh)
                continue
            
            bbox_w = float(sw) * img_w
            bbox_h = float(sh) * img_h
            min_x = float(sx) * img_w - bbox_w / 2
            min_y = float(sy) * img_h - bbox_h / 2
            max_x = bbox_w + min_x
            max_y = bbox_h + min_y
            
            text_file.write(f"{min_x} {min_y} {max_x} {max_y}\n")
            ground_truth_boxes.append((min_x, min_y, max_x, max_y))
            
            if build_debug_output:
                verts = np.array([(int(min_x), int(min_y)), (int(min_x), int(max_y)), (int(max_x), int(max_y)), (int(max_x), int(min_y))])
                cv2.fillPoly(sanity_check_image, pts=[verts], color=(255, 255, 255))
                cv2.imwrite(str(out_testdata_path / f'{i}_target_output.png'), sanity_check_image)
                
    # Draw filtered detections and GT
    if thesis_output_imgs:
        draw_img = input_img.copy()
        for box in ground_truth_boxes:
            cv2.rectangle(draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(255, 255, 0), thickness=2)
        for box in boxes:
            cv2.rectangle(draw_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(out_testdata_path / f'{i}_preds_and_gts.png'), draw_img)