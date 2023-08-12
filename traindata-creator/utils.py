import os
from pathlib import Path
import random
import shutil
import sys
from typing import Literal, List
import cv2
from cv2 import Mat
import numpy as np
from copy import deepcopy
from shapely import Polygon, transform, intersection, LineString, Point

# --- Paths -------------------------------------------------------------------------------------------------------------------------

def swap_underscore_ending_in_path(p: Path, ending: str):
    return p.with_name(f'{"_".join(p.stem.split("_")[:-1])}{ending}')

def get_adjacent_files_with_ending(file_paths: List[Path], ending: str):
    paths = []
    for fpath in file_paths:
        if type(fpath) is str:
            fpath = Path(fpath)
        paths.append(swap_underscore_ending_in_path(fpath, ending))
    return paths   

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

def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# --- Textfiles -------------------------------------------------------------------------------------------------------------------------

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)
        
# --- Math -------------------------------------------------------------------------------------------------------------------------

def apply_homography(point2D_list, h, convert_to_int = True):
    hps = [h @ (p[0], p[1], 1) for p in point2D_list] 
    ps = [(p[0] / p[2], p[1] / p[2]) for p in hps]
    
    if convert_to_int:
        return [(int(p[0]), int(p[1])) for p in ps]
    else:
        return ps

def add_by_point(point2D_list, a):
    return [(p[0] + a[0], p[1] + a[1]) for p in point2D_list]

def mult_by_point(point2D_list, m):
    return [(p[0] * m[0], p[1] * m[1]) for p in point2D_list]

def divide_by_point(point2D_list, d):
    return [(p[0] / d[0], p[1] / d[1]) for p in point2D_list]

def flatten(list):
    return [item for sublist in list for item in sublist]

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]

def inflate_poly(p: Polygon, amount):
    centroid = p.centroid
    return transform(p, lambda x: np.array([(p[0] + (p[0] - centroid.x) * amount, p[1] + (p[1] - centroid.y) * amount) for p in x]))

def get_bounds(point2D_list):
    x = min([p[0] for p in point2D_list])
    y = min([p[1] for p in point2D_list])
    xe = max([p[0] for p in point2D_list])
    ye = max([p[1] for p in point2D_list])
    w = xe - x
    h = ye - y
    return x, y, w, h, xe, ye

def keep_box_in_bounds(xyxy_bbox, bounds):
    for i in range(2):
        if xyxy_bbox[i] < bounds[i]:
            xyxy_bbox[i] = bounds[i]
    for i in range(2,4):
        if xyxy_bbox[i] > bounds[i]:
            xyxy_bbox[i] = bounds[i]
    return xyxy_bbox

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

def resize_and_pad(img: Mat, desired_size: int, background_color = [0, 0, 0], border_type = cv2.BORDER_CONSTANT):
    old_size = img.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    rimg = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    brimg = cv2.copyMakeBorder(rimg, top, bottom, left, right, border_type, value=background_color)
    
    return brimg, new_size, top, left

def center_crop_to_11_aspect_ratio(img: Mat, size: int = -1):
    width, height = img.shape[1], img.shape[0]

    crop_size = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_size/2), int(crop_size/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    
    if size > 0:
        crop_img = cv2.resize(crop_img, (size, size))
    
    return crop_img

def create_random_persp_mat(img_size_wh, perspective_strength = 0.3):
    persp_strength_x = perspective_strength *0.5*img_size_wh[0]
    persp_strength_y = perspective_strength *0.5*img_size_wh[1]
    src_points = np.float32([[0, 0], [img_size_wh[0], 0], [img_size_wh[0], img_size_wh[1]], [0, img_size_wh[1]]])
    dst_points = np.float32([[0, 0], [img_size_wh[0], 0], [img_size_wh[0], img_size_wh[1]], [0, img_size_wh[1]]])
    persp_side = random.randrange(0, 4)
    if persp_side == 0:
        dst_points[0,0] += persp_strength_x
        dst_points[1,0] -= persp_strength_x
    elif persp_side == 1:
        dst_points[1,1] += persp_strength_y
        dst_points[2,1] -= persp_strength_y
    elif persp_side == 2:
        dst_points[3,0] += persp_strength_x
        dst_points[2,0] -= persp_strength_x
    elif persp_side == 3:
        dst_points[0,1] += persp_strength_y
        dst_points[3,1] -= persp_strength_y
    else:
        print('Invalid side!')
    return cv2.getPerspectiveTransform(src_points, dst_points)

# Taken from https://stackoverflow.com/questions/48979219/opencv-composting-2-images-of-differing-size and modified
def combine_two_color_images(back_img, fore_img, x = 0, y = 0, alpha = 0.5):
    back_img = back_img.copy()
    fore_img_h, fore_img_w = fore_img.shape[:2]
    
    # do composite on the upper-left corner of the background image.
    blended_portion = cv2.addWeighted(fore_img,
                alpha,
                back_img[y:fore_img_h+y,x:fore_img_w+x,:],
                1 - alpha,
                0,
                back_img)
    back_img[y:fore_img_h+y,x:fore_img_w+x,:] = blended_portion
    return back_img

# --- Traindata imgs -------------------------------------------------------------------------------------------------------------------------

def resize_and_pad_with_labels(img: Mat, desired_size: int, polys: List[Polygon], background_color = [0, 0, 0], border_type = cv2.BORDER_CONSTANT):
    img_h, img_w = img.shape[:2]
    rp_img, new_size, top, left = resize_and_pad(img, desired_size, background_color, border_type)
    
    # Transform polys into new coordinate system
    polys = [transform(p, lambda x: x * [new_size[1] / img_w, new_size[0] / img_h] + [left, top]) for p in polys]
    
    return rp_img, polys

def rasterize_polys(draw_img: Mat, polys: List[Polygon], draw_color: tuple = (255, 255, 255)):
    # print(polys)
    polys = list(filter(lambda poly: 
        poly is not Point and 
        poly is not LineString and 
        not str(poly).__contains__('LINESTRING') and 
        not str(poly).__contains__('POINT'), 
        polys))
    # print(polys)
    
    vertices_per_obj = [[(
        int(max(0, min(draw_img.shape[1], point[0]))), 
        int(max(0, min(draw_img.shape[0], point[1])))) 
                         for point in poly.exterior.coords[:-1]] for poly in polys]
    
    for vertices in vertices_per_obj:
        if vertices != []:
            np_vertices = np.array(vertices)
            cv2.fillPoly(draw_img, pts=[np_vertices], color=draw_color)
        
    return draw_img

# Taken from https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None, blurSize=5):
    """
    @brief      Overlays a transparant PNG onto another image using CV2
	
    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
	
    @return     Background image with overlay on top
    """
    
    bg_img = background_img.copy()
    
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
    else:
        img_to_overlay_t = img_to_overlay_t.copy()
        
    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,blurSize)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
    print('img1_bg, range', roi.shape, mask.shape)
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
    print('overlay_transparent, range', (x,y,w,h), bg_img.shape)
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    
    return bg_img

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
    return mixed_img
    
def segment_img_between_poly_labels(img: Mat, polys: List[Polygon], dim: Literal[0,1], collage_padding = 5):
    img_h, img_w = img.shape[:2]
    
    if dim == 0:
        polys = sorted(polys, key=lambda x: x.centroid.x)
    else:
        polys = sorted(polys, key=lambda x: x.centroid.y)
        
    segments = []
    for i in range(len(polys)-1):
        upper_poly_lower_bound = max(0, polys[i].bounds[2 if dim == 0 else 3])
        lower_poly_upper_bound = max(0, polys[i+1].bounds[0 if dim == 0 else 1])
        dim_distance = lower_poly_upper_bound - upper_poly_lower_bound
        if dim_distance > collage_padding and lower_poly_upper_bound < img.shape[1 - dim]:
            if len(segments) > 0:
                last_end = segments[-1]['end']
            else:
                last_end = 0
            # print("upper_poly_lower_bound", upper_poly_lower_bound,
            #   "lower_poly_upper_bound", lower_poly_upper_bound,
            #   "i", i,
            #   "polys[i+1].bounds", polys[i+1].bounds,)
            end = int((upper_poly_lower_bound + lower_poly_upper_bound) / 2)
            if dim == 0:
                corners = list(filter(lambda x: x.exterior.coords[0][0] > last_end and x.exterior.coords[0][0] < end, polys))
                seg_img = img[:, last_end:end]
            else:
                corners = list(filter(lambda x: x.exterior.coords[0][1] > last_end and x.exterior.coords[0][1] < end, polys))
                seg_img = img[last_end:end, :]
            segments.append({   
                'end': end,
                'beginning': last_end,
                'size': end - last_end,
                'poly_index': i,
                'corners': corners,
                'img': seg_img,
                })
    if len(segments) > 0:
        last_end = segments[-1]['end']
    else: 
        last_end = 0
    if dim == 0: # x
        end = img_w
        corners = list(filter(lambda x: x.exterior.coords[0][0] > last_end and x.exterior.coords[0][0] < end, polys))
        seg_img = img[:, last_end:end]
    else: # y
        end = img_h
        corners = list(filter(lambda x: x.exterior.coords[0][1] > last_end and x.exterior.coords[0][1] < end, polys))
        seg_img = img[last_end:end, :]
    new_segment = { 
        'end': end,
        'beginning': last_end,
        'size': end - last_end,
        'poly_index': len(polys)-1,
        'corners': corners,
        'img': seg_img,
        }
    segments.append(new_segment)
    # if new_segment['size'] != new_segment['img'].shape[1 - dim]:
    #     print("new_segment['size']", new_segment['size'], 
    #           "new_segment['img'].shape", new_segment['img'].shape,
    #           "dim", dim,
    #           "end", end,
    #           "last_end", last_end,
    #           "segments", segments,
    #           img_h, img_w,)
    assert new_segment['size'] == new_segment['img'].shape[1 - dim]
    
    return segments

def rebuild_img_from_segments(segments, out_img_size_wh, dim: Literal[0,1]):
    aug_image = np.zeros(tuple(reversed(out_img_size_wh)) + (3,), dtype = np.uint8)
    aug_polys = []
    pos = 0
    for seg in segments:
        if dim == 0: # x
            aug_image[:,pos:pos+seg['size']] = seg['img']
        else: # y
            aug_image[pos:pos+seg['size'],:] = seg['img']
            
        for poly_corners in seg['corners']:
            if dim == 0:
                aug_polys.append(Polygon([(x[0] - seg['beginning'] + pos, x[1]) for x in poly_corners.exterior.coords]))
            else:
                aug_polys.append(Polygon([(x[0], x[1] - seg['beginning'] + pos) for x in poly_corners.exterior.coords]))
        pos += seg['size']
    
    #cv2.imwrite(str(train_dir / (Path(other_img_path).stem + "_aug.png")), aug_image)
    
    return aug_image, aug_polys

def smart_grid_shuffle(img, polys: List[Polygon], img_size_wh):
    # segment in y first
    segments_y = segment_img_between_poly_labels(img, polys, 1)
    for seg_y in segments_y:
        segs_x = segment_img_between_poly_labels(seg_y['img'], seg_y['corners'], 0)
        random.shuffle(segs_x)
        
        seg_img_h, seg_img_w = seg_y['img'].shape[:2]
        segs_img, segs_polys = rebuild_img_from_segments(segs_x, (seg_img_w, seg_img_h), 0)
        
        seg_y['img'] = segs_img
        seg_y['corners'] = segs_polys
    random.shuffle(segments_y)
    return rebuild_img_from_segments(segments_y, img_size_wh, 1)

def homogeneous_mat_transform(
    img: Mat, 
    polys: List[Polygon], 
    img_size_wh, 
    M: Mat, 
    background_color = [0, 0, 0], 
    border_type = cv2.BORDER_CONSTANT,
    min_label_visiblity = 0.25,
    ):
    # If M is Affine make it homogeneous
    if M.shape[0] == 2:
        M = np.vstack([M, np.array([0, 0, 1])])
    
    # Transform
    img = cv2.warpPerspective(img, M, img_size_wh, borderMode=border_type, borderValue=background_color)
    polys = [transform(p, lambda x: np.array(apply_homography(x, M, convert_to_int=False))) for p in polys]
    
    # Drop low visibility labels
    img_area = get_poly_from_bounds((0,0,img_size_wh[0],img_size_wh[1]))
    polys = drop_low_visibility_labels(img, polys, img_area, min_label_visiblity)
    
    return img, polys

def random_crop(img: Mat, polys: List[Polygon], target_size_wh: tuple):
    if img.shape[1] < target_size_wh[0] + 1 or img.shape[0] < target_size_wh[1] + 1:
        return img, polys
    
    crop_pos_x = random.randrange(0, img.shape[1] - target_size_wh[0])
    crop_pos_y = random.randrange(0, img.shape[0] - target_size_wh[1])
    crop_img = img[crop_pos_y:crop_pos_y+target_size_wh[1], crop_pos_x:crop_pos_x+target_size_wh[0]]
    
    crop_area = get_poly_from_bounds((crop_pos_x,crop_pos_y,target_size_wh[0],target_size_wh[1]))
    polys = drop_low_visibility_labels(crop_img, polys, crop_area)
    polys = [transform(p, lambda x: np.array([(p[0] - crop_pos_x, p[1] - crop_pos_y) for p in x] )) for p in polys]
    
    return crop_img, polys

def random_crop_v2(img: Mat, polys: List[Polygon], target_size_wh: tuple, min_size_wh: tuple = (320, 320)):
    img_h, img_w = img.shape[:2]
    if img_w < min_size_wh[0] + 1 or img_h < min_size_wh[1] + 1:
        return img, polys
    
    target_size_ratio = target_size_wh[0] / target_size_wh[1]
    
    crop_pos_x = random.randrange(0, img.shape[1] - min_size_wh[0])
    crop_pos_y = random.randrange(0, img.shape[0] - min_size_wh[1])
    crop_width = random.randrange(max(min_size_wh[0], min_size_wh[1] * target_size_ratio), min(img.shape[1] - crop_pos_x, (img.shape[0] - crop_pos_y) * target_size_ratio))
    crop_height = int(crop_width / target_size_ratio)
    crop_img = img[crop_pos_y:crop_pos_y+crop_height, crop_pos_x:crop_pos_x+crop_width]
    crop_img = cv2.resize(crop_img, target_size_wh, interpolation=cv2.INTER_CUBIC)
    # print("crop_width", crop_width, crop_height)
    
    resize_w = target_size_wh[0] / crop_width
    resize_h = target_size_wh[1] / crop_height
    polys = [transform(p, lambda x: np.array([((p[0] - crop_pos_x) * resize_w, (p[1] - crop_pos_y) * resize_h) for p in x] )) for p in polys]
    crop_area = get_poly_from_bounds((0,0,target_size_wh[0],target_size_wh[1]))
    polys = drop_low_visibility_labels(crop_img, polys, crop_area)
    
    return crop_img, polys

def poly_label_dropout(img: Mat, polys: List[Polygon], draw_color: tuple = ()):
    
    if len(polys) == 0:
        return img, polys
    
    pi = random.randrange(len(polys))
    
    if len(draw_color) != 3:
        c = polys[pi].centroid
        # Sample color from poly centroid in img
        draw_color = [int(x) for x in img[int(c.y), int(c.x)]]
    
    img = rasterize_polys(img, [inflate_poly(polys[pi], 0.2)], draw_color)
    polys.pop(pi)
    
    return img, polys

def poly_label_move(img: Mat, polys: List[Polygon], draw_color: tuple = ()):
    img_h, img_w = img.shape[:2]
    pi = random.randrange(len(polys))
    
    if len(draw_color) != 3:
        c = polys[pi].centroid
        # Sample color from poly centroid in img
        draw_color = [int(x) for x in img[int(c.y), int(c.x)]]
    
    # # Input debug out
    # cv2.imwrite('./in_img.png', img)
    # debug_seg_img = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    # debug_seg_img = rasterize_polys(debug_seg_img, polys)
    # debug_seg_img = rasterize_polys(debug_seg_img, [polys[pi]], (0, 255, 0))
    # cv2.imwrite('./in_polys.png', debug_seg_img)
    
    # Extract the label poly to be moved
    move_poly = polys[pi]
    move_poly_og_bounds = [int(x) for x in move_poly.bounds]
    move_poly_img = img[move_poly_og_bounds[1]:move_poly_og_bounds[3], move_poly_og_bounds[0]:move_poly_og_bounds[2]].copy()
    # cv2.imwrite('./move_poly_img.png', move_poly_img)
    
    # Take out poly label
    img = rasterize_polys(img, [inflate_poly(move_poly, 0.2)], draw_color)
    polys.pop(pi)
    
    # Create mask
    alpha_channel_img = np.zeros(move_poly_img.shape[:2] + (1,), dtype = np.uint8)
    move_poly = transform(move_poly, lambda x: np.array( [(p[0] - move_poly_og_bounds[0], p[1] - move_poly_og_bounds[1]) for p in x] ))
    alpha_channel_img = rasterize_polys(alpha_channel_img, [move_poly], (255))
    # cv2.imwrite('./alpha_channel_img.png', alpha_channel_img)
    
    # Apply mask
    move_poly_img = cv2.cvtColor(move_poly_img, cv2.COLOR_BGR2BGRA)
    move_poly_img[:, :, 3] = np.squeeze(alpha_channel_img)
    
    # Reinsert poly label
    new_pos_x = random.randrange(0, img_w - (move_poly_og_bounds[2] - move_poly_og_bounds[0]))
    new_pos_y = random.randrange(0, img_h - (move_poly_og_bounds[3] - move_poly_og_bounds[1]))
    # cv2.imwrite('./move_poly_img2.png', move_poly_img)
    img = overlay_transparent(img, move_poly_img, new_pos_x, new_pos_y, blurSize=1)
    move_poly = transform(move_poly, lambda x: np.array( [(p[0] + new_pos_x, p[1] + new_pos_y) for p in x] ))
    polys.append(move_poly)
    # cv2.imwrite('./img.png', img)
    # print(polys)
    # sys.exit(0)
    
    return img, polys

def poly_label_move_v2(img: Mat, polys: List[Polygon], draw_color: tuple = ()):
    img_h, img_w = img.shape[:2]
    blur_strength = 7
    
    # Choose random polygon to be moved
    pi = random.randrange(len(polys))
    
    if len(draw_color) != 3:
        c = polys[pi].centroid
        # Sample color from poly centroid in img
        draw_color = [int(x) for x in img[int(c.y), int(c.x)]]
    
    # Extract the label poly to be moved
    move_poly = polys[pi]
    move_poly_og_bounds = [int(x) for x in move_poly.bounds]
    move_poly_og_bounds = [move_poly_og_bounds[0] - blur_strength, 
                           move_poly_og_bounds[1] - blur_strength,
                           move_poly_og_bounds[2] + blur_strength,
                           move_poly_og_bounds[3] + blur_strength]
    move_poly_img = img[move_poly_og_bounds[1]:move_poly_og_bounds[3], move_poly_og_bounds[0]:move_poly_og_bounds[2]].copy()
    img = rasterize_polys(img, [inflate_poly(move_poly, 0.2)], draw_color)
    polys.pop(pi)
    
    # Set target position
    og_bounds_width = move_poly_og_bounds[2] - move_poly_og_bounds[0]
    og_bounds_height = move_poly_og_bounds[3] - move_poly_og_bounds[1]
    while True:
        new_pos_x = random.randrange(0, img_w - og_bounds_width)
        new_pos_y = random.randrange(0, img_h - og_bounds_height)
        target_pos_poly = get_poly_from_bounds((new_pos_x, new_pos_y, og_bounds_width, og_bounds_height))
        if not any([x.intersects(target_pos_poly) for x in polys]):
            break
    
    # Adapt colors roughly to target position
    target_img_section = img[new_pos_y:new_pos_y+og_bounds_height, new_pos_x:new_pos_x+og_bounds_width]
    target_img_avgs = np.array([np.average(target_img_section[:,:,x]) for x in range(3)])
    move_poly_img_avgs = np.array([np.average(move_poly_img[:,:,x]) for x in range(3)])
    diff_avgs = target_img_avgs - move_poly_img_avgs
    for y in range(move_poly_img.shape[0]):
        for x in range(move_poly_img.shape[1]):
            for c in range(3):
                move_poly_img[y, x, c] += diff_avgs[c]
    #cv2.imwrite('test.png', move_poly_img)
    
    # Create mask
    alpha_channel_img = np.zeros(move_poly_img.shape[:2] + (1,), dtype = np.uint8)
    move_poly = transform(move_poly, lambda x: np.array( [(p[0] - move_poly_og_bounds[0], p[1] - move_poly_og_bounds[1]) for p in x] )) # Moved to mask coord system for rasterization
    alpha_channel_img = rasterize_polys(alpha_channel_img, [move_poly], (255))
    # Blur mask edges
    kernel = cv2.getGaussianKernel(blur_strength, 1)
    alpha_channel_img = cv2.filter2D(alpha_channel_img, -1, kernel)
    kernel = cv2.transpose(kernel)
    alpha_channel_img = cv2.filter2D(alpha_channel_img, -1, kernel)
    
    # Apply mask
    move_poly_img = cv2.cvtColor(move_poly_img, cv2.COLOR_BGR2BGRA)
    move_poly_img[:, :, 3] = np.squeeze(alpha_channel_img)
    #cv2.imwrite('test2.png', move_poly_img)
    
    # Reinsert poly label
    img = overlay_transparent_fore_alpha(img, move_poly_img, new_pos_x, new_pos_y)
    move_poly = transform(move_poly, lambda x: np.array( [(p[0] + new_pos_x, p[1] + new_pos_y) for p in x] ))
    polys.append(move_poly)
    
    return img, polys

def get_poly_from_bounds(bounds_xywh: tuple):
    return Polygon([[bounds_xywh[0], bounds_xywh[1]], [bounds_xywh[2], bounds_xywh[1]], [bounds_xywh[2], bounds_xywh[3]], [bounds_xywh[0], bounds_xywh[3]]])

def drop_low_visibility_labels(img: Mat, polys: List[Polygon], visible_area: Polygon, min_label_visiblity = 0.6):
    new_polys = []
    
    for i in range(len(polys)):
        visible_label_poly = intersection(polys[i], visible_area)
        visibility = visible_label_poly.area / polys[i].area
        
        if visibility < 1:
            polys[i] = visible_label_poly
        if visibility >= min_label_visiblity:
            new_polys.append(polys[i])
        elif not polys[i].is_empty:
            c = [polys[i].centroid.x, polys[i].centroid.y]
            if c[0] < 0:
                c[0] = 0
            if c[1] < 0:
                c[1] = 0
            if c[0] > img.shape[1] - 1:
                c[0] = img.shape[1] - 1
            if c[1] > img.shape[0] - 1:
                c[1] = img.shape[0] - 1
            # Sample color from poly centroid in img
            draw_color = [int(x) for x in img[int(c[1]), int(c[0])]]
            print(c)
            img = rasterize_polys(img, [inflate_poly(polys[i], 0.2)], draw_color)
            
    return new_polys
    