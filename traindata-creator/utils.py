import os
from pathlib import Path
import random
import shutil
from typing import Literal
import cv2
from cv2 import Mat
import numpy as np
from shapely import Polygon, transform

# --- Paths -------------------------------------------------------------------------------------------------------------------------

def get_adjacent_files_with_ending(file_paths: list[Path], ending):
    paths = []
    for fpath in file_paths:
        if type(fpath) is str:
            fpath = Path(fpath)
        paths.append(fpath.with_name(f'{"_".join(fpath.stem.split("_")[:-1])}{ending}'))
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
    
def get_bounds(point2D_list):
    x = min([p[0] for p in point2D_list])
    y = min([p[1] for p in point2D_list])
    xe = max([p[0] for p in point2D_list])
    ye = max([p[1] for p in point2D_list])
    w = xe - x
    h = ye - y
    return x, y, w, h, xe, ye

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

# --- Traindata imgs -------------------------------------------------------------------------------------------------------------------------

def resize_and_pad_with_labels(img: Mat, desired_size: int, polys: list[Polygon], background_color = [0, 0, 0], border_type = cv2.BORDER_CONSTANT):
    img_h, img_w = img.shape[:2]
    rp_img, new_size, top, left = resize_and_pad(img, desired_size, background_color, border_type)
    
    # Transform polys into new coordinate system
    polys = [transform(p, lambda x: x * [new_size[1] / img_w, new_size[0] / img_h] + [left, top]) for p in polys]
    
    return rp_img, polys

def rasterize_polys(draw_img: Mat, polys: list[Polygon], draw_color: tuple = (255, 255, 255)):
    vertices_per_obj = [[(int(point[0]), int(point[1])) for point in poly.exterior.coords[:-1]] for poly in polys]
    
    for vertices in vertices_per_obj:
        np_vertices = np.array(vertices)
        cv2.fillPoly(draw_img, pts=[np_vertices], color=draw_color)
        
    return draw_img
    
def segment_img_between_poly_labels(img: Mat, polys: list[Polygon], dim: Literal[0,1], collage_padding = 5):
    img_h, img_w = img.shape[:2]
    
    if dim == 0:
        polys = sorted(polys, key=lambda x: x.centroid.x)
    else:
        polys = sorted(polys, key=lambda x: x.centroid.y)
        
    segments = []
    for i in range(len(polys)-1):
        upper_poly_lower_bound = polys[i].bounds[2 if dim == 0 else 3]
        lower_poly_upper_bound = polys[i+1].bounds[0 if dim == 0 else 1]
        dim_distance = lower_poly_upper_bound - upper_poly_lower_bound
        if dim_distance > collage_padding:
            if len(segments) > 0:
                last_end = segments[-1]['end']
            else:
                last_end = 0
            end = int((upper_poly_lower_bound + lower_poly_upper_bound) / 2)
            if dim == 0:
                corners = list(filter(lambda x: x.exterior.coords[0][0] > last_end and x.exterior.coords[0][0] < end, polys))
                seg_img = img[0:img_w, last_end:end]
            else:
                corners = list(filter(lambda x: x.exterior.coords[0][1] > last_end and x.exterior.coords[0][1] < end, polys))
                seg_img = img[last_end:end, 0:img_h]
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
    end = img_w
    if dim == 0:
        corners = list(filter(lambda x: x.exterior.coords[0][0] > last_end and x.exterior.coords[0][0] < end, polys))
        seg_img = img[0:img_w, last_end:end]
    else:
        corners = list(filter(lambda x: x.exterior.coords[0][1] > last_end and x.exterior.coords[0][1] < end, polys))
        seg_img = img[last_end:end, 0:img_h]
    segments.append({ 
        'end': end,
        'beginning': last_end,
        'size': end - last_end,
        'poly_index': len(polys)-1,
        'corners': corners,
        'img': seg_img,
        })
    
    return segments

def rebuild_img_from_segments(segments, out_img_size_wh, dim: Literal[0,1]):
    aug_image = np.zeros(tuple(reversed(out_img_size_wh)) + (3,), dtype = np.uint8)
    aug_polys = []
    pos = 0
    for seg in segments:
        if dim == 0: # x
            aug_image[0:out_img_size_wh[1],pos:pos+seg['size']] = seg['img']
        else: # y
            aug_image[pos:pos+seg['size'],0:out_img_size_wh[0]] = seg['img']
            
        #print('y_pos',y_pos)
        #print('seg[beginning]',seg['beginning'])
        
        #print(dim, 'seg corners',seg['corners'])
        for poly_corners in seg['corners']:
            if dim == 0:
                aug_polys.append(Polygon([(x[0] - seg['beginning'] + pos, x[1]) for x in poly_corners.exterior.coords]))
            else:
                aug_polys.append(Polygon([(x[0], x[1] - seg['beginning'] + pos) for x in poly_corners.exterior.coords]))
        pos += seg['size']
    
    #cv2.imwrite(str(train_dir / (Path(other_img_path).stem + "_aug.png")), aug_image)
    #print('aug_gcircs',len(aug_gcircs),aug_gcircs)
    #print('gcircs',len(gcircs),gcircs)
    
    return aug_image, aug_polys

def smart_grid_shuffle(img, polys: list[Polygon], img_size_wh):
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

def homogeneous_mat_transform(img, polys: list[Polygon], img_size_wh, M: Mat, background_color = [0, 0, 0], border_type = cv2.BORDER_CONSTANT):
    if M.shape[0] == 2:
        M = np.vstack([M, np.array([0, 0, 1])])
    
    img = cv2.warpPerspective(img, M, img_size_wh, borderMode=border_type, borderValue=background_color)
    polys = [transform(p, lambda x: np.array(apply_homography(x, M, convert_to_int=False))) for p in polys]
    
    return img, polys

def poly_label_dropout(img: Mat, polys: list[Polygon], draw_color: tuple = ()):
    
    pi = random.randrange(0, len(polys))
    
    if len(draw_color) != 3:
        c = polys[pi].centroid
        # Sample color from poly centroid in img
        draw_color = [int(x) for x in img[int(c.x), int(c.y)]]
    
    img = rasterize_polys(img, [inflate_poly(polys[pi], 0.2)], draw_color)
    polys.pop(pi)
    
    return img, polys