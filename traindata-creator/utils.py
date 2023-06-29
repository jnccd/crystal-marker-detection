from typing import Literal
import cv2
from cv2 import Mat
import numpy as np
from sympy import Polygon

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

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]

def resize_and_pad(img: Mat, desired_size: int):
    old_size = img.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    rimg = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    brimg = cv2.copyMakeBorder(rimg, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return brimg, new_size, top, left

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
    
def segment_img_between_poly_labels(img, polys, dim: Literal[0,1], collage_padding = 5):
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
            segments.append({   'end': end,
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