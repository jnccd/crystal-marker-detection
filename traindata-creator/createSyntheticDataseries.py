import math
import cv2
from utils import *
from pathlib import Path
from shapely import LineString, Point, Polygon

target_size = 1280
num_pics_to_gen = 200
dataseries_name = 'test'

print('Loading paths...')
root_dir = Path(__file__).resolve().parent
dataseries_dir = create_dir_if_not_exists(root_dir / 'dataseries' / f'synth-{dataseries_name}', clear=True)
assets_dir = root_dir / 'raw/synthetic-builder-assets'
aruco_texture_path = assets_dir / 'in-img-marker.png'
foreground_textures_dir = assets_dir / 'foreground_textures'
background_textures_dir = assets_dir / 'background_textures'

print('Loading aruco_img...')
aruco_img = cv2.imread(str(aruco_texture_path))
aruco_img = cv2.cvtColor(aruco_img, cv2.COLOR_RGB2RGBA)
aruco_img_size = int(target_size/15)
aruco_img = cv2.resize(aruco_img, (aruco_img_size, aruco_img_size), interpolation=cv2.INTER_NEAREST)

print('Loading textures...')
back_textures = [cv2.imread(str(p)) for p in get_files_from_folders_with_ending([background_textures_dir], '.jpg')]#[:3]]
fore_textures = [cv2.imread(str(p), -1) for p in get_files_from_folders_with_ending([foreground_textures_dir], '.jpg')]#[:3]]

# Remapping for curve
print('Loading remapping curve...')
half_target_size = target_size / 2
curvature_height = 128
map_x = np.zeros((target_size + curvature_height*2, target_size), np.float32)
map_y = np.zeros((target_size + curvature_height*2, target_size), np.float32)
for y in range(target_size + curvature_height*2):
    for x in range(target_size):
        map_x[y, x] = x
        map_y[y, x] = y + math.sqrt((half_target_size)**2 - (x-half_target_size)**2) / half_target_size * curvature_height #int(128.0 * math.sin(3.14 * x / target_size))

for i in range(num_pics_to_gen):
    print(f'Generating image {i}...')
    img_size = (target_size, target_size)
    
    # Init pics
    back_img = center_crop_to_11_aspect_ratio(back_textures[i % len(back_textures)], target_size)
    fore_img = center_crop_to_11_aspect_ratio(fore_textures[i % len(fore_textures)], target_size)
    
    # Setup aruco marker pos
    aruco_marker_x = int(fore_img.shape[1]/2 - aruco_img.shape[1]/2) + random.randint(-int(target_size/10), int(target_size/10))
    aruco_marker_y = int(fore_img.shape[0]/2 - aruco_img.shape[0]/2) + random.randint(-int(target_size/10), int(target_size/10))
    
    # Build strip image
    strip_brightness = random.randrange(50, 100)
    strip_width = int(target_size/5)
    strip_opacity = 100
    strip_img = np.zeros((target_size, target_size) + (3,), dtype = np.uint8)
    strip_img = cv2.cvtColor(strip_img, cv2.COLOR_RGB2RGBA)
    strip_img[:, :, 3] = 0
    cv2.rectangle(strip_img, (0, int(target_size/2 - strip_width/2)), (target_size, int(target_size/2 + strip_width/2)), (strip_brightness, strip_brightness, strip_brightness, strip_opacity), -1)
    strip_img[aruco_marker_y:aruco_marker_y+aruco_img.shape[0], aruco_marker_x:aruco_marker_x+aruco_img.shape[1], 3] = aruco_img[:, :, 0].astype('float32') / 255 * strip_opacity
    #cv2.imwrite(str(root_dir / 'strip_img.png'), strip_img)
    
    # Add strip image to fore img
    fore_img = overlay_transparent_fore_alpha(fore_img, strip_img)
    fore_img = cv2.cvtColor(fore_img, cv2.COLOR_RGB2RGBA)
    fore_img[:, :, 3] = 255
    
    # Give fore_img extra headroom (literally)
    extra_embedding_height = curvature_height*2
    embedding_fore_img = np.zeros((target_size + extra_embedding_height, target_size) + (4,), np.float32)
    embedding_fore_img[curvature_height:target_size+curvature_height, 0:target_size] = fore_img
    # Curve fore img
    fore_img = cv2.remap(embedding_fore_img, map_x, map_y, cv2.INTER_LINEAR)
    #cv2.imwrite(str(root_dir / 'embedding_fore_img.png'), embedding_fore_img)
    
    # Create poly in fore img coord system
    label_poly = Polygon([(aruco_marker_x, aruco_marker_y), (aruco_marker_x + aruco_img_size, aruco_marker_y), (aruco_marker_x + aruco_img_size, aruco_marker_y + aruco_img_size), (aruco_marker_x, aruco_marker_y + aruco_img_size)])
    
    # --- Mat transform fore img ---------------------------------------------------------------------------------------------------------------
    mats = []
    # Perspective
    mats.append(create_random_persp_mat(img_size, perspective_strength=0.1))
    # Rotation
    mats.append(np.vstack([
        cv2.getRotationMatrix2D(
            (target_size/2, target_size/2), 
            random.randrange(-20, 20), 
            0.7), 
        np.array([0, 0, 1])]))
    final_mat = np.identity(3)
    mats.reverse()
    for mat in mats:
        final_mat = final_mat @ mat
    fore_img, mat_label_polys = homogeneous_mat_transform(fore_img, [label_poly], img_size, final_mat, border_type=cv2.BORDER_TRANSPARENT)
    # -------------------------------------------------------------------------------------------------------------------------------------------
    
    img = overlay_transparent_fore_alpha(back_img, fore_img)
    
    cv2.imwrite(str(dataseries_dir / f'{i}_in.png'), img)
    
    vertices_per_obj = [[(int(point[0]), int(point[1])) for point in poly.exterior.coords[:-1]] for poly in mat_label_polys]
    write_textfile(str(vertices_per_obj), dataseries_dir / f'{i}_vertices.txt')