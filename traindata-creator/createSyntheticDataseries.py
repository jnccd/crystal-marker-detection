import cv2
from utils import *
from pathlib import Path
from shapely import LineString, Point, Polygon

target_size = 1280
num_pics_to_gen = 200
dataseries_name = 'test'

root_dir = Path(__file__).resolve().parent
dataseries_dir = create_dir_if_not_exists(root_dir / 'dataseries' / f'synth-{dataseries_name}', clear=True)
assets_dir = root_dir / 'raw/synthetic-builder-assets'
aruco_texture_path = assets_dir / 'in-img-marker.png'
foreground_textures_dir = assets_dir / 'foreground_textures'
background_textures_dir = assets_dir / 'background_textures'

aruco_img = cv2.imread(str(aruco_texture_path))
aruco_img = cv2.cvtColor(aruco_img, cv2.COLOR_RGB2RGBA)
aruco_img_size = int(target_size/10)
aruco_img = cv2.resize(aruco_img, (aruco_img_size, aruco_img_size), interpolation=cv2.INTER_NEAREST)

back_textures = [cv2.imread(str(p)) for p in get_files_from_folders_with_ending([background_textures_dir][:3], '.jpg')]
fore_textures = [cv2.imread(str(p), -1) for p in get_files_from_folders_with_ending([foreground_textures_dir][:3], '.jpg')]

for i in range(num_pics_to_gen):
    img_size = (target_size, target_size)
    
    back_img = center_crop_to_11_aspect_ratio(back_textures[i % len(back_textures)], target_size)
    fore_img = center_crop_to_11_aspect_ratio(fore_textures[i % len(fore_textures)], target_size)
    
    # Add aruco img to fore img and add alpha channel
    #fore_img = overlay_transparent(fore_img, aruco_img, int(fore_img.shape[1]/2 - aruco_img.shape[1]/2), int(fore_img.shape[0]/2 - aruco_img.shape[0]/2))
    #fore_img = cv2.addWeighted(fore_img,0.4,aruco_img,0.1,0)
    fore_img = cv2.cvtColor(fore_img, cv2.COLOR_RGB2RGBA)
    aruco_marker_x = int(fore_img.shape[1]/2 - aruco_img.shape[1]/2) + random.randint(-int(target_size/10), int(target_size/10))
    aruco_marker_y = int(fore_img.shape[0]/2 - aruco_img.shape[0]/2) + random.randint(-int(target_size/10), int(target_size/10))
    fore_img = combine_two_color_images(
        fore_img, 
        aruco_img, 
        aruco_marker_x,
        aruco_marker_y, 
        alpha=0.25)
    #cv2.imwrite(str(root_dir / 'test2.png'), fore_img)
    # Set fore img alpha to 255
    fore_img[:, :, 3] = 255
    
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
    #print(fore_img.shape)
    # -------------------------------------------------------------------------------------------------------------------------------------------
    
    img = overlay_transparent(back_img, fore_img, int(back_img.shape[1]/2 - fore_img.shape[1]/2), int(back_img.shape[0]/2 - fore_img.shape[0]/2))
    
    cv2.imwrite(str(dataseries_dir / f'{i}_in.png'), img)
    
    vertices_per_obj = [[(int(point[0]), int(point[1])) for point in poly.exterior.coords[:-1]] for poly in mat_label_polys]
    write_textfile(str(vertices_per_obj), dataseries_dir / f'{i}_vertices.txt')
    
    #exit()