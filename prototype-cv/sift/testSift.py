import random
import cv2
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent
border_marker_img_path = root_dir/'sift-base'/'border-marker-scaled.png'
in_img_marker_img_path = root_dir/'sift-base'/'in-img-marker-scaled.png'
test_img_path = Path("N:\Downloads\Archives\FabioBilder\\the_good_pics_for_sift\DSC_3741.JPG")

img1 = cv2.imread(str(in_img_marker_img_path))
img2 = cv2.imread(str(test_img_path))

size_mult = 2

img2 = cv2.resize(img2, dsize=(600*size_mult, 400*size_mult), interpolation=cv2.INTER_CUBIC)

cv2.imshow('in img',img2)

# background_col = np.array([151, 164, 186])
# foreground_col = np.array([150, 139, 127])

imgg = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#imgr = cv2.inRange(img2, foreground_col, background_col)
imgg = cv2.adaptiveThreshold(imgg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,50*size_mult+1,2)
imgg = 255-imgg
#imgg = cv2.GaussianBlur(imgg,(3,3),0)
final_img = imgg

cv2.imshow('Thresholded img',final_img)
#cv2.waitKey(0)

orb = cv2.ORB_create()
sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

kpl, desl = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(final_img,None)

matches = bf.match(desl,des2)
matches = sorted(matches, key = lambda x:x.distance)

n = len(matches)
print(n)

good_matches = matches[:int(n/2)]
readable_good_matches = [(kpl[dmatch.queryIdx].pt, kp2[dmatch.trainIdx].pt) for dmatch in good_matches]

print(readable_good_matches)
long_img1 = cv2.copyMakeBorder(
        img1,
        top=0,
        bottom=final_img.shape[0] - img1.shape[0],
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
print(long_img1.shape, final_img.shape)
img3 = cv2.hconcat([long_img1, final_img])
for readable_good_match in readable_good_matches:
    cv2.line(img3, 
             (int(readable_good_match[0][0]), int(readable_good_match[0][1])), 
             (int(readable_good_match[1][0] + img1.shape[1]), int(readable_good_match[1][1])), 
             (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
             6)
# img3 = cv2.drawMatches(img1, kpl, final_img, kp2, matches[:int(n/2)], final_img, flags=2)

print_img = img3
cv2.imwrite(str(root_dir / 'full_sift_matches.png'),print_img)
cv2.imshow('SIFT',print_img)
cv2.waitKey(0)
