from pathlib import Path
import random
import cv2

from utils import *

def main():
    root_dir = Path(__file__).resolve().parent
    output_folder = create_dir_if_not_exists(root_dir / 'output/pt-seg')
    eval_folder = create_dir_if_not_exists(output_folder / 'eval')
    to_rect_output_folder = create_dir_if_not_exists(root_dir / 'output/to-rect')
    
    pred_img_paths = [Path(x) for x in get_files_from_folders_with_ending([eval_folder], '_pred.png')]
    for pred_img_path in pred_img_paths:
        img = cv2.imread(str(pred_img_path),0)
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_pyr_img.png'), img)

        # Apply edge detection
        edges = cv2.Canny(img, threshold1=1, threshold2=100)
        
        print(edges.shape)
        print(edges)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        eps = 0.03
        # draw the approximated contour on the image
        output = img.copy()
        #output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        
        # approximate the contour
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            #approx = [x for x in approx if len(x) >= 3 & len(x) <= 6]
            # print(approx)
            # print(approx.num_pts)
            # print([len(x[0]) for x in approx])
            if len(approx) == 4:
                cv2.drawContours(output, [approx], -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
        text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
        # cv2.putText(output, text, (0, 0 - 15), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.9, (0, 255, 0), 2)
        # show the approximated contour image
        print("[INFO] {}".format(text))
        cv2.imshow("Approximated Contour", output)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    main()