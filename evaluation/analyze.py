import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
from matplotlib import pyplot as plt

import numpy as np

num_classes = 1
bbox_inflation = 0

def analyze(
    run_or_testdata: str,
    ):
    
    root_dir = Path(__file__).resolve().parent
    if os.path.exists(run_or_testdata) and os.path.isdir(run_or_testdata):
        testdata_path = Path(run_or_testdata)
    else:
        runs_dir = root_dir / '../cmd_tf/runs'
        run_dir = runs_dir / f'{run_or_testdata}'
        testdata_path = run_dir / 'test'
    if len(get_files_from_folders_with_ending([testdata_path], '_input.png')) == 0:
        testdata_path = testdata_path / 'test'
        if os.path.exists(testdata_path) and \
            os.path.isdir(testdata_path) and \
            len(get_files_from_folders_with_ending([testdata_path], '_input.png')) > 0:
            testdata_path = testdata_path
        else:
            print('Couldnt find test data folder!')
            sys.exit(1)
    eval_path = create_dir_if_not_exists(testdata_path / 'evals')
        
    ins = get_files_from_folders_with_ending([testdata_path], '_input.png')
    bbox_target_outs = get_files_from_folders_with_ending([testdata_path], '_target_output.txt')
    pic_target_outs = get_files_from_folders_with_ending([testdata_path], '_target_output.png')
    bbox_network_outs = get_files_from_folders_with_ending([testdata_path], '_network_output.txt')
    pic_network_outs = get_files_from_folders_with_ending([testdata_path], '_network_output.png')
    
    if len(bbox_network_outs) == 0:
        # Convert pic outs to bbox outs
        
        target_bboxes_per_img = []
        for target_out in pic_target_outs:
            img = cv2.imread(str(target_out), cv2.IMREAD_GRAYSCALE)
            target_bboxes_per_img.append(cluster_boxes_from_grayscale_img(img,target_out,True))
            
        network_bboxes_per_img = []
        for network_out in pic_network_outs:
            img = cv2.imread(str(network_out), cv2.IMREAD_GRAYSCALE)
            network_bboxes_per_img.append(cluster_boxes_from_grayscale_img(img,network_out,True))
    else:
        # Read bbox outs
        
        target_bboxes_per_img = []
        for target_out, in_img in zip(bbox_target_outs, ins):
            with open(target_out, 'r') as file:
                target_out_bbox_lines = file.readlines()
            bboxes_xyxy = []
            for line in target_out_bbox_lines:
                if line.__contains__(' '):
                    bboxes_xyxy.append(tuple([float(x) for x in line.split(' ')]))
            target_bboxes_per_img.append(bboxes_xyxy)
            
            cv2.imwrite(str(target_out)+'.bbox_test.png', draw_bboxes(cv2.imread(str(in_img), cv2.IMREAD_GRAYSCALE), bboxes_xyxy))
        
        network_bboxes_per_img = []
        for network_out in bbox_network_outs:
            with open(network_out, 'r') as file:
                network_out_bbox_lines = file.readlines()
            bboxes_xyxy = []
            for line in network_out_bbox_lines:
                if line.__contains__(' '):
                    bboxes_xyxy.append(tuple([float(x) for x in line.split(' ')]))
            network_bboxes_per_img.append(bboxes_xyxy)
            
            cv2.imwrite(str(network_out)+'.bbox_test.png', draw_bboxes(cv2.imread(str(in_img), cv2.IMREAD_GRAYSCALE), bboxes_xyxy))
    
    # print('target_bboxes',target_bboxes_per_img)
    # print('network_bboxes',network_bboxes_per_img)
    # print(iou_between_bboxes((1,1,5,5), (1,1,5,5)))
    
    # Match target to pred bboxes for each image
    flat_best_iou_matches = []
    for target_boxes, network_boxes in zip(target_bboxes_per_img, network_bboxes_per_img):
        # TODO: Use something better than n^2 matching
        for target_box in target_boxes:
            max_iou_match = 0
            for network_box in network_boxes:
                iou = iou_between_bboxes(network_box, target_box)
                if iou > max_iou_match:
                    max_iou_match = iou
            flat_best_iou_matches.append(max_iou_match)
    print('flat_best_iou_matches',flat_best_iou_matches)
    
    # Match pred to target bboxes for each image and build mAP table
    mAP_table = []
    for i, (target_boxes, network_boxes) in enumerate(zip(target_bboxes_per_img, network_bboxes_per_img)):
        for nbi, network_box in enumerate(network_boxes):
            max_match_iou = 0
            max_match_target_box = ()
            max_match_target_box_index = -1
            for tbi, target_box in enumerate(target_boxes):
                iou = iou_between_bboxes(network_box, target_box)
                if iou > max_match_iou:
                    max_match_iou = iou
                    max_match_target_box = target_box
                    max_match_target_box_index = tbi + 1000000*i
            network_box_index = nbi + 1000000*i # It just works
            
            # Get pred conf
            if len(network_box) >= 5:
                #print('got conf from bbox')
                conf = network_box[4]
            else:
                conf = max_iou_match
            
            mAP_table.append({
                'img_index': i,
#                'dt': network_box,
                'dt_index': network_box_index,
                'conf': conf,
                'iou': max_match_iou,
#                'gt': max_match_target_box,
                'gt_index': max_match_target_box_index,
                })
    
    total_gts = sum([len(target_boxes) for target_boxes in target_bboxes_per_img])
    voc_2007_recall_points = [x/10 for x in range(0, 11, 1)]
    
    # Compute VOC mAPs
    voc2007_mAP, _, _ = compute_mAP(mAP_table, total_gts, voc_2007_recall_points)
    print('voc2007_mAP:',voc2007_mAP)
    voc2010_mAP, _, _ = compute_mAP(mAP_table, total_gts)
    print('voc2010_mAP:',voc2010_mAP)
    
    # Compute COCO mAP
    coco_mAPs = []
    ious = [x/20 for x in range(10, 20, 1)]
    coco_recall_points = [x/10 for x in range(0, 101, 1)]
    pr_curves_per_iou = []
    for iou in ious:
        mAP, rp, pp = compute_mAP(mAP_table, total_gts, coco_recall_points, iou)
        pr_curves_per_iou.append({'rp':rp,'pp':pp})
        coco_mAPs.append(mAP)
    coco_mAP = np.mean(coco_mAPs)
    print('coco_mAP:',coco_mAP)
    
    # Plot PR Curves
    for i in range(len(ious)):
        rp = pr_curves_per_iou[i]['rp']
        pp = pr_curves_per_iou[i]['pp']
        
        plt.clf()
        plt.title(f"PR Curve for IoU:{ious[i]}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)
        plt.plot(rp, pp)
        
        iou_str = str(ious[i])
        if len(iou_str) <= 3:
            iou_str = iou_str + '0'
        plt.savefig(eval_path / f'eval_PR_Curve_{iou_str}.pdf', dpi=100)
    
    # Write out said metrics
    eval_dict_path = eval_path / 'evals.json'
    with open(eval_dict_path, "w") as text_file:
        text_file.write(json.dumps({
            'avg_iou':float(np.average(flat_best_iou_matches)),
            'voc2007_mAP':float(voc2007_mAP),
            'voc2010_mAP':float(voc2010_mAP),
            'coco_mAP':float(coco_mAP),
            'version':1.1,
            }, indent=4))

def compute_mAP(mAP_table, total_gts, recall_points = None, IoU = 0.5):
    mAP_table = sorted(mAP_table, key=lambda x: -x['conf'])
    
    # Decide TP/FP cases
    taken_gt_boxes = []
    for entry in mAP_table:
        if entry['iou'] > IoU and not entry['gt_index'] in taken_gt_boxes:
            entry['tpfp'] = True
            taken_gt_boxes.append(entry['gt_index'])
        else:
            entry['tpfp'] = False
    
    # Build Acc TP, Acc FP, Precision and Recall
    acc_tp = 0
    acc_fp = 0
    for entry in mAP_table:
        if entry['tpfp']:
            acc_tp += 1
        else:
            acc_fp += 1
            
        entry['acc_tp'] = acc_tp
        entry['acc_fp'] = acc_fp
        entry['precision'] = acc_tp / (acc_tp + acc_fp)
        entry['recall'] = acc_tp / total_gts if total_gts > 0 else 0
    
    # Generate PR curve values
    mAP_table = sorted(mAP_table, key=lambda x: x['recall'])
    rp = [x['recall'] for x in mAP_table]
    pp = [x['precision'] for x in mAP_table]
    
    # Add some extra values to make the curve go to 1
    if len(rp) == 0 or rp[-1] < 1:
        # Append point downwards
        rp.append(rp[-1] if len(rp) > 0 else 0)
        pp.append(0)
        
        # Append point at (1, 0)
        rp.append(1)
        pp.append(0)
    
    if recall_points == None:
        APs = []
        last_r = 0
        for i in range(len(rp)):
            next_precisions = pp[i:]
            if len(next_precisions) == 0:
                max_next_precisions = 0
            else:
                max_next_precisions = max(next_precisions)
                
            r_diff = rp[i] - last_r
            AP = max_next_precisions * r_diff
            #print(rp[i], r_diff, max_next_precisions, AP)
            
            APs.append(AP)
            last_r = rp[i]
        mAP = np.sum(APs)
    else:
        APs = []
        for i in recall_points:
            recall_breakpoint_index = -1
            for j in range(len(rp)):
                if rp[j] > i:
                    recall_breakpoint_index = j
                    break
            
            if recall_breakpoint_index == -1:
                AP = 0
            else:
                AP = max(pp[j:])
            
            APs.append(AP)
        #print('recall_points:',recall_points)
        #print('APs:',APs)
        mAP = np.mean(APs)
    
    return mAP, rp, pp

def iou_between_bboxes(xyxy_bbox_a, xyxy_bbox_b):
    # xyxy_bbox = (0: x_min, 1: y_min, 2: x_max, 3: y_max)
    x_overlap = max(0, min(xyxy_bbox_a[2], xyxy_bbox_b[2]) - max(xyxy_bbox_a[0], xyxy_bbox_b[0]))
    y_overlap = max(0, min(xyxy_bbox_a[3], xyxy_bbox_b[3]) - max(xyxy_bbox_a[1], xyxy_bbox_b[1]))
    overlap_area = x_overlap * y_overlap
    
    a_w = xyxy_bbox_a[2] - xyxy_bbox_a[0]
    a_h = xyxy_bbox_a[3] - xyxy_bbox_a[1]
    b_w = xyxy_bbox_b[2] - xyxy_bbox_b[0]
    b_h = xyxy_bbox_b[3] - xyxy_bbox_b[1]
    
    a_area = a_w * a_h
    b_area = b_w * b_h
    total_area = a_area + b_area - overlap_area
    
    return overlap_area / total_area
    
def cluster_boxes_from_grayscale_img(img, target_out='', print_bbox_img=False):
    global bbox_inflation
    
    bboxes_xyxy = []
    img_h, img_w = img.shape[:2]
    if print_bbox_img:
        sanity_check_img = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    assert img is not None, "file could not be read!"
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,_ = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x -= bbox_inflation
        y -= bbox_inflation
        w += bbox_inflation*2
        h += bbox_inflation*2
        bboxes_xyxy.append((x,y,x+w,y+h))
        if print_bbox_img:
            cv2.rectangle(sanity_check_img,(x,y),(x+w,y+h),(0,255,0),2)
    if print_bbox_img:
        cv2.imwrite(str(target_out)+'.cluster_test.png', sanity_check_img)
    
    return bboxes_xyxy

def draw_bboxes(img,bboxes):
    img_h, img_w = img.shape[:2]
    sanity_check_img = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    for bbox in bboxes:
        cv2.rectangle(sanity_check_img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
    return sanity_check_img

if __name__ == '__main__':
    from utility import get_files_from_folders_with_ending, create_dir_if_not_exists
    
    parser = argparse.ArgumentParser(prog='analyze', description='Analyzes object detection and segmentation models by unifying the results as bboxes through pixel clustering.')
    parser.add_argument('-av','--analyze-valdata-from', type=str, default='', help='Set to a run name to analyze the validation data of or to a path to a folder containing validation data')
    args = parser.parse_args()
    
    analyze(args.analyze_valdata_from)
    