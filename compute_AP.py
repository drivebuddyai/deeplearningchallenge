"""
Company: DriveBuddyAi 
Created on 15/12/20
@author hemantc

Description:
Compute mAP and AP for list of classes using pickle format
Save mAP value in a csv file at the same location at prd.pickle file

"""
import os
import pickle
import time
from collections import Counter
from os.path import join, split

import pandas as pd
import torch

root = split(__file__)[0]

os.path.abspath(root)

classes = ['animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'motorbike', 'person', 'rider', 'truck', 'tempo']


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=9
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
        [mAP , [AP per class]]
    """
    
    # list storing all AP for respective classes
    average_precisions = []
    
    # used for numerical stability later on
    epsilon = 1e-6
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
    
    average_precisions = [float(iter) for iter in average_precisions]
    # print(f"average_precisions :{len(average_precisions)}")
    # print(f"average_precisions :{average_precisions}")
    
    return [sum(average_precisions) / len(average_precisions), average_precisions]


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    
    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] // 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] // 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] // 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] // 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] // 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] // 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] // 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] // 2
    
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def compute_mAP(gt_path, prd_path):
    gt = pickle.load(open(gt_path, 'rb'))
    prd = pickle.load(open(prd_path, 'rb'))
    
    gt_list = []
    prd_list = []
    
    # Convert gt and prd into list for AP calculation
    for _gt in gt:
        file_id = _gt[0]
        for iter in _gt[1]:
            gt_list.append([file_id, iter[0], iter[2], iter[3][0], iter[3][1], iter[3][2], iter[3][3]])
    
    for _prd in prd:
        file_id = _prd[0]
        for iter in _prd[1]:
            prd_list.append([file_id, iter[0], iter[2], iter[3][0], iter[3][1], iter[3][2], iter[3][3]])
    
    res = []
    iou_list = [round(iter * 0.1, 2) for iter in range(1, 10)]
    for iter in iou_list:
        res.append([iter, mean_average_precision(prd_list, gt_list, iou_threshold=iter)])
    print(f"Result :{res}")
    
    master_mAP = []
    for index, iter in enumerate(res):
        print(f"iter :{iter}")
        _list = [iter[0], iter[1][0]]
        for _iter in iter[1][1]:
            _list.append(_iter)
        master_mAP.append(_list)
    col = ['IOU', 'mAP'] + classes
    
    master_mAP = pd.DataFrame(master_mAP, columns=col)
    _name = gt_path.split('/')[-2] + '_' + prd_path.split('/')[-2]
    master_mAP.to_csv(join(split(prd_path)[0], f'mAP_{_name}.csv'))
    
    return master_mAP


GT_PICKLE = "data/train/gt.pickle"
PRD_PICKLE = ""

if __name__ == "__main__":
    start = time.time()
    compute_mAP(GT_PICKLE, PRD_PICKLE)
    print(f"Time taken = {time.time() - start}")
