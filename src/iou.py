"""
IOU explanation： https: // zhuanlan.zhihu.com/p/47189358
"""

import numpy as np


def get_iou(pred_bbox, gt_bbox):
    '''
    :param pred_bbox: [x1, y1, x2, y2]
    :param gt_bbox:  [x1, y1, x2, y2]
    :return: iou
    '''

    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.)
    ih = np.maximum(iymax - iymin + 1.0, 0.)

    inters = iw * ih

    # uni=s1+s2-inters
    uni = (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0) + \
          (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0) - inters

    iou = inters / uni

    return iou


def get_max_iou(pred_bboxes, gt_bbox):
    '''
    :param pred_bboxs: [[x1, y1, x2, y2] [x1, y1, x2, y2],,,]
    :param gt_bbox: [x1, y1, x2, y2]
    :return:
    '''
    ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
    iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
    ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
    iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])

    iws = np.maximum(ixmax - ixmin + 1.0, 0.)
    ihs = np.maximum(iymax - iymin + 1.0, 0.)

    inters = iws * ihs

    unis = (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.0) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.0) + (
            gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0) - inters

    ious = inters / unis
    max_iou = np.max(ious)
    max_index = np.argmax(ious)

    print(ious)
    return ious, max_iou, max_index


if __name__ == "__main__":
    # top-left: <50, 50>, bottom-down: <90, 100>, <x-axis, y-axis>
    pred_bbox = np.array([50, 50, 90, 100])
    gt_bbox = np.array([70, 80, 120, 150])
    print(get_iou(pred_bbox, gt_bbox))

    pred_bboxes = np.array([[15, 18, 47, 60],
                            [50, 50, 90, 100],
                            [70, 80, 120, 145],
                            [130, 160, 250, 280],
                            [25.6, 66.1, 113.3, 147.8]])
    gt_bbox = np.array([70, 80, 120, 150])

    print(get_max_iou(pred_bboxes, gt_bbox))
