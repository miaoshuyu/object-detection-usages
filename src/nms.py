# coding: utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

'''
NMS explanation: https://zhuanlan.zhihu.com/p/54709759
'''

import numpy as np


def nms(dets, thres):
    '''
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
    :param thres: for example 0.5
    :return: the rest ids of dets
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, 4]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(xx2 - xx1 + 1.0, 0.0)
        h = np.maximum(yy2 - yy1 + 1.0, 0.0)

        inters = w * h
        unis = areas[i] + areas[order[1:]] - inters
        ious = inters / unis

        inds = np.where(ious <= thres)[0]  # return the rest boxxes whose iou<=thres

        order = order[
            inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3

    return keep

def soft_nms(dets, iou_thresh=0.3, sigma=0.5, thresh=0.001, method=2):
    '''
    https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
    :param dets: [[x1, y1, x2, y2, score]，[x1, y1, x2, y2, score]，[x1, y1, x2, y2, score]]
    :param iou_thresh: iou thresh
    :param sigma: std of gaussian
    :param thresh: the last score thresh
    :param method: 1、linear 2、gaussian 3、originl nms
    :return: keep bboxes
    '''
    N = dets.shape[0]  # the size of bboxes
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        temp_box = dets[i, :4]
        temp_score = dets[i, 4]
        temp_area = areas[i]
        pos = i + 1

        if i != N - 1:
            maxscore = np.max(dets[:, 4][pos:])
            maxpos = np.argmax(dets[:, 4][pos:])
        else:
            maxscore = dets[:, 4][-1]
            maxpos = -1

        if temp_score < maxscore:
            dets[i, :4] = dets[maxpos + i + 1, :4]
            dets[maxpos + i + 1, :4] = temp_box

            dets[i, 4] = dets[maxpos + i + 1, 4]
            dets[maxpos + i + 1, 4] = temp_score

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = temp_area

        xx1 = np.maximum(x1[i], x1[pos:])
        xx2 = np.minimum(x2[i], x2[pos:])
        yy1 = np.maximum(y1[i], y1[pos:])
        yy2 = np.minimum(y2[i], y2[pos:])

        w = np.maximum(xx2 - xx1 + 1.0, 0.)
        h = np.maximum(yy2 - yy1 + 1.0, 0.)

        inters = w * h
        ious = inters / (areas[i] + areas[pos:] - inters)

        if method == 1:
            weight = np.ones(ious.shape)
            weight[ious > iou_thresh] = weight[ious > iou_thresh] - ious[ious > iou_thresh]
        elif method == 2:
            weight = np.exp(-ious * ious / sigma)
        else:
            weight = np.ones(ious.shape)
            weight[ious > iou_thresh] = 0

        dets[pos:, 4] = dets[pos:, 4] * weight

    inds = np.argwhere(dets[:, 4] > thresh)
    keep = inds.astype(int)[0]

    return keep


if __name__ == '__main__':
    dets = np.array([[100, 120, 170, 200, 0.98],
                     [20, 40, 80, 90, 0.99],
                     [20, 38, 82, 88, 0.96],
                     [200, 380, 282, 488, 0.9],
                     [19, 38, 75, 91, 0.8]])

    print(nms(dets, 0.5))
    print(soft_nms(dets, method=3))
