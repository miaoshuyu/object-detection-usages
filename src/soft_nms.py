# coding:utf-8
'''
softnms explanation:
https://blog.csdn.net/diligent_321/article/details/85859462
https://blog.csdn.net/weixin_41665360/article/details/99818073
'''
import numpy as np


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
        xx2 = np.maximum(x2[i], x2[pos:])
        yy1 = np.maximum(y1[i], y1[pos:])
        yy2 = np.maximum(y2[i], y2[pos:])

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
    boxes = np.array(
        [[200, 200, 400, 400, 0.9],
         [220, 220, 420, 420, 0.8],
         [200, 240, 400, 440, 0.7],
         [240, 200, 440, 400, 0.6],
         [1, 1, 2, 2, 0.5]])

    print(soft_nms(boxes, method=3))
