'''
Blog:https://blog.csdn.net/qq_31622015/article/details/103364467
'''

import torch


def giou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)

    # overlap
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps

    # ious
    ious = inters / uni

    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)

    # enclose erea
    enclose = ew * eh + eps

    giou = ious - (enclose - uni) / enclose

    loss = 1 - giou

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss


if __name__ == '__main__':
    pred_bboxes = torch.tensor([[15, 18, 47, 60],
                                [50, 50, 90, 100],
                                [70, 80, 120, 145],
                                [130, 160, 250, 280],
                                [25.6, 66.1, 113.3, 147.8]], dtype=torch.float)
    gt_bbox = torch.tensor([[70, 80, 120, 150]], dtype=torch.float)
    print(giou_loss(pred_bboxes, gt_bbox))
