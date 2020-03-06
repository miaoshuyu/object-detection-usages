'''
Blog:https://www.sohu.com/a/306064501_500659
'''

import torch
import math


def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    https://github.com/gurkirt/FPN.pytorch1.0/blob/ba9338a0ed511bd7b8659b57562bfad3788684a6/modules/box_utils.py#L117
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_anchors): the loss for each example.
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


if __name__ == '__main__':
    loss = torch.randn(3, 5)
    labels = torch.LongTensor([
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 4],
        [0, 0, 0, 3, 0]])
    print(loss)
    print(labels)
    out = hard_negative_mining(loss, labels)
    print(out)
