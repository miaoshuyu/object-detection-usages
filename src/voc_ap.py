'''
Blog:https://zhuanlan.zhihu.com/p/48992451、https://zhuanlan.zhihu.com/p/94597205
'''

import numpy as np


def voc_ap(rec, prec, use_07_metric=True):
    '''
    https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/eval.py#L364
    :param prec: [n]
    :param rec: [n]
    :param use_07_metric: use_o7_metric or use_10_metric
    :return: ap
    '''
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        #
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


if __name__ == '__main__':
    precision = np.array(
        [1 / 1, 2 / 2, 2 / 3, 2 / 4, 2 / 5, 3 / 6, 4 / 7, 4 / 8, 4 / 9, 4 / 10, 5 / 11, 5 / 12, 5 / 13, 5 / 14, 5 / 15,
         6 / 16, 6 / 17, 6 / 18, 6 / 19, 6 / 20], dtype=np.float)
    recall = np.array(
        [1 / 6, 2 / 6, 2 / 6, 2 / 6, 2 / 6, 3 / 6, 4 / 6, 4 / 6, 4 / 6, 4 / 6, 5 / 6, 5 / 6, 5 / 6, 5 / 6, 5 / 6, 6 / 6,
         6 / 6, 6 / 6, 6 / 6, 6 / 6], dtype=np.float)

    print(voc_ap(recall, precision, False))
