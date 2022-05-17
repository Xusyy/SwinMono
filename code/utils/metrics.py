import torch
import numpy as np
import math

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}


def align_img(pred_depth, ground_depth, c):
    """
    align predicted depth maps with ground truth.

    :param pred_depth: torch.foat32
    :param ground_depth: torch.foat32
    :param c:
    :return: aligned predicted depth

    """
    pred_avg = torch.mean(pred_depth[c])
    pred_var = torch.var(pred_depth[c])
    gt_avg = torch.mean(ground_depth[c])
    gt_var = torch.var(ground_depth[c])

    # print('pred: ', pred_avg, pred_var, 'gt: ', gt_avg, gt_var)

    pred_depth_f = (pred_depth - pred_avg) * math.sqrt(gt_var / pred_var) + gt_avg

    pred_depth_f[pred_depth_f > 65535] = 65535
    pred_depth_f[pred_depth_f < 0] = 0

    # pred_aligned = pred_depth_f.astype(np.uint16)

    return pred_depth_f

def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval
    
    a = pred == 0
    b = gt_depth == 0
    c = gt_depth > 0
    if len(c) == 0:
        return None

    gt_depth = gt_depth * pred[c].mean() / gt_depth[c].mean()

    pred[a] = 1
    pred[b] = 1
    gt_depth[b] = 1

    pred_aligned = align_img(pred, gt_depth, c)
    pred_aligned[b] = 1
    pred_aligned[pred_aligned == 0] = 1
    pred_aligned[gt_depth == 0] = 1

    # gt_depth = gt_depth.astype(float)
    return pred[c], gt_depth[c]

