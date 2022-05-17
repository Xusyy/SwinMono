import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class RelateLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
      
    def forward(self, depth_est, depth_gt):
        mask = depth_gt > 0
        depth_gt = depth_gt * torch.mean(depth_est[mask]) / torch.mean(depth_gt[mask])
        depth_gt[depth_gt < 0] = 0

        valid_mask = depth_gt > 1.0
        
        diff_log = torch.log(depth_gt[valid_mask]) - torch.log(depth_est[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

