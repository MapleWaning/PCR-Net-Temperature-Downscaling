import torch
import torch.nn as nn
import torch.nn.functional as F


class PureMSELoss(nn.Module):
    def forward(self, pred, target, mask, guidance_map=None, input_dem=None):
        diff = (pred - target) ** 2
        valid = mask.sum() * pred.shape[1]
        return (diff * mask).sum() / (valid + 1e-6)


class HybridRefinementLoss_V2(nn.Module):
    def __init__(self, lambda_grad=5.0, alpha_terrain=2.0):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.alpha_terrain = alpha_terrain
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.raw_loss_station = torch.tensor(0.0)
        self.raw_loss_grad = torch.tensor(0.0)

    def gradient(self, x):
        channels = x.shape[1]
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(x, self.sobel_x.repeat(channels, 1, 1, 1), groups=channels)
        grad_y = F.conv2d(x, self.sobel_y.repeat(channels, 1, 1, 1), groups=channels)
        return grad_x, grad_y

    def forward(self, pred, target, mask, guidance_map, input_dem):
        slope = input_dem[:, 1:2, :, :]
        slope_dilated = F.max_pool2d(slope, kernel_size=7, stride=1, padding=3)
        weight = (1.0 + self.alpha_terrain * slope_dilated).repeat(1, pred.shape[1], 1, 1)

        # Station supervision is amplified in locally complex terrain.
        station_loss = (((pred - target) ** 2) * weight * mask).sum()
        station_loss = station_loss / (mask.sum() * pred.shape[1] + 1e-6)

        pred_grad_x, pred_grad_y = self.gradient(pred)
        with torch.no_grad():
            guide_grad_x, guide_grad_y = self.gradient(guidance_map)
        grad_loss = F.mse_loss(pred_grad_x, guide_grad_x) + F.mse_loss(pred_grad_y, guide_grad_y)

        self.raw_loss_station = station_loss.detach()
        self.raw_loss_grad = grad_loss.detach()
        return station_loss + self.lambda_grad * grad_loss
