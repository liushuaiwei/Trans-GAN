import torch.nn as nn
from pytorch_msssim import MS_SSIM, SSIM


class SSIM_Loss(nn.Module):
    def __init__(self, channel=3):
        super().__init__()
        self.ssim = SSIM(data_range=1, size_average=True, channel=channel)

    def forward(self, output, target):
        return 1 - self.ssim(output, target)