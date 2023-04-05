import numpy as np
import math
import torch.nn as nn
from skimage.measure import compare_psnr
from pytorch_msssim import MS_SSIM, SSIM

def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


class MS_SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)

    def forward(self, output, target):
        return 1 - self.ms_ssim(output, target)


class SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1, size_average=True, channel=3)

    def forward(self, output, target):
        return 1 - self.ssim(output, target)


