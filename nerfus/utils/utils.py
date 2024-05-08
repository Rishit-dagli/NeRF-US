import os

import gdown
import torch


def get_gaussian_kernel1d(kernel_size: int, sigma: float):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def download_pt():
    url = "https://drive.google.com/uc?id=1iioFQrH8cCmYxjLSoBb6DHHP7pP4QnIp"
    os.makedirs("./pt", exist_ok=True)
    download_path = str("./pt/nbusters.ckpt")
    gdown.download(url, output=download_path)
    return download_path
