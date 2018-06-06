import numpy as np
import torch
from torch import nn
from test_utils import gaussian_log_pdf

def binary_crossentropy_loss(x_true, x_distr, b):
    K = x_distr.shape[0] // x_true.shape[0]
    b = b.view(x_true.shape[0], 1, x_true.shape[1])
    x_true = x_true.view(x_true.shape[0], 1, x_true.shape[1])
    x_distr = x_distr.view(x_true.shape[0], K, x_true.shape[2])
    loss = b * (x_true * x_distr.clamp(min=1e-7).log() + (1 - x_true) * (1 - x_distr).clamp(min=1e-7).log())
    loss = loss.sum(dim=2)
    return loss

def gaussian_loss(x_true, x_distr, b):
    K = x_distr.shape[0] // x_true.shape[0]
    n, d = x_true.shape
    mu, sigma = x_distr[:, :d].view(n, K, d), x_distr[:, d:].view(n, K, d)
    sigma = torch.exp(sigma).clamp(min=1e-5)
    samples = x_true.view(n, 1, d)
    b = b.view(n, 1, d)
    loss = b * (-sigma.log() - np.log(2 * np.pi) - (((samples - mu) / sigma) ** 2) / 2)
    return loss.sum(dim=-1)