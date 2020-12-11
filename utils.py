"""A set of generically helpful utility methods and constants"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Which device to train the model on"""


def seed_everything(seed=42):
    """Seed everything to make the code more reproducable.

    This code is the same as that found from many public Kaggle kernels.

    Parameters
    ----------
    seed: int
        seed value to ues

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def display_output(v0, vk, dim1=28, dim2=28):
    """Dsplaying the original and reconstructed images for comparison

    Parameters
    ----------
    v0: Tensor
        the original image
    vk: Tensor
        the reconstructed image
    dim1: int
        number of pixels on first dimension for plotting
    dim2: int
        number of pixels on second dimension for plotting

    """
    print("Original (top) and Reconstructed (bottom)")
    img = make_grid(v0.view(v0.shape[0], 1, dim1, dim2).data)
    npimg = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.show()
    img = make_grid(vk.view(vk.shape[0], 1, dim1, dim2).data)
    npimg = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


def display_2d_repr(data, labels):
    """Display a 2d representation of the MNIST digits

    Parameters
    ----------
    data: Tensor
        2d representation of MNIST digits
    labels: list
        the label for each data point in data

    """

    digit_to_color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                      "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    xs = np.array([x[0] for x in data])
    ys = np.array([x[1] for x in data])

    fig, ax = plt.subplots()
    labels_to_show = labels[0:len(data)]
    for digit in range(10):
        ix = np.where(labels_to_show == digit)
        ax.scatter(xs[ix], ys[ix], c=digit_to_color[digit],
                   label=digit, marker=".")
    ax.legend()
    plt.show()
