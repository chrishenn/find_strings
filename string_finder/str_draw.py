import torch as t
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc

import scipy.interpolate as sinterp



def str_draw(strs, img=None, linewidths=0.01, dpi=150, max_size=32):
    ## DO NOT USE THIS CODE - VIS CODE IS INLINE IN STRING_FINDER.PY

    ## strs = [locs[y,x] locs[y,x] norms[dy,dx] dev-frac]

    dev = strs.device
    mag = 25
    ax_max = max_size * mag

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')
    ax.set_ylim(ax_max, 0)
    ax.set_xlim(0, ax_max)
    plt.axis('off')

    ## NOT IN USE

    plt.show(block=False)
    print("no")
