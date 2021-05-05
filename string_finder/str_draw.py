import torch as t
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc



def str_draw(draw_im, imgid, centers, norms, locs_lf, locs_rt, p_rowids, img=None, dpi=150, im_size=32):

    im_mask = imgid.eq(draw_im).cpu().numpy()

    mag = 25
    ax_max = im_size * mag
    h_size = 0.01 * mag * 36

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')
    ax.set_ylim(ax_max, 0)
    ax.set_xlim(0, ax_max)
    plt.axis('off')

    palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
    colors = t.arange(norms.size(0))[:, None].mul(20).float().mul(palette).fmod(255).div(255)
    colors = colors.numpy()
    colors[:, 3] = 1

    seg_centers_ = centers.clone().cpu().numpy() * mag
    norms_ = norms.clone().cpu().numpy() * mag
    locs_lf_, locs_rt_ = locs_lf.clone().cpu().numpy() * mag, locs_rt.clone().cpu().numpy() * mag
    p_rowids_ = p_rowids.clone().cpu().numpy()


    p_vecs = seg_centers_[p_rowids_] - seg_centers_
    p_colors = colors[p_rowids_]

    p_vecs = p_vecs[im_mask]
    p_colors = p_colors[im_mask]


    seg_centers_ = seg_centers_[im_mask]
    norms_ = norms_[im_mask]
    locs_lf_, locs_rt_ = locs_lf_[im_mask], locs_rt_[im_mask]
    colors = colors[im_mask]


    ax.quiver(seg_centers_[:, 1], seg_centers_[:, 0], p_vecs[:, 1], p_vecs[:, 0], angles='xy', units='xy',
              scale=1, width=0.01 * mag, headwidth=h_size, headlength=h_size + 2, headaxislength=h_size + 1, color=p_colors)

    # ax.quiver(seg_centers_[:, 1], seg_centers_[:, 0], norms_[:, 1], norms_[:, 0], angles='xy', units='xy',
    #           scale=1, width=0.01 * mag, headwidth=h_size, headlength=h_size + 2, headaxislength=h_size + 1, color=colors)
    #
    # ax.scatter(seg_centers_[:, 1], seg_centers_[:, 0], s=mag, alpha=1, marker="x", color=colors)
    # ax.scatter(locs_lf_[:, 1], locs_lf_[:, 0], s=mag, alpha=1, marker=".", color=colors)
    # ax.scatter(locs_rt_[:, 1], locs_rt_[:, 0], s=mag, alpha=1, marker=".", color=colors)

    locs_lf_ = np.stack([locs_lf_[:, 1], locs_lf_[:, 0]], axis=1)
    locs_rt_ = np.stack([locs_rt_[:, 1], locs_rt_[:, 0]], axis=1)

    lc = mc.LineCollection(list(zip(locs_lf_, locs_rt_)), linewidths=.1 * mag, color=colors)
    ax.add_collection(lc)

    topil = transforms.ToPILImage()
    img = topil(img)
    img = img.resize([ax_max, ax_max], resample=0)
    plt.imshow(img)

    plt.show(block=False)
