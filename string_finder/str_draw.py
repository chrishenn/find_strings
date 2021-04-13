import torch as t
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc

import scipy.interpolate as sinterp



def str_draw(strs, img=None, linewidths=0.01, dpi=150, max_size=32):
    ## strs = [locs[y,x] locs[y,x] norms[dy,dx] dev-frac]

    dev = strs.device
    mag = 25
    ax_max = max_size * mag

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')
    ax.set_ylim(ax_max, 0)
    ax.set_xlim(0, ax_max)
    plt.axis('off')

    if img is not None:
        topil = transforms.ToPILImage()
        if img.min() < -1e-4:
            img = img.add(1).div(2)
        img = topil(img.cpu())
        img = img.resize([ax_max, ax_max], resample=0)
        plt.imshow(img)

    strs = strs.clone()
    strs.add_(0.5).mul_(mag)

    locs_lf, locs_rt = strs[:,:2], strs[:, 2:4]
    vecs = locs_rt.sub( locs_lf )

    ### spl_locs include some deviation dev-frac
    dev_fracs = strs[:,-1]

    seg_centers = locs_rt.add(locs_lf).div(2)


    ##########
    dev_dists = dev_fracs.div( vecs.norm(dim=1) )

    seg_norms = strs[:, 4:6]
    dev_locs = seg_norms.mul(dev_dists[:,None]).add(seg_centers)

    h_size = linewidths*mag*36
    ax.quiver(locs_lf.cpu().numpy()[:,1], locs_lf.cpu().numpy()[:,0], vecs[:,1].cpu().numpy(), vecs[:,0].cpu().numpy(), angles='xy', units='xy', scale=1, width=linewidths*mag,
              headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=[0,0,1,1])

    color = t.zeros([seg_norms.size(0), 4], dtype=t.float)
    color[:,[1,3]] = 1

    ax.quiver(seg_centers.cpu().numpy()[:,1], seg_centers.cpu().numpy()[:,0], seg_norms[:,1].cpu().numpy(), seg_norms[:,0].cpu().numpy(), angles='xy', units='xy', scale=1, width=linewidths*mag,
              headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=color.numpy())

    ax.scatter(seg_centers.cpu().numpy()[:, 1], seg_centers.cpu().numpy()[:, 0], s=mag, alpha=1, marker=".")

    ax.scatter(dev_locs.cpu().numpy()[:, 1], dev_locs.cpu().numpy()[:, 0], s=mag, alpha=1, marker="x", color=[0,0,1,1])

    # plt.show(block=False)
    ##########


    x_proj_dist = vecs.matmul( t.tensor([[0.],[1.]]).to(dev) )
    interp_x = t.linspace(0,1,20).to(dev)[None].mul(x_proj_dist)[...,None].add(locs_lf[:,None])[...,1]

    dev_locs, locs_lf, locs_rt = dev_locs.cpu().numpy(), locs_lf.cpu().numpy(), locs_rt.cpu().numpy()
    interp_x = interp_x.cpu().numpy()
    written=0
    for i in range(dev_locs.shape[0]):

        x = t.tensor([dev_locs[i,1], locs_lf[i,1], locs_rt[i,1]])
        y = t.tensor([dev_locs[i,0], locs_lf[i,0], locs_rt[i,0]])

        x, sortids = x.sort()
        y = y[sortids]

        x, uids = x.unique_consecutive(return_inverse=True)
        uids = uids.unique()
        y = y[uids]

        if x.size(0) < 3: continue
        written += 1

        # spl = sinterp.make_interp_spline(x, y, k=3, bc_type='natural')
        # spl = sinterp.make_interp_spline(x, y, k=3, bc_type='clamped')
        spl = sinterp.make_interp_spline(x, y, k=2)
        interp_y = spl(interp_x[i])

        ax.plot(interp_x[i], interp_y)

    plt.show(block=False)
    print("no")
