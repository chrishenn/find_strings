import torch as t
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc

import scipy.interpolate as sinterp



def str_draw(strs, linewidths=0.01, dpi=150, max_size=32):
    ## strs = [locs[y,x] locs[y,x] norms[dy,dx] dev-frac]

    dev = strs.device
    mag = 25
    ax_max = max_size * mag

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')
    ax.set_ylim(ax_max, 0)
    ax.set_xlim(0, ax_max)
    plt.axis('off')

    strs = strs.clone()
    strs.mul_(mag)

    locs_lf, locs_rt = strs[:,:2], strs[:, 2:4]
    vecs = locs_rt.sub( locs_lf )

    #### spl_locs running directly from lf->rt
    # offsets = t.linspace(0, 1, 20)[None].repeat(vecs.size(0), 1).to(dev)
    # offsets.mul_( vecs.norm(dim=1, keepdim=True) )
    #
    # vecs.div_( vecs.norm(dim=1, keepdim=True) )
    # spl_locs = vecs[:,None].mul( offsets[...,None] ).add(locs_lf[:,None])

    ### spl_locs include some deviation dev-frac
    dev_fracs = strs[:,-1]
    seg_centers = locs_rt.add(locs_lf).div(2)
    seg_norms = vecs.clone()
    seg_norms[:,1].mul_(-1)
    seg_norms.div_( vecs.norm(dim=1, keepdim=True) )

    dev_dists = dev_fracs.mul( vecs.norm(dim=1) )

    dev_locs = seg_norms.mul(dev_dists[:,None]).add(seg_centers)


    x_proj_dist = vecs.matmul( t.tensor([[0.],[1.]]).to(dev) )
    interp_x = t.linspace(0,1,10).to(dev)[None].mul(x_proj_dist)[...,None].add(locs_lf[:,None])[...,1]

    dev_locs, locs_lf, locs_rt = dev_locs.cpu().numpy(), locs_lf.cpu().numpy(), locs_rt.cpu().numpy()
    interp_x = interp_x.cpu().numpy()
    for i in range(dev_locs.shape[0]):

        x = t.tensor([dev_locs[i,1], locs_lf[i,1], locs_rt[i,1]])
        y = t.tensor([dev_locs[i,0], locs_lf[i,0], locs_rt[i,0]])

        x, sortids = x.sort()
        y = y[sortids]

        x, uids = x.unique_consecutive(return_inverse=True)
        uids = uids.unique()
        y = y[uids]

        if x.size(0) < 2: continue

        spl = sinterp.make_interp_spline(x, y, k=2)
        interp_y = spl(interp_x[i])

        ax.plot(interp_x[i], interp_y)


    h_size = linewidths*mag*36
    ax.quiver(dev_locs[:,1], dev_locs[:,0], strs[:,5].cpu().numpy(), strs[:,4].cpu().numpy(), angles='xy', units='xy', scale=1, width=linewidths*mag,
              headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1)


    plt.show(block=False)
    print("no")