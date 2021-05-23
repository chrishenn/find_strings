import torch as t
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Circle
matplotlib.rcParams['image.interpolation'] = 'none'

import numpy as np
import cv2



def oodl_draw(visimg_id=None, pts=None, imgid=None, img=None, edges=None, as_vecs=None, o_vectors=None, strs=None, groupids=None, draw_obj=False, dot_locs=None,
                         o_scale=1, max_size=32, linewidths=0.01, dpi=150):
    '''
    Each element to draw is composited onto a singular canvas, and is drawn from the image given by visimg_id, indexed into
        the batch of images included in pts and imgid. Each row of imgid provides an img id for the object in that row in pts.

    visimg_id=int: imgid of image whose elements we may draw
    pts=[n_obj x 6] tensor: each row is an object as [y, x, z, angle, size, <n/a>]
    imgid=[n_obj] tensor: each elem is an img id in the batch

    img=[h x w x c] tensor: [draw image] image-tensor will be de-norm'd to [0,1] if it spans [-1,1]

    edges=[n_edges x 2] tensor: [draw edges] where each elem is a row of pts
    vecs=[n_edges x 2] tensor: [draw vectors] where each elem is a row of pts; each vec drawn with a color unique to the destination object
    groupids=[n_ojb] tensor: [draw groups] where each elem is a unique groupid; drawn with a unique color

    draw_obj=bool: [draw objects] draw each object in each row of pts as [y, x, z, angle, size, <n/a>]

    max_size=float/int: max size of underlying image corresponding to drawn inputs
    linewidths=float/int: thicknesses of edges and vector stems
    dpi=int: dpi resolution of output image
    '''

    magnify = 25
    fig, ax = plt.subplots(dpi=dpi)

    ymax = max_size * magnify
    xmax = max_size * magnify
    ax.set_aspect('equal')
    ax.set_ylim(ymax, 0)
    ax.set_xlim(0, xmax)
    plt.axis('off')

    if img is not None:
        topil = transforms.ToPILImage()

        if img.min() < -1e-4:
            img = img.add(1).div(2)
        img = topil(img.cpu())
        img = img.resize([xmax, ymax], resample=0)
        plt.imshow(img)

    if draw_obj or (edges is not None) or (as_vecs is not None) or (groupids is not None) or (strs is not None) or (dot_locs is not None):

        if (pts is not None) or (imgid is not None):
            pts_full = pts.clone().detach().cpu()
            imgid = imgid.clone().detach().cpu()

            pts_full[:,:2].add_(0.5)
            if pts_full.size(1) > 2:
                pts_full[:,[0,1,2,4]] = pts_full[:,[0,1,2,4]].mul(magnify)
            else:
                pts_full[:,:2] = pts_full[:,:2].mul(magnify)

            im_mask = imgid.eq(visimg_id)
            pts_im = pts_full[im_mask]

        colors = None
        if draw_obj:
            draw_objects(ax, pts_im, xmax, ymax, magnify, o_scale)
        if groupids is not None:
            groupids = groupids.clone().detach().cpu()
            palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 0], dtype=t.long)
            colors = groupids[:, None].mul(palette).fmod(255).div(255)
            colors[:, 3] = 1
            colors = colors.numpy()
            draw_groupids(ax, pts_im, colors[im_mask], magnify)
        if edges is not None:
            edges = edges.cpu()
            edges_im = edges[im_mask[edges[:, 0]]]
            draw_edges(ax, pts_full, edges_im, linewidths, magnify, colors=colors)
        if as_vecs is not None:
            vecs = as_vecs.cpu()
            vecs_im = vecs[im_mask[vecs[:, 0]] | im_mask[vecs[:, 1]]]
            draw_vectors(ax, pts_full, vecs_im, linewidths, magnify)
        if o_vectors is not None:
            o_vectors = o_vectors.cpu()
            vecs_im = o_vectors[im_mask]
            vecs_im.mul_(magnify)
            draw_o_vectors(ax, pts_im, vecs_im, linewidths, magnify)
        if strs is not None:
            strs = strs.cpu()
            strs[:, :4].add_(0.5).mul_(magnify)
            draw_strs(ax, strs, linewidths, magnify)
        if dot_locs is not None:
            dot_locs.add_(0.5).mul_(magnify)
            dot_locs = dot_locs.cpu()
            draw_dots(ax, dot_locs, magnify)

    plt.show(block=False)


def draw_objects(ax, pts_im, xmax, ymax, magnify, o_scale):

    # y, x, z, angle, size, <n/a>
    if pts_im.size(1) > 2:
        kp_list = [cv2.KeyPoint(y=object[0].item(), x=object[1].item(), _angle=np.rad2deg(object[3].item()), _size=object[4].item()*o_scale) for object in pts_im]
    else:
        kp_list = [cv2.KeyPoint(y=object[0].item(), x=object[1].item(), _angle=0, _size=magnify*o_scale) for object in pts_im]

    topil = transforms.ToPILImage()
    img = t.full([3,xmax,ymax], 0, dtype=t.uint8)
    img = topil(img)
    img = img.resize([xmax, ymax], resample=0)
    img = np.array(img)
    img = cv2.drawKeypoints(img, kp_list, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img = t.tensor(img)

    opaque = img[:,:,:3].sum(2).floor_divide(3).byte()
    img = t.cat([img, opaque[:,:,None]],2)
    img = img.permute(2,0,1)
    img = topil(img)

    ax.imshow(img)


def draw_groupids(ax, pts_im, colors_im, magnify):
    alpha = 0.75

    circles = [Circle((pts_im[i,1].item(),pts_im[i,0].item()), 0.25*magnify) for i in range(len(colors_im))]
    circ_cl = mc.PatchCollection(circles, facecolor=colors_im, alpha=alpha)
    ax.add_collection(circ_cl)


def draw_edges(ax, pts_full, edges_im, linewidths, magnify, colors):

    locs_lf, locs_rt = pts_full[:,:2][edges_im[:, 0]], pts_full[:,:2][edges_im[:, 1]]
    locs_lf, locs_rt = t.stack([locs_lf[:,1], locs_lf[:,0]],1), t.stack([locs_rt[:,1], locs_rt[:,0]],1)

    if colors is not None:
        colors_edges = colors[ edges_im[:, 0] ]
        lc = mc.LineCollection(list(zip(locs_lf.numpy(), locs_rt.numpy())), linewidths=linewidths*magnify, colors=colors_edges)
    else:
        lc = mc.LineCollection(list(zip(locs_lf.numpy(), locs_rt.numpy())), linewidths=linewidths*magnify)
    ax.add_collection(lc)


def draw_vectors(ax, pts_full, vecs_im, linewidths, magnify):

    palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
    colors = vecs_im[:,1,None].float().mul(palette).fmod(255).div(255)
    colors[:,3] = 1
    colors = colors.numpy()

    vectors = pts_full[:,:2][vecs_im[:,1]] - pts_full[:,:2][vecs_im[:,0]]
    locs_lf = pts_full[:,:2][vecs_im[:,0]]

    h_size = linewidths*magnify*36
    ax.quiver(locs_lf[:,1], locs_lf[:,0], vectors[:,1], vectors[:,0], angles='xy', units='xy', scale=1, width=linewidths*magnify,
              color=colors, headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1)

def draw_o_vectors(ax, pts_im, vecs_im, linewidths, magnify):

    palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
    colors = t.arange(pts_im.size(0))[:,None].float().mul(palette).fmod(255).div(255)
    colors[:,3] = 1
    colors = colors.numpy()

    vectors = vecs_im
    locs_lf = pts_im[:,:2]

    h_size = linewidths*magnify*36
    ax.quiver(locs_lf[:,1], locs_lf[:,0], vectors[:,1], vectors[:,0], angles='xy', units='xy', scale=1, width=linewidths*magnify,
              color=colors, headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1)


def draw_strs(ax, strs, linewidths, magnify):

    palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
    colors = t.arange(strs.size(0))[:,None].float().mul(palette).fmod(255).div(255)
    colors[:,3] = 1
    colors = colors.numpy()

    norm_locs = strs[:,:2].add(strs[:,2:4]).div(2)

    h_size = linewidths*magnify*36
    ax.quiver(norm_locs[:,1], norm_locs[:,0], strs[:,5], strs[:,4], angles='xy', units='xy', scale=1, width=linewidths*magnify,
              color=colors, headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1)

    locs_lf, locs_rt = t.stack([strs[:,1], strs[:,0]],1), t.stack([ strs[:,3], strs[:,2] ],1)
    lc = mc.LineCollection(list(zip(locs_lf.numpy(), locs_rt.numpy())), linewidths=linewidths*magnify)
    ax.add_collection(lc)

def draw_dots(ax, dot_locs, magnify):
    dot_locs = dot_locs.numpy()

    ax.scatter(dot_locs[:,1], dot_locs[:,0], s=magnify/2, alpha=0.5, marker=".")









