import atexit, math, os, time, PIL
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
from numba import njit, prange

import torch as t
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as transforms

from datasets.artificial_dataset import make_topbox, make_topbox_plus, make_nine, make_vbar, make_blob, make_tetris, make_topbar, make_circle_seed
from frnn_opt_brute import frnn_cpu
from string_finder import oodl_utils, oodl_draw
from string_finder.oodl_utils import regrid

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_opt_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_tree_brute/build/libfrnn_ts.so")

# t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/write_row/build/libwrite_row.so")

t.manual_seed(7)

import sparse
from annoy import AnnoyIndex

import torch.multiprocessing as mp
import queue


def mmm(data):
    return data.min(),data.max(),data.mean()





class Orientation(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1):
        super().__init__()

        self.chan_in = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = stride

        center = kernel_size // 2
        x = t.arange(kernel_size, dtype=t.float).sub(center)
        y = t.arange(kernel_size, dtype=t.float).sub(center)
        xx, yy = t.meshgrid(x, y)

        gaussian = t.exp((xx.pow(2) + yy.pow(2)).div(-2))
        dist = (xx.pow(2) + yy.pow(2)).sqrt()

        gaussian.masked_fill_(dist > center, 0)
        gaussian = gaussian.mul(-1)
        gaussian = gaussian.sub(gaussian.min())
        gaussian.masked_fill_(dist > center, 0)
        gaussian = gaussian.div(gaussian.max())
        gaussian = gaussian.mul(center)

        x_kernel = gaussian.mul(yy.sign())[None,None,...]
        y_kernel = gaussian.mul(xx.sign())[None,None,...]

        self.register_buffer('x_kernel', x_kernel)
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, batch):

        if batch.min() < -1e-5:
            batch = batch.clone().detach().add(1).div(2)

        batch = batch.pow(2).sum(1, keepdim=True).sqrt()

        ## 0-pad here
        xs = F.conv2d(batch, self.x_kernel, stride=self.stride, padding=self.padding)
        ys = F.conv2d(batch, self.y_kernel, stride=self.stride, padding=self.padding)

        return t.atan2(ys, xs + 1e-5)



class Canny(nn.Module):
    def __init__(self, opt, thresh_lo, thresh_hi, scale=1, sigma_gauss=None):
        super().__init__()

        self.low_threshold, self.high_threshold = thresh_lo, thresh_hi
        self.scale = scale
        self.sigma_gauss = sigma_gauss
        self.img_size = opt.img_size

        self.gen_sobel(5)
        if sigma_gauss is not None: self.gen_gauss(5, sigma=1)
        self.gen_selection_map()
        self.gen_hysteresis()

    def gen_selection_map(self):
        zeros = t.zeros([3,3])

        hori_lf = zeros.clone()
        hori_rt = zeros.clone()
        hori_lf[0,1] = 1
        hori_rt[2,1] = 1

        vert_up = zeros.clone()
        vert_dn = zeros.clone()
        vert_up[1,0] = 1
        vert_dn[1,2] = 1

        diag_tlf = zeros.clone()
        diag_brt = zeros.clone()
        diag_tlf[0,0] = 1
        diag_brt[2,2] = 1

        diag_blf = zeros.clone()
        diag_trt = zeros.clone()
        diag_blf[0,2] = 1
        diag_trt[2,0] = 1

        kernels = t.stack([hori_lf,hori_rt, vert_up,vert_dn, diag_tlf,diag_brt, diag_blf,diag_trt],0).unsqueeze(1)

        self.selection = nn.Conv2d(in_channels=1, out_channels=8,
                                   kernel_size=3, padding=1, bias=False).requires_grad_(False)
        self.selection.weight.data = kernels

        selection_ids = t.tensor([[0,1],[4,5],[2,3],[6,7], [0,1],[4,5],[2,3],[6,7]],dtype=t.long)
        self.register_buffer('selection_ids',selection_ids)

    def gen_sobel(self, k_size):

        range = t.linspace(-(k_size // 2), k_size // 2, k_size)
        x, y = t.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x.pow(2) + y.pow(2))
        sobel_2D_denominator[:, k_size // 2] = 1
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        sobel_2D.div_(6)

        self.sobel_x = nn.Conv2d(1, 1, kernel_size=k_size, padding=k_size // 2, padding_mode='reflect',
                                 bias=False).requires_grad_(False)
        self.sobel_y = nn.Conv2d(1,1, kernel_size=k_size, padding=k_size // 2, padding_mode='reflect',
                                 bias=False).requires_grad_(False)
        self.sobel_x.weight.data = sobel_2D.clone().t().view(1,1,k_size,k_size)
        self.sobel_y.weight.data = sobel_2D.view(1,1,k_size,k_size)

    def gen_gauss(self, k_gauss, sigma=0.8):

        D_1 = t.linspace(-1, 1, k_gauss)
        x, y = t.meshgrid(D_1, D_1)
        sq_dist = x.pow(2) + y.pow(2)

        # compute the 2 dimension gaussian
        gaussian = (-sq_dist / (2 * sigma**2)).exp()
        gaussian = gaussian / (2 * math.pi * sigma**2)
        gaussian = gaussian / gaussian.sum()

        self.gauss_conv = nn.Conv2d(1,1,kernel_size=k_gauss,
                                    padding=k_gauss // 2,
                                    padding_mode='reflect',
                                    bias=False).requires_grad_(False)
        self.gauss_conv.weight.data = gaussian.view(1,1,k_gauss,k_gauss)

    def gen_hysteresis(self):
        self.hysteresis = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='reflect',
                                    bias=False).requires_grad_(False)
        self.hysteresis.weight.data = t.ones((1, 1, 3, 3))

    def forward(self, images):

        ## de-normalize: important
        if images.min() < -1e-3:
            images = images.add(1).div(2)

        ## gauss blur
        if self.sigma_gauss is not None:
            images = self.gauss_conv(images)

        ## take intensity, flattening color channels to 1-D
        images = images.norm(p=2, dim=1, keepdim=True)
        images.div_(images.max())

        ## upsample
        if self.scale > 1: images = F.interpolate(images, scale_factor=self.scale, mode='area')

        ## take intensity-gradients
        sobel_x = self.sobel_x(images)
        sobel_y = self.sobel_y(images)

        grad_mag = (sobel_x.pow(2) + sobel_y.pow(2)).sqrt()
        grad_phase = t.atan2( sobel_x, sobel_y +1e-5 )

        ## non-maximum suppression
        grad_phase = grad_phase.div(math.pi/4).round().add(4).fmod(8)
        grad_phase = grad_phase.long()

        selections = self.selection(grad_mag)
        neb_ids = self.selection_ids[grad_phase]
        nebs = selections.gather(1, neb_ids[:,0,...].permute(0,3,1,2))

        mask1 = grad_mag <= nebs[:,0,None,...]  # using one gradient-direction-neighbor as a tiebreaker
        mask2 = grad_mag < nebs[:,1,None,...]
        mask = mask1 | mask2
        grad_mag = t.where(mask, t.zeros_like(mask).float(), grad_mag)

        ## downsample to original size
        if self.scale > 1: grad_mag = F.interpolate(grad_mag, size=self.img_size, mode='nearest')

        ## thresholds, hysteresis
        mask = grad_mag < self.low_threshold
        grad_mag = t.where(mask, t.zeros_like(mask).float(), grad_mag)

        weak_mask = (grad_mag < self.high_threshold) & (grad_mag > self.low_threshold)

        high_mask = grad_mag > self.high_threshold

        high_nebs = self.hysteresis(high_mask.float())
        weak_keep = weak_mask & (high_nebs > 0)
        mask = weak_keep.logical_not() & high_mask.logical_not()

        grad_mag = t.where(mask, t.zeros_like(mask).float(), grad_mag)

        return grad_mag, t.cat([sobel_y, sobel_x], 1)



#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################






class OOSampler(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) / 2
        self.chan_in = opt.c_init

        self.gen_init(opt)

    def gen_init(self, opt):
        D1 = t.arange(opt.img_size, dtype=t.int)
        D2 = t.arange(opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        gridy = gridy[None, None, ...].float()
        gridx = gridx[None, None, ...].float()

        dists = (gridx.sub(self.center).pow(2) + gridy.sub(self.center).pow(2)).sqrt().squeeze()
        self.register_buffer('dists', dists)
        img_filter = dists < (self.center + 0.5)
        self.register_buffer('img_filter', img_filter)

        pts = t.cat([gridy, gridx], 1)
        pts = pts.permute([0, 2, 3, 1])
        self.register_buffer('pts', pts)

    def forward(self, batch, sizes):

        batch_size = t.tensor(batch.size(0), device=batch.device)

        if sizes[0] > 0:
            sizes = sizes.sub(1).true_divide(2)[:, None, None]
            resize_mask = self.dists[None].repeat(batch_size, 1, 1) < sizes.add(0.5)

            filter = resize_mask
        else:
            filter = self.img_filter[None].repeat(batch_size, 1, 1)

        ## pad outside center circle in image with 0's
        batch.permute(0, 2, 3, 1)[filter.logical_not()] = 0

        ## sample objects
        tex = batch.permute(0, 2, 3, 1)
        tex = tex[filter]

        pts = self.pts.clone().repeat(batch_size, 1, 1, 1)
        pts = pts[filter]

        imgid = t.arange(batch_size)[:, None, None].repeat(1, self.img_size, self.img_size)
        imgid = imgid[filter].to(pts.device)

        return tex, pts, imgid, batch_size

class OOSampler_Full(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) / 2
        self.chan_in = opt.c_init

        self.gen_init(opt)
        self.o_layer = Orientation(in_channels=self.chan_in, kernel_size=5)

    def gen_init(self, opt):
        D1 = t.arange(opt.img_size, dtype=t.int)
        D2 = t.arange(opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        gridy = gridy[None, None, ...].float()
        gridx = gridx[None, None, ...].float()

        zeros = t.zeros_like(gridx)
        ones = t.ones_like(gridx)
        pts = t.cat([gridy, gridx, zeros, zeros, ones, zeros], 1)
        pts = pts.squeeze().permute(1,2,0)
        pts = pts.reshape(-1, pts.size(-1))
        self.obj_perim = pts.size(0)
        pts = pts.repeat(opt.batch_size, 1)
        self.register_buffer('pts', pts)

        imgid = t.arange(opt.batch_size).repeat_interleave(self.obj_perim)
        self.register_buffer('imgid', imgid)

    def forward(self, batch, sizes, canny_mask):

        batch_size = t.tensor(batch.size(0), device=batch.device)
        cutoff = batch_size * self.obj_perim

        pts = self.pts[:cutoff, :].clone()
        imgid = self.imgid[:cutoff].clone()

        tex = batch.permute(0,2,3,1)
        tex = tex.reshape(-1, tex.size(-1))

        return tex, pts, imgid, batch_size

class OOSampler_Patch(nn.Module):
    def __init__(self, opt, pdesc_size, pdesc_r=3):
        super().__init__()

        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) / 2
        self.chan_in = opt.c_init
        self.pdesc_size = pdesc_size
        self.pdesc_r = pdesc_r

        self.gen_sample_locs(opt)

    def gen_sample_locs(self, opt):
        D1 = t.arange(opt.img_size, dtype=t.int)
        D2 = t.arange(opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        gridy = gridy[None, None, ...].float()
        gridx = gridx[None, None, ...].float()

        dists = (gridx.sub(self.center).pow(2) + gridy.sub(self.center).pow(2)).sqrt().squeeze()
        dists = dists[None].repeat(opt.batch_size, 1, 1)
        self.register_buffer('dists', dists)

        img_filter = dists < (self.center + 0.5)
        self.register_buffer('img_filter', img_filter)

        zeros = t.zeros_like(gridx)
        ones = t.ones_like(gridx)
        pts = t.cat([gridy, gridx, zeros, zeros, ones, zeros], 1)
        pts = pts.squeeze().permute(1, 2, 0).reshape(-1,6)

        locs = pts[:,:2].clone()
        self.obj_perim = locs.size(0)

        pts = pts.repeat(opt.batch_size,1)
        self.register_buffer('pts', pts)

        locs = locs[...,None].repeat(1,1,self.pdesc_size).permute(0,2,1)
        locs = locs[None].repeat(opt.batch_size,1,1,1)
        self.register_buffer('locs', locs)

        imgid = t.arange(opt.batch_size).repeat_interleave(self.obj_perim)
        self.register_buffer('imgid', imgid)

        offsets = t.cartesian_prod(t.tensor([-2,-1,0,1, 2]), t.tensor([-2,-1,0,1, 2]))[None,None]
        self.register_buffer('offsets', offsets)

    def forward(self, batch, sizes):
        batch_size = batch.size(0)

        if sizes[0] > 0:
            sizes = sizes.sub(1).true_divide(2).add(0.5)[:, None, None]
            filter = self.dists[ :batch_size, ...].lt( sizes )
            filter_flat = filter.reshape(-1)
        else:
            filter = self.img_filter[ :batch_size, ...]
            filter_flat = filter.reshape(-1)

        locs = self.locs[ :batch_size, ...].clone()

        ## rand
        locs.add_(self.offsets)

        # locs.add_(offsets)
        locs.sub_(self.center + 0.5)
        locs.div_(self.center + 0.5)

        intensity = batch.norm(dim=1, keepdim=True)
        pdesc = F.grid_sample(intensity, locs, align_corners=True).squeeze()
        pdesc.div_(2)
        pdesc = pdesc.reshape(-1, pdesc.size(-1))[filter_flat]

        tex = batch.permute(0, 2, 3, 1)[filter]

        imgid = self.imgid[filter_flat].clone()
        pts = self.pts[filter_flat].clone()

        return tex, pts, imgid, t.tensor(batch.size(0), device=batch.device), pdesc




#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################




class String_Finder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        thresh_lo, thresh_hi = 0.1, 0.1
        self.canny = Canny(opt, thresh_lo, thresh_hi)


    def forward(self, batch):
        dev = batch.device

        ## TEST ################################
        # batch_size = t.tensor(batch.size(0), device=batch.device)
        # depth = 6
        #
        # locs = t.tensor([1, 3, 5, 7], dtype=t.long, device=0)
        # locs = t.cartesian_prod(locs, locs).float()
        # tree_table = t.zeros([locs.size(0) // 2, locs.size(0) // 2 * depth], dtype=t.uint8, device=locs.device)
        # tree_table[0,1], tree_table[4,1] = 1,1
        #
        # pair_ids = t.arange(locs.size(0) // 2, device=locs.device).repeat(2)
        # imgid = t.zeros_like(locs[:, 0]).long()
        # edges = t.ops.fbt_op.fbt_kern(locs, imgid, t.tensor(4+0.2).cuda(), t.tensor(1.0).cuda(), pair_ids, tree_table, batch_size)[0]
        # oodl_draw.oodl_draw(0, locs, imgid, draw_obj=True, as_vecs=edges, max_size=12)
        #
        # ##
        #
        # [te for te in locs]
        # [te for te in tree_table]
        # [(te.item(), te1) for (te, te1) in zip(pair_ids, locs)]
        ################################


        ###### Initial Detection and building primitives
        #### detect canny-edges, connect them, select norms from canny sites
        batch_size = t.tensor(batch.size(0), device=batch.device)
        b_edges, sobel = self.canny(batch)

        edge_ends = b_edges.nonzero()
        imgid = edge_ends[:,0]
        locs = edge_ends[:, -2:].float().contiguous()

        edges = t.ops.my_ops.frnn_ts_kernel(locs, imgid, t.tensor(np.sqrt(2)+0.01).cuda(), t.tensor(1).cuda(), batch_size)[0]

        ids = edge_ends.split(1, dim=1)
        norms = sobel[ ids[0], :, ids[2], ids[3] ]
        norms = norms.squeeze().div( norms.squeeze().norm(dim=1)[:,None] )

        #### compose initial strings
        ## strs[i] = [ end_lf[y,x], end_rt[y,x], str_norm[dy,dx], dev-%, len-pix ]
        locs_lf, locs_rt = locs[edges[:,0]], locs[edges[:,1]]

        norms_lf, norms_rt = norms[edges[:,0]], norms[edges[:,1]]
        norms = norms_lf.add( norms_rt ).div(2)

        strs = t.cat([ locs_lf, locs_rt, norms, t.zeros_like(edges[:,0,None]),  locs_lf.sub(locs_rt).norm(dim=1,keepdim=True) ], 1)


        ###### MERGING TREE

        ### PSEUDOCODE #############
        ## original strings become root nodes in search tree. These unmerged strings represent the choice to 'not merge', and thus should remain in the tree for
        # subsequent scoring.

        ## for enough steps to reach an arbitrary tree depth:
            ## for each neighboring (qualified) pair of strings, generate a possible combined string.
            ## "Similarity Score": compute the similarity score for strings that could possibly merge; store in tree-node for possible new string

            ## a string cannot merge with any parent, since a string represents all space covered by all parents.
            ## strings that share a parent cannot merge for the same reason.

            ## increase the search size for neighboring strings proportional to their size

        ## "Underlying Data Scores": generate scores for each node on the tree

        ## extract best-scored strings from pruned tree.
        ### ################


        ## TODO: iterate start

        depth = 6
        n_strs = strs.size(0)

        ## each row gives a node's inheritance. A '1' in a row gives a parent, whose id is the column-id in which the '1' appears
        ## since each node (string) has two ends, the tree_table will necessarily have half as many rows as there are locations
        tree_table = t.zeros([n_strs, n_strs * depth], dtype=t.uint8, device=dev)

        ## find edges between the ends of edges - excluding my own other string-end, and excluding any parents I inherit from

        all_ends = t.cat([locs_lf, locs_rt])
        pair_ids = t.arange(n_strs,device=locs_lf.device).repeat(2)
        ## TODO: support batching
        imgid = t.zeros_like(all_ends[:,0]).long()




        ## find edges from all string-ends
        edges_from_all_ends = t.ops.my_ops.frnn_ts_kernel(all_ends, imgid, t.tensor(0.5).cuda(), t.tensor(1).cuda(), batch_size)[0]

        pairids_lf, pairids_rt = pair_ids[edges_from_all_ends[:,0]], pair_ids[edges_from_all_ends[:,1]]

        # filter ends that define the same string
        same_str = pairids_lf.eq( pairids_rt )

        # filter ends that share an ancestor-string
        share_ancester = (tree_table[pairids_lf] & tree_table[pairids_rt]).sum(dim=1).bool()

        # filter ends that inherit from each other
        inherit_lfrt = tree_table[pairids_lf].gather(1, pairids_rt[:,None]).squeeze().bool()
        inherit_rtlf = tree_table[pairids_rt].gather(1, pairids_lf[:,None]).squeeze().bool()

        end_edges = edges_from_all_ends[ (same_str | share_ancester | inherit_lfrt | inherit_rtlf).logical_not() ]


        ## generate possible combined strings from valid end_edges
        ends_lf, ends_rt = all_ends[end_edges[:,0]], all_ends[end_edges[:,1]]


        ## NOTE: assumes strings' ends are mirrored across the halfway-row of all_ends
        ## where end_edges.ge(n_strs) gives locs_rt in strs
        pairids_lf, pairids_rt = pair_ids[end_edges[:, 0]], pair_ids[end_edges[:, 1]]

        other_endpt_cols_lf = t.where( end_edges[:,0].ge(n_strs)[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
        other_endpt_cols_rt = t.where( end_edges[:,1].ge(n_strs)[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
        other_endpts_lf = strs[pairids_lf].gather(1, other_endpt_cols_rt)
        other_endpts_rt = strs[pairids_rt].gather(1, other_endpt_cols_lf)


        ## TEST ################################
        # pts_lf = t.cat([ends_lf, other_endpts_lf])
        # pair_ids = t.arange(pts_lf.size(0)//2, device=locs_lf.device)[:,None].repeat(1,2)
        # pair_ids[:,1].add_(pts_lf.size(0)//2)
        # oodl_draw.oodl_draw(0, pts_lf, t.zeros_like(pts_lf[:,0]).long(), draw_obj=False, edges=pair_ids, max_size=self.opt.img_size)
        #
        # pts_rt = t.cat([ends_rt, other_endpts_rt])
        # pair_ids = t.arange(pts_rt.size(0)//2, device=locs_rt.device)[:,None].repeat(1,2)
        # pair_ids[:,1].add_(pts_rt.size(0)//2)
        # oodl_draw.oodl_draw(0, pts_rt, t.zeros_like(pts_rt[:,0]).long(), draw_obj=False, edges=pair_ids, max_size=self.opt.img_size)
        ################################


        # draw a straight line between the farther endpoints of the possible string
        # string deviation direction is referenced to string normal
        # new string deviation is generated by max perp dist from straight-line, given by other possible endpoints - not current deviations
        ## TODO: generate possible string from edges that are farther away, but some other ends of those strings are close together?








        ## TODO: similarity scores will be generated here

        ## TODO: update tree_table, pair_ids, all_ends, imgid




        return None























