import atexit, math, os, time, PIL
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import collections as mc

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
from string_finder.str_draw import str_draw
from unionfind.unionfind import UnionFind

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_opt_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_tree_brute/build/libfrnn_ts.so")

# t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/write_row/build/libwrite_row.so")

t.manual_seed(7)

import scipy.interpolate as sinterp

# import torch.multiprocessing as mp
# import queue


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
        t.cuda.set_device(dev)


        ###### Initial Detection and building primitives
        #### detect canny-edges, connect them, select norms from canny sites
        batch_size = t.tensor(batch.size(0), device=dev)
        b_edges, sobel = self.canny(batch)

        edge_ends = b_edges.nonzero()
        imgid = edge_ends[:,0]
        locs = edge_ends[:, -2:].float().contiguous()

        edges = t.ops.my_ops.frnn_ts_kernel(locs, imgid, t.tensor(np.sqrt(2)+0.01).cuda(), t.tensor(1).cuda(), batch_size)[0]

        ids = edge_ends.split(1, dim=1)
        norms = sobel[ ids[0], :, ids[2], ids[3] ]
        norms = norms.squeeze().div( norms.squeeze().norm(dim=1)[:,None] )


        #### compose initial strings
        ## strs[i] = [ end_lf[y,x], end_rt[y,x], str_norm[dy,dx], dev-frac ]
        locs_lf, locs_rt = locs[edges[:,0]], locs[edges[:,1]]

        norms_lf, norms_rt = norms[edges[:,0]], norms[edges[:,1]]
        norms = norms_lf.add( norms_rt ).div(2)

        strs = t.cat([ locs_lf, locs_rt, norms, t.zeros_like(edges[:,0,None]) ], 1)


        ## TEST ################################
        # oodl_draw.oodl_draw(0, strs=strs, max_size=self.opt.img_size)
        b_edges[b_edges.gt(0)] = 1

        # str_draw(strs, max_size=self.opt.img_size)
        ################################




        ###### MERGING TREE

        ### PSEUDOCODE #############
        ## original strings become root nodes in search tree. These unmerged strings represent the choice to 'not merge', and thus should remain in the tree for subsequent scoring.

        ## for enough steps to reach some desired tree depth:
            ## for each neighboring pair of strings, generate a possible combined string.
            ## "Score": compute the score for strings that could possibly merge
                ## merging score is some arbitrary combination of similarity of curvature and representation of underlying data
        ### ################


        depth = 2

        ##
        tree_table = UnionFind()
        [tree_table.add(i) for i in range(strs.size(0))]

        rads = [1, 2]

        for i in range(depth):

            ## TODO: support batching
            all_ends = t.cat([strs[:,[0,1]], strs[:,[2,3]]])
            str_ids = t.arange(strs.size(0), device=dev).repeat(2)
            imgid = t.zeros_like(all_ends[:,0]).long()


            ## find edges from all string-ends
            end_edges = t.ops.my_ops.frnn_ts_kernel(all_ends, imgid, t.tensor( rads[i] ).cuda(), t.tensor(1).cuda(), batch_size)[0]

            str_edges = t.where(end_edges.ge(strs.size(0)), end_edges.sub(strs.size(0)), end_edges)


            # filter ends that define the same string
            same_str = str_edges[:,0].eq( str_edges[:,1] )

            ## filter ends that define a string that share an ancestor, or inherit from each other
            related = t.tensor( [tree_table.connected(str_edge[0].item(), str_edge[1].item()) for str_edge in str_edges] )

            end_edges = end_edges[ (same_str | related.cuda()).logical_not() ]
            str_edges = str_edges[ (same_str | related.cuda()).logical_not() ]


            ###### generate combined strings from valid end_edges
            ## draw a "straight-line" between the farther endpoints (far_ends_lf/far_ends_rt) of the neighboring strings
            ## string deviation direction is referenced to string normal. normal points toward center of curvature.
            ## new string deviation is generated by avg perp dists from straight-line seg to other possible endpoints (not current deviations)
            ## new string normal is avg direction of the two contributing strings

            close_ends_lf, close_ends_rt = all_ends[end_edges[:,0]], all_ends[end_edges[:,1]]

            strids_lf, strids_rt = str_ids[str_edges[:, 0]], str_ids[str_edges[:, 1]]

            other_endpt_cols_lf = t.where( end_edges[:,0].ge(strs.size(0))[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
            other_endpt_cols_rt = t.where( end_edges[:,1].ge(strs.size(0))[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
            far_ends_lf = strs[strids_lf].gather(1, other_endpt_cols_rt)
            far_ends_rt = strs[strids_rt].gather(1, other_endpt_cols_lf)

            ## TEST ################################
            ## ends_lf and ends_rt are neighboring string-ends that are  close together
            # neb_locs = t.cat([close_ends_lf, close_ends_rt])
            # pair_ids = t.arange(neb_locs.size(0)//2, device=locs_rt.device)[:,None].repeat(1,2)
            # pair_ids[:,1].add_(neb_locs.size(0)//2)
            # oodl_draw.oodl_draw(0, neb_locs, t.zeros_like(neb_locs[:,0]).long(), draw_obj=False, edges=pair_ids, max_size=self.opt.img_size)

            ## other_ends_lf and other_ends_rt span the farthest-away extremes of strings whose ends have found neighbors. sometimes they are coincident
            # neb_locs = t.cat([far_ends_lf, far_ends_rt])
            # pair_ids = t.arange(neb_locs.size(0)//2, device=locs_rt.device)[:,None].repeat(1,2)
            # pair_ids[:,1].add_(neb_locs.size(0)//2)
            # oodl_draw.oodl_draw(0, neb_locs, t.zeros_like(neb_locs[:,0]).long(), draw_obj=False, as_vecs=pair_ids, max_size=self.opt.img_size)
            ################################

            ##### filter out zero-length string possibles
            seg_lens = F.pairwise_distance(far_ends_lf, far_ends_rt)
            not_coinc = seg_lens.gt(1e-3)

            ## filter distances and endpoints
            seg_lens = seg_lens[not_coinc]
            close_ends_lf, close_ends_rt = close_ends_lf[not_coinc], close_ends_rt[not_coinc]
            far_ends_lf, far_ends_rt = far_ends_lf[not_coinc], far_ends_rt[not_coinc]

            ## filter pair_ids
            strids_lf, strids_rt = strids_lf[not_coinc], strids_rt[not_coinc]
            str_edges = str_edges[not_coinc]

            ## filtered string normals
            norm_cols = t.tensor([[4, 5]], device=dev).repeat(strids_lf.size(0), 1)
            seg_norms_lf, seg_norms_rt = strs[strids_lf].gather(1, norm_cols), strs[strids_rt].gather(1, norm_cols)


            #### generate new possible strings

            ## deviation_pt describes the location of deviation from straight-line connection between string far-ends
            dev_locs = close_ends_lf.add(close_ends_rt).div(2)

            ## new deviation frac of merged string will be the right-angle deviation of deviation_pt to straight-line connection, relative to straight-line seg length
            test_pts = dev_locs
            dev_dists = ( (far_ends_rt[:,1] - far_ends_lf[:,1]) * (far_ends_lf[:,0] - test_pts[:,0]) )  -  ( (far_ends_lf[:,1] - test_pts[:,1]) * (far_ends_rt[:,0] - far_ends_lf[:,0]) )
            new_dev_fracs = dev_dists.div(seg_lens)

            ## new string normals
            new_str_norms = seg_norms_lf.add(seg_norms_rt).div(2)
            new_str_norms.div_( new_str_norms.norm(dim=1, keepdim=True) )

            ## pack new strs datastrct
            new_strs = t.cat([ far_ends_lf, far_ends_rt, new_str_norms, new_dev_fracs[:,None] ], 1)


            ##### apply scores

            ## artificial norms at right-angles to vecs lf->rt
            locs_lf, locs_rt = new_strs[:, :2], new_strs[:, 2:4]
            vecs = locs_rt.sub(locs_lf)

            new_norms = vecs.div( vecs.norm(dim=1, keepdim=True) )
            new_norms = t.cat([new_norms, t.zeros_like(new_norms[:, 0, None])], 1)
            unit_z = t.tensor([[0., 0., 1.]]).repeat(new_norms.size(0), 1).to(dev)
            new_norms = t.cross(new_norms, unit_z)[:, :-1]

            ## reflect these artificial norms to match the general direction of the organic ones
            old_norms = new_strs[:, 4:6].clone()
            norm_diff = old_norms - new_norms
            new_norms = t.where(norm_diff.norm(dim=1, keepdim=True).lt(np.sqrt(2)), new_norms, -new_norms)

            new_strs[:, 4:6] = new_norms



            ###### TEST ####################
            mag = 25
            ax_max = self.opt.img_size * mag
            h_size = 0.01 * mag * 36

            fig, ax = plt.subplots(dpi=150)
            ax.set_aspect('equal')
            ax.set_ylim(ax_max, 0)
            ax.set_xlim(0, ax_max)
            plt.axis('off')

            locs_lf_tmp, locs_rt_tmp = locs_lf.clone().cpu().numpy(), locs_rt.clone().cpu().numpy()
            seg_center = (((locs_lf_tmp + locs_rt_tmp) / 2) + 0.5) * mag
            old_norms = old_norms.clone().cpu().numpy() * mag
            new_norms = new_norms.clone().cpu().numpy() * mag
            dev_locs_tmp = dev_locs.clone().add(0.5).cpu().numpy() * mag

            palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
            colors = t.arange(new_norms.shape[0])[:,None].mul(30).float().mul(palette).fmod(255).div(255)
            colors = colors.numpy()
            colors[:, 3] = 1

            ax.quiver(seg_center[:, 1], seg_center[:, 0], old_norms[:, 1], old_norms[:, 0], angles='xy', units='xy',
                      scale=1, width=0.01 * mag,
                      headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=colors)

            ax.quiver(seg_center[:, 1], seg_center[:, 0], new_norms[:, 1], new_norms[:, 0], angles='xy', units='xy',
                      scale=1, width=0.01 * mag,
                      headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=colors)

            locs_lf_tmp, locs_rt_tmp = (locs_lf_tmp + 0.5) * mag, (locs_rt_tmp + 0.5) * mag
            locs_lf_tmp = np.stack([locs_lf_tmp[:, 1], locs_lf_tmp[:, 0]], axis=1)
            locs_rt_tmp = np.stack([locs_rt_tmp[:, 1], locs_rt_tmp[:, 0]], axis=1)
            lc = mc.LineCollection(list(zip(locs_lf_tmp, locs_rt_tmp)), linewidths=.01 * mag, color=colors)
            ax.add_collection(lc)

            # ax.scatter(seg_center[:, 1], seg_center[:, 0], s=mag, alpha=1, marker="x", color=colors)
            ax.scatter(dev_locs_tmp[:, 1], dev_locs_tmp[:, 0], s=mag, alpha=1, marker="x", color=colors)

            ax.scatter(locs_lf_tmp[:, 0], locs_lf_tmp[:, 1], s=mag, alpha=1, marker=".", color=colors)
            ax.scatter(locs_rt_tmp[:, 0], locs_rt_tmp[:, 1], s=mag, alpha=1, marker=".", color=colors)

            # plt.show(block=False)
            ################################



            ## make smooth splines
            # TODO: transform locs into coords referenced by string normals.

            n_samples = 10

            x_proj_dist = vecs.matmul(t.tensor([[0.], [1.]]).to(dev))
            interp_x = t.linspace(0, 1, n_samples, device=dev)[None].mul(x_proj_dist)[..., None].add(locs_lf[:, None])[..., 1]

            locs_x = t.stack([ locs_lf[:,1], dev_locs[:,1], locs_rt[:,1] ], 1)
            locs_y = t.stack([ locs_lf[:,0], dev_locs[:,0], locs_rt[:,0] ], 1)

            interp_x_np = interp_x.cpu().numpy()

            # splines = [batch, string, samples, x, y]
            splines = t.zeros([batch_size, new_strs.size(0), n_samples, 2], device=dev)
            splines[0,:,:,0] = interp_x

            jitter = t.rand(3, device=dev).sub(0.5) * (1e-1)
            for i in range(dev_locs.shape[0]):

                x = locs_x[i]
                y = locs_y[i]

                x = x.add(jitter)

                x, sortids = x.sort()
                y = y[sortids]

                # spl = sinterp.make_interp_spline(x, y, k=3, bc_type='natural')
                # spl = sinterp.make_interp_spline(x, y, k=3, bc_type='clamped')
                spl = sinterp.make_interp_spline(x.cpu().numpy(), y.cpu().numpy(), k=2)

                interp_y = spl(interp_x_np[i])

                splines[0, i, :, 1] = t.from_numpy( interp_y ).to(dev)

            # TODO: un-transform splines from normal-ref'd coords into image-ref'd coords

            ###### TEST ####################


            # mag = 25
            # ax_max = self.opt.img_size * mag
            #
            # fig, ax = plt.subplots(dpi=150)
            # ax.set_aspect('equal')
            # ax.set_ylim(ax_max, 0)
            # ax.set_xlim(0, ax_max)
            # plt.axis('off')

            splines_tmp = splines.clone().add(0.5).mul(mag).cpu().numpy()
            for i in range(splines_tmp.shape[1]):
                ax.plot(splines_tmp[0,i,:,0], splines_tmp[0,i,:,1])

            img = b_edges[0].clone()
            topil = transforms.ToPILImage()
            if img.min() < -1e-4:
                img = img.add(1).div(2)
            img = topil(img.cpu())
            img = img.resize([ax_max, ax_max], resample=0)
            plt.imshow(img)

            plt.show(block=False)
            ################################


            ## sample values from b_edges using spline locations
            sample_splines = splines.clone().div(self.opt.img_size -1).sub(0.5).mul(2)
            sampled = F.grid_sample(b_edges, sample_splines, mode='nearest', align_corners=True)

            ## filter out splines and strings that do not represent the edges found
            good_splines = sampled.sum(dim=-1).gt( n_samples * 0.9 )[0]
            splines = splines[None, good_splines]
            new_strs = new_strs[good_splines[0]]


            ## TEST ################################
            fig, ax = plt.subplots(dpi=150)
            ax.set_aspect('equal')
            ax.set_ylim(ax_max, 0)
            ax.set_xlim(0, ax_max)
            plt.axis('off')

            splines_tmp = splines.clone().add(0.5).mul(mag).cpu().numpy()
            for i in range(splines_tmp.shape[1]):
                ax.plot(splines_tmp[0,i,:,0], splines_tmp[0,i,:,1])

            img = b_edges[0].clone()
            topil = transforms.ToPILImage()
            if img.min() < -1e-4:
                img = img.add(1).div(2)
            img = topil(img.cpu())
            img = img.resize([ax_max, ax_max], resample=0)
            plt.imshow(img)

            plt.show(block=False)
            ########################################


            #### update tree_table; all old nodes in the tree get connected
            old_str_ids = [i for i in range(strs.size(0))]
            [tree_table.union(item, old_str_ids[ (i + 1) % strs.size(0)]) for i, item in enumerate(old_str_ids)]

            new_str_ids = [i + strs.size(0) for i in range(new_strs.size(0))]

            ## add new str_ids into uf as disconnected elems
            [tree_table.add(new_id) for new_id in new_str_ids]

            ## cat old strs and new strs
            strs = t.cat([strs, new_strs])

            print("no")

        return None























