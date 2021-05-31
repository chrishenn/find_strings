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

import ccpt.ccpt
from datasets.artificial_dataset import make_topbox, make_topbox_plus, make_nine, make_vbar, make_blob, make_tetris, make_topbar, make_circle_seed
from frnn_opt_brute import frnn_cpu
from string_finder import oodl_utils, oodl_draw
from string_finder.oodl_net import OONet_local
from string_finder.oodl_utils import regrid
from string_finder.str_draw import str_draw
from unionfind.unionfind import UnionFind

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_opt_brute/build/libfrnn_ts.so")
# t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_brute/build/libfrnn_ts.so")
# t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_tree_brute/build/libfrnn_ts.so")

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/stack_cols/build/libstack_cols.so")

t.manual_seed(7)

import scipy.interpolate as sinterp

# import torch.multiprocessing as mp
# import queue


def mmm(data):
    data = data.clone().float()
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
        # if self.scale > 1: images = F.interpolate(images, scale_factor=self.scale, mode='area')

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







class Pyramid_Strings(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # thresh_lo, thresh_hi = 0.5, 0.85
        thresh_lo, thresh_hi = 0.0, 0.1
        self.canny = Canny(opt, thresh_lo, thresh_hi)

        # self.py_sizes = [64, 32, 16]
        self.py_sizes = [256, 64]
        self.interp_mode = 'area'

        self.grids = list()
        for i,size in enumerate(self.py_sizes):
            D = t.arange(size, dtype=t.int)
            gridy, gridx = t.meshgrid(D, D)
            gridy = gridy[None, None, ...].float()
            gridx = gridx[None, None, ...].float()
            grid = t.stack([gridy, gridx], -1).add(0.5).repeat(self.opt.batch_size, 1,1,1,1)
            self.grids.append(grid)

        self.lin_ids = list()
        for i, grid in enumerate(self.grids):
            if i-1 >= 0:
                prev_max = self.lin_ids[i-1][-1,-1,-1,-1] + 1
            else: prev_max = 0

            base = grid[..., 0]
            b_ids = t.arange(base.numel())
            shape = base.shape
            b_ids = b_ids.reshape(shape).add(prev_max)

            self.lin_ids.append(b_ids)

        self.img_ids = list()
        for i, grid in enumerate(self.grids):
            imgid = grid[..., 0].clone()

            for b in range(imgid.size(0)):
                imgid[b].fill_(b)

            self.img_ids.append( imgid )


    def forward(self, batch):
        '''
        string format:
            strs[i] = [ end_lf[y,x], end_rt[y,x], str_norm[dy,dx], dev-frac ]
            where dev-frac gives a fraction of deviation of the string on that row, relative to its length
            parent_rowids gives the row of my parent string
        '''

        dev = batch.device
        t.cuda.set_device(dev)

        for i,grid in enumerate(self.grids): self.grids[i] = grid.cuda()
        for i,lin_id in enumerate(self.lin_ids): self.lin_ids[i] = lin_id.cuda()
        for i,img_id in enumerate(self.img_ids): self.img_ids[i] = img_id.cuda()

        ## at high res
        b_edges, sobel = self.canny(batch)
        # b_edges[b_edges.gt(1e-6)] = 1

        # oodl_utils.tensor_imshow(b_edges[0])
        # oodl_utils.tensor_imshow(b_edges[1])
        # oodl_utils.tensor_imshow(b_edges[2])

        ## to starting res
        base_size = self.py_sizes[0]
        mode = 'bilinear'
        b_edges, sobel = F.interpolate(b_edges, size=base_size, mode=mode, align_corners=True), F.interpolate(sobel, size=base_size, mode=mode, align_corners=True)
        b_edges = t.clamp(b_edges, 0,1)
        # b_edges[b_edges.gt(1e-6)] = 1

        oodl_utils.tensor_imshow(b_edges[0])
        oodl_utils.tensor_imshow(b_edges[1])
        oodl_utils.tensor_imshow(b_edges[2])
        # oodl_utils.tensor_imshow(b_edges[3])

        ## list of different sizes
        py_data = list()
        py_data.append((b_edges.clone(), sobel.clone()))
        for size in self.py_sizes[1:]:
            b_edges, sobel = F.interpolate(py_data[0][0], size=size, mode=self.interp_mode), F.interpolate(py_data[0][1], size=size, mode=self.interp_mode)
            # b_edges[b_edges.gt(1e-6)] = 1
            # b_edges[b_edges.lt(0.1)] = 0

            oodl_utils.tensor_imshow(b_edges[0])
            oodl_utils.tensor_imshow(b_edges[1])
            oodl_utils.tensor_imshow(b_edges[2])
            # oodl_utils.tensor_imshow(b_edges[3])

            py_data.append((b_edges.clone(), sobel.clone()))

        ## build hierarchy
        p_linids = t.empty(0, dtype=t.long, device=dev)
        my_linids = t.empty(0, dtype=t.long, device=dev)
        centers = t.empty([0,2], device=dev)
        norms = t.empty([0,2], device=dev)
        sizes = t.empty(0, device=dev)
        imgid = t.empty(0, device=dev)

        for i, (b_edges, sobel) in enumerate(py_data):
            my_size = b_edges.size(2)
            b_edges_b = b_edges.bool()

            ## one lower and one higher in pyramid
            if i+1 < len(py_data):
                parent_linids = F.interpolate(self.lin_ids[i + 1].float(), size=my_size, mode='nearest').long()
                parent_linids = parent_linids[b_edges_b]
            else:
                parent_linids = self.lin_ids[i][b_edges_b]

            object_linids = self.lin_ids[i][b_edges_b]
            str_sizes = t.ones_like(object_linids)

            ## scale centers to reference into the base image of size base_size
            transform_rat = base_size / my_size
            edge_centers = self.grids[i][b_edges_b].mul(transform_rat)
            edge_norms = sobel.permute(0,2,3,1)[b_edges_b.squeeze()]

            str_sizes = str_sizes.mul(transform_rat)

            im_id = self.img_ids[i][b_edges_b]

            p_linids = t.cat([p_linids, parent_linids])
            my_linids = t.cat([my_linids, object_linids])
            centers = t.cat([centers, edge_centers])
            norms = t.cat([norms, edge_norms])
            sizes = t.cat([sizes, str_sizes])
            imgid = t.cat([imgid, im_id])

        ## transform linids from pyramid for parents into the rows of those parents
        rowids = t.arange(my_linids.size(0), device=dev)
        p_rowids = t.empty(p_linids.max().int().item() + 1, dtype=t.long, device=dev)
        p_rowids.index_put_((my_linids,), rowids)
        p_rowids = p_rowids[p_linids]

        norms = norms.div(norms.norm(dim=1,keepdim=True))


        ## generate strings from centers and norms
        z_vec = t.tensor([[0.,0.,1.]], device=dev).repeat(norms.size(0), 1)
        rt_vecs = t.cross(t.cat([norms, t.zeros_like(norms[:,0,None])], -1), z_vec).mul(sizes[:,None]).div(2)[:,:2]

        locs_lf, locs_rt = centers.add(rt_vecs), centers.sub(rt_vecs)

        ## string format
        ## strs[i] = [ end_lf[y,x], end_rt[y,x], str_norm[dy,dx], dev-frac ]
        ## where dev-frac gives a fraction of deviation of the string on that row, relative to its length
        ## parent_rowids gives the row of my parent string
        # strings = t.cat([ locs_lf, locs_rt, norms, t.zeros_like(norms[:,0,None]) ], 1)

        # return strings, p_rowids

        ###### TEST ####################
        for im in [0,1]:
            str_draw(im, imgid, centers, norms, locs_lf, locs_rt, p_rowids, img=py_data[0][0][im], im_size=base_size)

        return None
        ################################

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

        ## detect canny-edges, connect them, select norms from canny sites
        batch_size = t.tensor(batch.size(0), device=dev)
        b_edges, sobel = self.canny(batch)
        b_edges[b_edges.gt(0.1)] = 1
        b_edges[b_edges.le(0.1)] = 0

        edge_ends = b_edges.nonzero()
        imgid = edge_ends[:,0]
        locs = edge_ends[:, -2:].float().contiguous()

        edges = t.ops.my_ops.frnn_ts_kernel(locs, imgid, t.tensor(np.sqrt(2)+0.01).cuda(), t.tensor(1).cuda(), batch_size)[0]

        ids = edge_ends.split(1, dim=1)
        norms = sobel[ ids[0], :, ids[2], ids[3] ]
        norms = norms.squeeze().div( norms.squeeze().norm(dim=1)[:,None] )

        ## compose initial strings
        ## strs[i] = [ end_lf[y,x], end_rt[y,x], str_norm[dy,dx], dev-frac ]
        locs_lf, locs_rt = locs[edges[:,0]], locs[edges[:,1]]

        norms_lf, norms_rt = norms[edges[:,0]], norms[edges[:,1]]
        norms = norms_lf.add( norms_rt ).div(2)

        strs = t.cat([ locs_lf, locs_rt, norms, t.zeros_like(edges[:,0,None]) ], 1)


        ##############################
        # temp = t.cat([locs_lf, locs_rt])
        # imgid_tmp = t.zeros_like(temp[:,0])
        # oodl_draw.oodl_draw(0, pts=locs, imgid=imgid, as_vecs=edges)


        ###### MERGING TREE
        ## for enough steps to reach some desired tree depth:
            ## for each neighboring pair of strings, generate a possible combined string.
            ## "Score": compute the score for strings that could possibly merge
                ## merging score is some tbd combination of similarity of [curvature, size ... ?] and representation of underlying data
        ### ################

        tree_table = UnionFind()
        [tree_table.add(i) for i in range(strs.size(0))]

        rads = [1, 2]

        ## TODO: support batching
        for i in range(len(rads)):

            all_ends = t.cat([strs[:,[0,1]], strs[:,[2,3]]])
            str_ids = t.arange(strs.size(0), device=dev).repeat(2)
            imgid = t.zeros_like(all_ends[:,0]).long()

            ## find edges from all string-ends
            end_edges = t.ops.my_ops.frnn_ts_kernel(all_ends, imgid, t.tensor( rads[i] ).cuda(), t.tensor(1).cuda(), batch_size)[0]

            str_edges = t.where(end_edges.ge(strs.size(0)), end_edges.sub(strs.size(0)), end_edges)

            ## filter ends that are part of the same string
            # same_str = str_edges[:,0].eq( str_edges[:,1] )
            #
            # ## filter ends that define a string that share an ancestor, or inherit from each other
            # related = t.tensor( [tree_table.connected(str_edge[0].item(), str_edge[1].item()) for str_edge in str_edges] )
            #
            # end_edges = end_edges[ (same_str | related.cuda()).logical_not() ]
            # str_edges = str_edges[ (same_str | related.cuda()).logical_not() ]


            ###### generate new strings from valid end_edges
            ## draw a "straight-line" between the farther endpoints (far_ends_lf/far_ends_rt) of the neighboring strings
            ## string deviation direction is referenced to string normal.

            close_ends_lf, close_ends_rt = all_ends[end_edges[:,0]], all_ends[end_edges[:,1]]

            strids_lf, strids_rt = str_ids[str_edges[:, 0]], str_ids[str_edges[:, 1]]

            other_endpt_cols_lf = t.where( end_edges[:,0].ge(strs.size(0))[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
            other_endpt_cols_rt = t.where( end_edges[:,1].ge(strs.size(0))[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
            far_ends_lf = strs[strids_lf].gather(1, other_endpt_cols_rt)
            far_ends_rt = strs[strids_rt].gather(1, other_endpt_cols_lf)

            ##### filter out zero-length string possibles
            seg_lens = F.pairwise_distance(far_ends_lf, far_ends_rt)
            not_coinc = seg_lens.gt(0.1)

            ## filter distances and endpoints
            seg_lens = seg_lens[not_coinc]
            str_edges = str_edges[not_coinc]
            close_ends_lf, close_ends_rt = close_ends_lf[not_coinc], close_ends_rt[not_coinc]
            far_ends_lf, far_ends_rt = far_ends_lf[not_coinc], far_ends_rt[not_coinc]

            ## filter str_ids
            strids_lf, strids_rt = strids_lf[not_coinc], strids_rt[not_coinc]

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

            ## make artificial norms at right-angles to vecs lf->rt
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

            ## make artificial dev_locs along string norm using dev_frac
            seg_centers = locs_lf.add(locs_rt).div(2)
            dev_locs = seg_centers + new_norms * seg_lens.mul(new_strs[:,-1])[:,None]


            ###### TEST ####################
            # mag = 25
            # h_size = 0.01 * mag * 36
            #
            # fig, ax = plt.subplots(dpi=150)
            # ax.set_aspect('equal')
            # ax.set_ylim(ax_max, 0)
            # ax.set_xlim(0, ax_max)
            # plt.axis('off')
            #
            # locs_lf_tmp, locs_rt_tmp = locs_lf.clone().cpu().numpy(), locs_rt.clone().cpu().numpy()
            # seg_center = (((locs_lf_tmp + locs_rt_tmp) / 2) + 0.5) * mag
            # old_norms = old_norms.clone().cpu().numpy() * mag
            # new_norms = new_norms.clone().cpu().numpy() * mag
            # dev_locs_tmp = dev_locs.clone().add(0.5).cpu().numpy() * mag
            #
            # palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
            # colors = t.arange(new_norms.shape[0])[:,None].mul(30).float().mul(palette).fmod(255).div(255)
            # colors = colors.numpy()
            # colors[:, 3] = 1
            #
            # ax.quiver(seg_center[:, 1], seg_center[:, 0], old_norms[:, 1], old_norms[:, 0], angles='xy', units='xy',
            #           scale=1, width=0.01 * mag,
            #           headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=colors)
            #
            # ax.quiver(seg_center[:, 1], seg_center[:, 0], new_norms[:, 1], new_norms[:, 0], angles='xy', units='xy',
            #           scale=1, width=0.01 * mag,
            #           headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=colors)
            #
            # locs_lf_tmp, locs_rt_tmp = (locs_lf_tmp + 0.5) * mag, (locs_rt_tmp + 0.5) * mag
            # locs_lf_tmp = np.stack([locs_lf_tmp[:, 1], locs_lf_tmp[:, 0]], axis=1)
            # locs_rt_tmp = np.stack([locs_rt_tmp[:, 1], locs_rt_tmp[:, 0]], axis=1)
            # lc = mc.LineCollection(list(zip(locs_lf_tmp, locs_rt_tmp)), linewidths=.01 * mag, color=colors)
            # ax.add_collection(lc)
            #
            # # ax.scatter(seg_center[:, 1], seg_center[:, 0], s=mag, alpha=1, marker="x", color=colors)
            # ax.scatter(dev_locs_tmp[:, 1], dev_locs_tmp[:, 0], s=mag, alpha=1, marker="x", color=colors)
            #
            # ax.scatter(locs_lf_tmp[:, 0], locs_lf_tmp[:, 1], s=mag, alpha=1, marker=".", color=colors)
            # ax.scatter(locs_rt_tmp[:, 0], locs_rt_tmp[:, 1], s=mag, alpha=1, marker=".", color=colors)
            #
            # plt.show(block=False)
            ################################



            #### make smooth splines

            ## transform each string's locs into coords aligned with their own string normals.
            norms = new_strs[:, 4:6]
            norms = t.cat([norms, t.zeros_like(norms[:,0,None])], 1)

            img_y = t.tensor([1.,0.,0.], device=dev)
            tran_cross = t.cross(img_y[None].repeat(norms.size(0),1), norms, dim=1)
            tran_s = tran_cross[:,-1]
            tran_c = norms.matmul(img_y)

            rot_mat = t.stack([t.stack([tran_c, -tran_s], 1), t.stack([tran_s, tran_c], 1)], 2)

            locs_lf = rot_mat.bmm(locs_lf[...,None]).squeeze()
            locs_rt = rot_mat.bmm(locs_rt[...,None]).squeeze()
            dev_locs = rot_mat.bmm(dev_locs[...,None]).squeeze()



            ###### TEST ####################
            # mag = 25
            # ax_max = self.opt.img_size * mag
            # h_size = 0.01 * mag * 36
            #
            # fig, ax = plt.subplots(dpi=150)
            # ax.set_aspect('equal')
            # plt.axis('off')
            #
            # palette = t.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1], dtype=t.long)
            # colors = t.arange(new_norms.shape[0])[:,None].mul(20).float().mul(palette).fmod(255).div(255)
            # colors = colors.numpy()
            # colors[:, 3] = 1
            #
            # seg_centers = locs_lf.add(locs_rt).div(2)
            # seg_centers = seg_centers.cpu().numpy() * mag
            #
            # norms = t.bmm(rot_mat, norms[:,:2, None]).squeeze()
            # norms = norms.cpu().numpy() * mag
            #
            # ax.quiver(seg_centers[:, 1], -seg_centers[:, 0], norms[:, 1], -norms[:, 0], angles='xy', units='xy',
            #           scale=1, width=0.01 * mag, headwidth=h_size, headlength=h_size+2, headaxislength=h_size+1, color=colors)
            #
            # locs_lf_tmp, locs_rt_tmp = locs_lf.clone().cpu().numpy() * mag, locs_rt.clone().cpu().numpy() * mag
            #
            # dev_locs_tmp = dev_locs.clone().cpu().numpy() * mag
            #
            # locs_lf_tmp = np.stack([locs_lf_tmp[:, 1], -locs_lf_tmp[:, 0]], axis=1)
            # locs_rt_tmp = np.stack([locs_rt_tmp[:, 1], -locs_rt_tmp[:, 0]], axis=1)
            # dev_locs_tmp = np.stack([dev_locs_tmp[:, 1], -dev_locs_tmp[:, 0]], axis=1)
            #
            # lc = mc.LineCollection(list(zip(locs_lf_tmp, locs_rt_tmp)), linewidths=.01 * mag, color=colors)
            # ax.add_collection(lc)
            #
            # ax.scatter(dev_locs_tmp[:, 0], dev_locs_tmp[:, 1], s=mag, alpha=1, marker="x", color=colors)
            # ax.scatter(locs_lf_tmp[:, 0], locs_lf_tmp[:, 1], s=mag, alpha=1, marker=".", color=colors)
            # ax.scatter(locs_rt_tmp[:, 0], locs_rt_tmp[:, 1], s=mag, alpha=1, marker=".", color=colors)
            #
            # plt.show(block=False)
            ################################


            ## interpolate splines
            n_samples = 10

            interp_x = t.linspace(0, 1, n_samples, device=dev).mul( locs_rt.sub(locs_lf).norm(dim=1,keepdim=True) )
            interp_x_np = interp_x.cpu().numpy()

            locs_x = t.stack([ locs_lf[:,1], dev_locs[:,1], locs_rt[:,1] ], 1)
            locs_y = t.stack([ locs_lf[:,0], dev_locs[:,0], locs_rt[:,0] ], 1)

            locs_x, sortids = locs_x.sort(dim=1)
            locs_y = locs_y.gather(1, sortids)

            least_x = locs_x[:, 0, None].clone()
            locs_x.sub_( least_x )

            least_y = locs_y[:,0,None].clone()
            locs_y.sub_( least_y )

            ## splines = [batch, string, samples, x, y]
            splines = t.zeros([batch_size, new_strs.size(0), n_samples, 2], device=dev)
            splines[0,:,:,0] = interp_x

            locs_x, locs_y = locs_x.cpu().numpy(), locs_y.cpu().numpy()
            for i in range(dev_locs.shape[0]):

                x, y = locs_x[i], locs_y[i]

                # spl = sinterp.make_interp_spline(x, y, k=3, bc_type='natural')
                # spl = sinterp.make_interp_spline(x, y, k=3, bc_type='clamped')
                spl = sinterp.make_interp_spline(x, y, k=2)

                interp_y = spl(interp_x_np[i])

                splines[0, i, :, 1] = t.from_numpy( interp_y ).to(dev)

            splines[0, :, :, 0].add_(least_x)
            splines[0, :, :, 1].add_(least_y)

            ## de-rotate splines
            splines = splines.squeeze().bmm(rot_mat.transpose(1,2))[None]

            ## sample values from b_edges using spline locations
            # sample_splines = splines.clone().div(self.opt.img_size -1).sub(0.5).mul(2)
            # rep_scores = F.grid_sample(b_edges, sample_splines, mode='nearest', align_corners=True)


            ###### TEST ####################
            # mag = 25
            # fig, ax = plt.subplots(dpi=150)
            # ax.set_aspect('equal')
            # plt.axis('off')
            #
            # splines_tmp = (splines.clone().cpu().numpy() + 0.5) * mag
            # for i in range(splines_tmp.shape[1]):
            #     ax.plot(splines_tmp[0,i,:,0], splines_tmp[0,i,:,1])
            #
            # ax_max = self.opt.img_size * mag
            # img = b_edges[0].clone()
            # topil = transforms.ToPILImage()
            # if img.min() < -1e-4:
            #     img = img.add(1).div(2)
            # img = topil(img.cpu())
            # img = img.resize([ax_max, ax_max], resample=0)
            # plt.imshow(img)
            #
            # plt.show(block=False)
            ####################



            #### score_2: similarity scores
            ## str_edges indexes into strs, two strs for each row of new_strs
            dev_lf, dev_rt = strs[str_edges[:,0], -1], strs[str_edges[:,1], -1]
            len_lf, len_rt = t.pairwise_distance( strs[str_edges[:,0]][:,[0,1]], strs[str_edges[:,0]][:,[2,3]] ), t.pairwise_distance( strs[str_edges[:,1]][:,[0,1]], strs[str_edges[:,1]][:,[2,3]] )
            norm_lf, norm_rt = strs[str_edges[:,0]][:,[4,5]], strs[str_edges[:,1]][:,[4,5]]

            dev_rat = dev_lf / (dev_rt + 1e-3)
            len_rat = len_lf / (len_rt + 1e-3)
            norm_dist = t.pairwise_distance( norm_lf, norm_rt )

            sim_scores = dev_rat + len_rat + norm_dist


            ## take best new strings
            new_strs = new_strs[None, sim_scores.gt(1)]
            splines = splines[0, sim_scores.gt(1)][None]


            ###### TEST: score_2 ####################
            mag = 25
            fig, ax = plt.subplots(dpi=150)
            ax.set_aspect('equal')
            plt.axis('off')

            splines_tmp = (splines.clone().cpu().numpy() + 0.5) * mag
            for i in range(splines_tmp.shape[1]):
                ax.plot(splines_tmp[0,i,:,0], splines_tmp[0,i,:,1])

            ax_max = self.opt.img_size * mag
            img = b_edges[0].clone()
            topil = transforms.ToPILImage()
            if img.min() < -1e-4:
                img = img.add(1).div(2)
            img = topil(img.cpu())
            img = img.resize([ax_max, ax_max], resample=0)
            plt.imshow(img)

            plt.show(block=False)
            ####################



            #### update tree_table; all old nodes in the tree get connected
            # old_str_ids = [i for i in range(strs.size(0))]
            # [tree_table.union(item, old_str_ids[ (i + 1) % strs.size(0)]) for i, item in enumerate(old_str_ids)]
            #
            # new_str_ids = [i + strs.size(0) for i in range(new_strs.size(0))]
            #
            # ## add new str_ids into uf as disconnected elems
            # [tree_table.add(new_id) for new_id in new_str_ids]
            #
            # ## cat old strs and new strs
            # # strs = t.cat([strs, new_strs.squeeze()])

            strs = new_strs.squeeze()

            print("no")

        return None



class Seg_Strings(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        thresh_lo, thresh_hi = 0.0, 0.1
        self.canny = Canny(opt, thresh_lo, thresh_hi)

        c_out = 256

        std = 0.1
        layer_1 = nn.Linear(70*5, 32)
        bn_1 = nn.BatchNorm1d(32, track_running_stats=False)
        rl_1 = nn.ReLU()
        layer_2 = nn.Linear(32, 64)
        bn_2 = nn.BatchNorm1d(64, track_running_stats=False)
        rl_2 = nn.ReLU()
        layer_3 = nn.Linear(64, 128)
        bn_3 = nn.BatchNorm1d(128, track_running_stats=False)
        rl_3 = nn.ReLU()
        layer_4 = nn.Linear(128, c_out)
        bn_4 = nn.BatchNorm1d(c_out, track_running_stats=False)
        rl_4 = nn.ReLU()
        nn.init.normal_(layer_1.weight.data, mean=0, std=std)
        nn.init.normal_(layer_2.weight.data, mean=0, std=std)
        nn.init.normal_(layer_3.weight.data, mean=0, std=std)
        nn.init.normal_(layer_4.weight.data, mean=0, std=std)

        self.layers = nn.Sequential(
            layer_1, bn_1, rl_1,
            layer_2, bn_2, rl_2,
            layer_3, bn_3, rl_3,
            layer_4, bn_4, rl_4,
        )

        bn = nn.BatchNorm1d(c_out, track_running_stats=False)
        fc = nn.Linear(c_out, opt.n_classes)
        relu = nn.ReLU()
        self.bn_fc_relu = nn.Sequential(bn, fc, relu)

    def avg_pool(self, data):
        tex, imgid, batch_size = data[0], data[1], data[-1]
        out = t.zeros([batch_size, tex.size(1)], dtype=tex.dtype, device=tex.device)
        out = out.index_add(0, imgid, tex)

        counts = t.zeros(batch_size, dtype=tex.dtype, device=tex.device).index_add_(0, imgid, t.ones_like(imgid).float())
        counts = t.where(counts.lt(1), t.ones_like(counts), counts)
        out = out.div(counts[:, None])
        return out

    def sum_pool(self, data):
        tex, imgid, batch_size = data[0], data[1], data[-1]
        out = t.zeros([batch_size, tex.size(1)], dtype=tex.dtype, device=tex.device)
        out = out.index_add(0, imgid, tex)
        return out

    def forward(self, batch):

        dev = batch.device
        batch_size = t.tensor(batch.size(0), device=dev)

        # for i,elem in enumerate(self.projectors): self.projectors[i] = elem.cuda()

        b_edges, sobel = self.canny(batch)

        ids = b_edges.nonzero()
        orientations = sobel.permute(0,2,3,1)[ids[:,[0,2,3]].split(1, dim=1)].squeeze()
        angles = t.atan2(orientations[:,0], orientations[:,1]).unsqueeze(1)
        imgid = ids[:,0].long().contiguous()
        locs = ids[:, -2:].float()

        pts = t.cat([locs, t.zeros_like(locs[:,0,None]), angles, t.ones_like(angles)], 1).contiguous()

        tex = batch[ids[:,0], :, ids[:,2], ids[:,3]]


        edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, t.tensor(np.sqrt(2) + 0.01).cuda(), t.tensor(1).cuda(), batch_size)[0]
        ccids = ccpt.ccpt.get_ccpt(edges, imgid).long()

        item_feat = t.cat([tex, orientations], 1)
        out_size = ccids.max().item()+1
        item_feat = t.ops.stack_op.stack_bind(item_feat, ccids, out_size)[0]

        nonempty = ccids.bincount().gt(0)
        nonempty_pad = F.pad(nonempty, [0,ccids.size(0)-nonempty.size(0)])

        imid_ccid = t.stack([imgid, ccids], 1).unique(dim=0)
        imgid = t.empty_like(ccids).index_put_((imid_ccid[:,1],), imid_ccid[:,0])[nonempty_pad]

        item_feat = item_feat[nonempty]

        data = self.layers(item_feat)
        data = self.avg_pool((data,imgid,batch_size))
        data = self.bn_fc_relu(data)
        return data




        # if self.opt.debug:
        #     oodl_draw.oodl_draw(0, pts=pts, imgid=imgid, img=b_edges[0], groupids=ccids, edges=edges)
        #     oodl_draw.oodl_draw(img=batch[0])
        #
        # break_30 = t.rand(edges.size(0)).gt(0.3)
        # edges = edges[break_30]
        #
        # ccids = ccpt.ccpt.get_ccpt(edges, imgid)
        #
        # if self.opt.debug:
        #     oodl_draw.oodl_draw(0, pts=pts, imgid=imgid, img=b_edges[0], groupids=ccids, edges=edges)




        # if self.opt.debug: return None
        # else: return out


        # out = self.net((tex,pts,imgid,batch_size))
        # edges = t.ops.my_ops.frnn_ts_kernel(locs, imgid, t.tensor(np.sqrt(2)+0.01).cuda(), t.tensor(1).cuda(), batch_size)[0]
        #
        # oodl_draw.oodl_draw(0, pts=locs, imgid=imgid, edges=edges, img=b_edges[0])
        # oodl_draw.oodl_draw(img=batch[0])



class Interp_String(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        thresh_lo, thresh_hi = 0.0, 0.1
        self.canny = Canny(opt, thresh_lo, thresh_hi)

    def forward(self, batch):
        dev = batch.device
        batch_size = t.tensor(batch.size(0), device=dev)

        b_edges, sobel = self.canny(batch)

        ids = b_edges.nonzero()
        orientations = sobel.permute(0, 2, 3, 1)[ids[:, [0, 2, 3]].split(1, dim=1)].squeeze()
        angles = t.atan2(orientations[:, 0], orientations[:, 1]).unsqueeze(1)
        imgid_all = ids[:, 0].long().contiguous()
        locs = ids[:, -2:].float()

        ## pts_all = [y, x, z, angle, size]
        pts_all = t.cat([locs, t.zeros_like(angles), angles, t.ones_like(angles)], 1).contiguous()

        ## "interpolate" real locations from 600x600 to 32x32
        pts_all[:,[0,1]] = pts_all[:,[0,1]].mul(0.05333333)

        edges_uni = t.ops.my_ops.frnn_ts_kernel(pts_all, imgid_all, t.tensor(1.5).cuda(), t.tensor(1).cuda(), batch_size)[0]
        edges_bi = t.cat([edges_uni, t.stack([edges_uni[:, 1], edges_uni[:, 0]], 1)])

        ## filter any ob's with a strictly-larger num of nebs than any one of their nebs
        n_nebs = t.zeros_like(imgid_all).index_add_(0, edges_bi[:,1], t.ones_like(edges_bi[:,1]))
        lf_gt = n_nebs[edges_uni[:,0]] > n_nebs[edges_uni[:,1]]
        rt_gt = n_nebs[edges_uni[:,0]] < n_nebs[edges_uni[:,1]]

        neb_gt = t.zeros_like(imgid_all).index_add_(0, edges_uni[:,0], lf_gt.long()).index_add_(0, edges_uni[:,1], rt_gt.long())
        neb_gt = neb_gt.bool().logical_not()

        pts_end = pts_all[neb_gt]
        imgid_end = imgid_all[neb_gt]
        oids_end = t.arange(pts_all.size(0))[neb_gt]

        ## if nebs in "same" location, choose one at random to drop
        edges_uni = t.ops.my_ops.frnn_ts_kernel(pts_end, imgid_end, t.tensor(1.2).cuda(), t.tensor(1).cuda(), batch_size)[0]

        locs = pts_end[:,[0,1]]
        locs_lf, locs_rt = locs[edges_uni[:,0]], locs[edges_uni[:,1]]
        dists = locs_lf.sub(locs_rt).pow(2).sum(dim=1)
        coinc_edges = dists.lt(0.9)
        drop_ids = edges_uni[:,0][coinc_edges]
        not_coinc = t.zeros_like(imgid_end).index_add_(0, drop_ids, t.ones_like(drop_ids).long()).bool().logical_not()

        pts_end = pts_end[not_coinc]
        imgid_end = imgid_end[not_coinc]
        oids_end = oids_end[not_coinc]


        # for i in [0,1,2,3]:
        #     oodl_draw.oodl_draw(i, pts=pts_end, imgid=imgid_end, draw_obj=True, max_size=32, img=batch[i])
        # for i in [4, 5]:
        #     oodl_draw.oodl_draw(i, pts=pts_all, imgid=imgid_all, draw_obj=True, max_size=32, img=batch[i])

        ######
        ## pts and imgid now give 'endpoints' for arcs ; oids_end gives the indexes of these endpoints in the pts_all set of all pts
        ######

        ######
        ## find equidistant nebs ('smoothly' distributed)


        ################# AVG METHOD

        ## endpts will always be active
        # active = t.rand(imgid_all.size(0)).lt(0.125).long().index_fill_(0, oids_end, 1)
        # oids = t.arange(active.size(0))
        # pinned = t.empty([0], dtype=t.long)
        #
        # for i in range(300):
        #     pts_act, imgid_act = pts_all[active.bool()], imgid_all[active.bool()]
        #     locs_act = pts_act[:, [0, 1]]
        #     oids_act = oids[active.bool()]
        #
        #     edges_uni = t.ops.my_ops.frnn_ts_kernel(pts_act, imgid_act, t.tensor(1.2).cuda(), t.tensor(1).cuda(), batch_size)[0]
        #     edges_bi = t.cat([edges_uni, t.stack([edges_uni[:, 1], edges_uni[:, 0]], 1)])
        #
        #     locs_lf, locs_rt = locs_act[edges_bi[:,0]], locs_act[edges_bi[:,1]]
        #     dists = F.pairwise_distance(locs_lf, locs_rt)
        #
        #     counts = t.zeros_like(pts_act[:, 0]).index_add_(0, edges_bi[:, 1], t.ones_like(edges_bi[:, 1]).float())
        #     av_dist = t.zeros_like(pts_act[:, 0]).index_add_(0, edges_bi[:,1], dists).div_(counts)
        #
        #     lo, hi = 0.98, 1.02
        #     active[oids_act[av_dist.lt(lo)]] = 0
        #
        #     pinned = t.cat([pinned, oids_act[av_dist.ge(lo) & av_dist.le(hi)] ])
        #     active[pinned] = 1
        #
        #     active[oids_end] = 1
        #
        #     ###################
        #     n_success = pinned.size(0)
        #
        #     pts_tmp, imgid_tmp = pts_all[active.bool()], imgid_all[active.bool()]
        #     if i % 100 == 0 or i + 1 == 300:
        #         for j in [0] :
        #
        #             oodl_draw.oodl_draw(j, pts=pts_tmp, imgid=imgid_tmp, draw_obj=True, max_size=32)
        #     ###################
        #
        #     active.index_fill_(0, t.randint(0, active.size(0), (active.size(0)//8,)), 1)







        ################# STRICT METHOD

        ## endpts will always be active
        active = t.rand(imgid_all.size(0)).lt(0.05).long().index_fill_(0, oids_end, 1)
        oids = t.arange(active.size(0))
        pinned = t.empty([0], dtype=t.long)

        for i in range(300):
            pts_act, imgid_act = pts_all[active.bool()], imgid_all[active.bool()]
            locs_act = pts_act[:, [0, 1]]
            oids_act = oids[active.bool()]

            edges_uni = t.ops.my_ops.frnn_ts_kernel(pts_act, imgid_act, t.tensor(1.2).cuda(), t.tensor(1).cuda(), batch_size)[0]

            locs_lf, locs_rt = locs_act[edges_uni[:,0]], locs_act[edges_uni[:,1]]
            dists = F.pairwise_distance(locs_lf, locs_rt)

            lo, hi = 0.95, 1.05

            passing = dists.gt(lo) & dists.lt(hi)
            passing = edges_uni[passing].flatten()
            passing = t.zeros_like(imgid_act).index_fill_(0, passing, 1)

            fail = dists.le(lo) | dists.ge(hi)
            fail = edges_uni[fail].flatten()
            passing[fail] = 0

            active[oids_act[fail]] = 0

            pinned = t.cat([pinned, oids_act[passing.bool()]])
            active[pinned] = 1

            active[oids_end] = 1

            ###################
            n_success = pinned.size(0)

            pts_tmp, imgid_tmp = pts_all[active.bool()], imgid_all[active.bool()]
            if i + 1 == 300:
                for j in [0,1,2,3] :
                    oodl_draw.oodl_draw(j, pts=pts_tmp, imgid=imgid_tmp, draw_obj=True, max_size=32)
            ###################

            active.index_fill_(0, t.randint(0, active.size(0), (int(active.size(0)*0.05),)), 1)


        pts_act, imgid_act = pts_all[active.bool()], imgid_all[active.bool()]
        for i in [0,1,2,3]:
            oodl_draw.oodl_draw(i, pts=pts_act, imgid=imgid_act, draw_obj=True, max_size=32, img=batch[i])

        print("no")













































