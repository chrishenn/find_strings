import atexit, math, os, time, PIL
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
# from nearpy.distances import EuclideanDistance
# from nearpy.filters import NearestFilter, DistanceThresholdFilter

from numba import njit, prange

import torch as t
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as transforms

from datasets.artificial_dataset import make_topbox, make_topbox_plus, make_nine, make_vbar, make_blob, make_tetris, make_topbar, make_circle_seed
from frnn_opt_brute import frnn_cpu
from oomodel import oodl_utils
from oomodel.oodl_utils import regrid

# t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_opt_brute/build/libfrnn_ts.so")
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







def store_sparse_mp_producer(work_q, staged_q, mem_size_tuple, rel_vec_width, lin_rad, mem_width, n_rolls):

    while not work_q.empty():

        try:
            data = work_q.get(timeout=1)

            rel_vec = build_rel(data, rel_vec_width, lin_rad, mem_width)
            if n_rolls > 0: rel_vec = roll_rel(rel_vec, n_rolls)

            mem_to_add = sparse.COO(coords=rel_vec.T, data=1, shape=mem_size_tuple)
            staged_q.put(mem_to_add)

        except: continue


def store_sparse_mp_consumer(staged_q, mem_q, mem_lock):

    while True:
        try:
            mem_to_add = staged_q.get(timeout=1)
            mem = mem_q.get(timeout=1)

            with mem_lock:
                mem += mem_to_add

            mem_q.put(mem)

            staged_q.task_done()

        except queue.Empty: continue

        except Exception as e:
            print('got unhandled exception in mem-storage consumer thread', e)
            return






def query_ann_k_worker(work_q, return_dict, k, index_size, ann_path, ann_type):
    ann_index = AnnoyIndex(index_size, ann_type)
    ann_index.load(ann_path, prefault=False)

    while not work_q.empty():
        try:
            chunk_id, queries = work_q.get(timeout=1)
            queries = queries.numpy()

            k_nebs = list()
            for i, vec in enumerate(queries):
                k_nebs.append(ann_index.get_nns_by_vector(vec, k, include_distances=True))

            return_dict[chunk_id] = k_nebs

        except: continue






def build_rel_mp_worker(work_q, staged_q, rel_vec_width, lin_rad, mem_width, n_rolls):

    while not work_q.empty():
        try:
            data = work_q.get_nowait()
            tex,pts,imgid = data

            rel_vec = build_rel(data, rel_vec_width, lin_rad, mem_width)
            if n_rolls > 0: rel_vec = roll_rel(rel_vec, n_rolls)

            rel_vec = t.from_numpy( rel_vec )
            if n_rolls > 0: pts_out = pts.repeat(n_rolls, 1)
            else: pts_out = pts

            staged_q.put( (rel_vec, pts_out) )

        except: continue






def build_rel(data, rel_vec_width, lin_rad, mem_width):
    tex,pts,imgid = data

    # rel_vec sizes, edges, vecs, tex's
    edges = frnn_cpu.frnn_cpu(pts.numpy(), imgid.numpy(), lin_rad)
    edges = t.from_numpy(edges)
    edges = t.cat([edges, t.stack([edges[:,1], edges[:,0]], 1)], 0)

    ## vecs in rel_vec will bin to [0, mem_width)
    locs = pts[:, :2].clone()
    locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
    vecs = locs_rt - locs_lf
    vecs.div_(lin_rad)
    vecs.add_(1)
    vecs.div_(2)
    vecs.mul_(mem_width)
    vecs.round_()

    tex = tex.round().to(t.int32)
    tex_rt = tex[edges[:, 1]]

    args = [tex, tex_rt, vecs, edges[:, 0].contiguous()]
    args = [te.numpy() for te in args]

    rel_vec = build_rel_cpu( rel_vec_width, *args )
    return rel_vec

@njit
def build_rel_cpu(rel_vec_width, tex, tex_rt, vecs, rowids):
    write_cols = np.ones(tex.shape[0],dtype=np.int32)
    rel_vec = np.zeros( (tex.shape[0], rel_vec_width),dtype=np.int32)

    rel_vec[:,0] = tex[:,0].copy()

    n_cols = rel_vec.shape[1]
    n_vecs = vecs.shape[0]
    for vec_i in range(n_vecs):

        o_id = rowids[vec_i]
        col = write_cols[o_id]

        if col+2 >= n_cols: continue

        write_cols[o_id] += 3

        vec = vecs[vec_i]
        vec_tex = tex_rt[vec_i,0]

        rel_vec[o_id, col + 0] = vec[0]
        rel_vec[o_id, col + 1] = vec[1]
        rel_vec[o_id, col + 2] = vec_tex

    return rel_vec


def roll_rel(rel_vec, n_roll):

    rolled_rel = np.empty_like(rel_vec).repeat(n_roll, axis=0)
    for i in range(n_roll):

        roll_tmp = rel_vec.copy()
        home_tex = roll_tmp[:,0]
        roll_vecs = np.roll(roll_tmp[:, 1:], 3, axis=1)
        roll_tmp[:,0] = home_tex
        roll_tmp[:,1:] = roll_vecs

        rolled_rel[i * rel_vec.shape[0] : (i+1) * rel_vec.shape[0], :] = roll_tmp

    return rolled_rel


def roll_rel_torch(rel_vec, n_roll):

    rolled_rel = t.empty_like(rel_vec).repeat(n_roll, 1)
    for i in range(n_roll):

        roll = rel_vec.clone()
        home_tex = roll[:,0]
        roll_vecs = roll[:, 1:].roll(3, dims=1)
        roll[:,0] = home_tex
        roll[:, 1:] = roll_vecs

        rolled_rel[i * rel_vec.size(0) : (i+1) * rel_vec.size(0), :] = roll

    return rolled_rel







## Deep_mem rel hash
class Deep_Mem(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) / 2
        self.train_flag = True
        self.opt = opt

        self.neb_width = 29
        self.neb_size = self.neb_width**2

        self.unfold = nn.Unfold(self.neb_width, padding=self.neb_width//2)

        self.calls = 0
        self.row_ptr = 0
        self.n_coalesce = 0

        # self._forward = self.stack_vecs

        # self._forward = self.build_ann

        self.init_recall_ann()
        self._forward = self.recall_ann



    def shutdown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()

        if hasattr(self, 'proc_pool'):
            self.proc_pool.terminate()

        if hasattr(self, 'procs'):
            [proc.kill() for proc in self.procs if proc is not None]

        t.set_num_threads(2)

    def forward(self, batch): return self._forward(batch)


    def init_store_sparse(self):
        self.manager = manager = mp.Manager()
        self.staged_q = manager.Queue()
        self.work_q = manager.Queue()

        n_procs_total = os.cpu_count()
        self.proc_pool = mp.Pool(n_procs_total)
        atexit.register(self.shutdown)

        self.n_procs = n_procs_total
        for i in range(self.n_procs):
            self.proc_pool.apply_async(store_sparse_mp_producer,
                                       args=(self.work_q, self.staged_q, self.mem_size_tuple,self.rel_vec_width.item(),self.lin_rad.item(),self.mem_width.item(), self.n_rolls.item()) )

    def init_recall_ann(self):
        self.n_procs = os.cpu_count()

        self.manager = manager = mp.Manager()
        self.staged_q = staged_q = manager.Queue()
        self.work_q = manager.Queue()

        self.proc_pool = mp.Pool(self.n_procs)
        atexit.register(self.shutdown)

        t.set_num_threads(2)

    def filter_sample(self, batch):
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1

        if not hasattr(self, 'sampler'): self.sampler = OOSampler(self.opt).to(self.opt.gpu_ids[0])

        sizes = t.zeros([1], dtype=t.int)
        data = self.sampler(batch, sizes)
        return data

    def filter_thresh(self, batch):
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1
        return batch


    ## store
    def stack_vecs(self, batch):
        block_size = 1000000

        if not hasattr(self, 'config_stack'):
            config_stack = t.empty([block_size, self.neb_size], dtype=t.uint8)
            self.register_buffer('config_stack', config_stack)

            values = t.ones(block_size, dtype=t.long)
            self.register_buffer('values', values)

        batch = self.filter_thresh(batch)
        batch = self.unfold(batch).clone()
        batch = batch.to(t.uint8)
        rel_vecs = batch.permute(0,2,1).reshape(-1, self.neb_size)

        # self.config_stack = t.cat([self.config_stack, rel_vecs])

        ###
        row_ptr = self.row_ptr
        if row_ptr+rel_vecs.size(0) > self.config_stack.size(0):
            self.config_stack = t.cat([self.config_stack, t.empty([block_size, self.neb_size], dtype=t.int)])
            self.values = t.cat([self.values, t.ones(block_size, dtype=t.long)])

        self.config_stack[row_ptr : row_ptr+rel_vecs.size(0), :] = rel_vecs

        self.row_ptr += rel_vecs.size(0)

        # self.calls += 1
        # if self.calls % 200 == 0:
        #     self.coalesce()

    def coalesce(self):
        print('coalescing configs')
        self.config_stack = self.config_stack[:self.row_ptr, :]
        self.values = self.values[:self.row_ptr]

        self.config_stack, old_ids, new_values = self.config_stack.unique(dim=0, return_counts=True, return_inverse=True)

        if self.n_coalesce > 0:
            new_values.index_add_(0, old_ids, self.values)
        self.values = new_values

        self.row_ptr = self.values.size(0)
        self.n_coalesce += 1





    ## build
    def build_ann(self, batch):

        print('ann index build start ...')
        path = os.path.join(self.opt.save_dir, self.opt.ann_filename)
        ann_index = AnnoyIndex(self.neb_size, self.opt.ann_type)
        ann_index.on_disk_build(path)

        for i,vec in enumerate( self.config_stack.cpu().numpy() ):
            ann_index.add_item(i, vec)

        ann_index.build(self.neb_size, n_jobs=-1)
        ann_index.save(path)

        print('ann index saved at \n', path)
        print('exiting ...')
        exit(0)



    ## recall
    def recall_ann(self, batch):

        print("\ncompositing the response from", batch.size(0), "seeded images onto a prediction image")
        print("MAKE SURE EACH SEED IMAGE IN THIS BATCH IS THE SAME IF THE BATCH SIZE IS > 1")

        k = 40

        seed = batch.clone()
        ymask = t.arange(seed.size(2)) - self.img_size//2
        seed[:,:,  (ymask > -4) & (ymask < 6), :] = 0

        # oodl_utils.tensor_imshow(seed[0], dpi=150)

        # topil = transforms.ToPILImage()
        # seed = topil(seed[0])
        # seed = transforms.functional.affine(seed, translate=[0,0], angle=0, scale=1, shear=0, resample=PIL.Image.BICUBIC)
        # totensor = transforms.ToTensor()
        # seed = totensor(seed)[None]

        # seed = make_topbar(self.opt)
        # seed = make_nine(self.opt)
        # seed = make_circle_seed(self.opt)

        ##
        seed = self.filter_thresh(seed)
        rel_vecs = self.unfold(seed).clone()
        rel_vecs = rel_vecs.permute(0,2,1).reshape(-1, self.neb_size)
        pts = self.gen_full_grid()

        k_confs = self.query_ann_mp(rel_vecs, k)
        # k_confs = self.query_ann(rel_vecs, k)


        d_thresh_lo = 0
        d_thresh_hi = 20
        conf_thresh = 0

        configs_write, confidences_write, pts_write = self.filter_configs(k_confs, pts, d_thresh_lo, d_thresh_hi, conf_thresh)
        compt = self.render_unroll(configs_write, confidences_write, pts_write, negative=False, weight=True, avg=True)
        compt_save = compt.clone()

        compt = compt_save.clone()
        compt.div_(compt.max())
        compt.squeeze_()
        compt = t.stack([t.zeros_like(compt), t.zeros_like(compt), compt], 0)
        compt[0, ...] = seed[0, 0]
        compt.clamp_(0, 0.05)
        oodl_utils.tensor_imshow(compt, dpi=150)

        compt = compt_save.clone()
        compt = compt.clone().cpu().numpy()[0,0]
        compt = compt / compt.max()

        print("no")




    def build_rel(self, batch):

        rel_vec, pts = self.build_rel_one(batch)
        rel_vec = self.roll_rel(rel_vec, self.n_rolls)

        pts = pts.repeat(self.n_rolls, 1)

        return rel_vec, pts

    def build_rel_one(self, batch):

        data = self.filter_sample(batch)
        tex, pts, imgid, batch_size = data

        edges = frnn_cpu.frnn_cpu(pts.numpy(), imgid.numpy(), self.lin_rad.item())
        edges = t.from_numpy(edges)

        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])
        edges = edges.to(self.opt.gpu_ids[0])

        ## vecs in rel_vec will bin to [0, mem_width)
        locs = pts[:, :2].clone()
        locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
        vecs = locs_rt - locs_lf
        vecs.div_(self.lin_rad)
        vecs.add_(1)
        vecs.div_(2)
        vecs.mul_(self.mem_width)
        vecs.round_()
        vecs = vecs.to( t.int32 )

        tex = tex.round().to( t.int32 )
        tex_rt = tex[edges[:,1]]

        args = [tex, tex_rt, vecs, edges[:, 0].contiguous() ]
        args = [te.numpy() for te in args]
        rel_vec = t.from_numpy( build_rel_cpu( self.rel_vec_width.item(), *args ) )

        return rel_vec, pts

    def build_rel_one_gpu(self, batch):

        data = self.filter_sample(batch)
        tex, pts, imgid, batch_size = data

        edges = t.ops.my_ops.frnn_ts_kernel(pts.cuda(), imgid.cuda(), self.lin_rad.cuda(), t.tensor(1).cuda(), batch_size.cuda())[0]
        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])

        ## vecs in rel_vec will bin to [0, mem_width)
        locs = pts[:, :2].clone()
        locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
        vecs = locs_rt - locs_lf
        vecs.div_(self.lin_rad)
        vecs.add_(1)
        vecs.div_(2)
        vecs.mul_(self.mem_width-1)
        vecs.round_()
        vecs = vecs.to( t.int32 )

        tex = tex.round().to( t.int32 )
        tex_rt = tex[edges[:,1]]

        t.ops.row_op.write_row_bind(self.rel_vec, self.write_cols, tex.cuda(), tex_rt.cuda(), vecs.cuda(), edges[:,0].cuda().contiguous())

        return self.rel_vec.clone().cpu(), pts

    def roll_rel(self, rel_vec, pts, n_roll):

        rolled_rel = t.empty_like(rel_vec).repeat(n_roll+1, 1)

        rolled_rel[ : rel_vec.size(0), :] = rel_vec.clone()

        for i in range(1, n_roll+1):

            roll = rel_vec.clone()
            home_tex = roll[:,0]
            roll_vecs = roll[:, 1:].roll(3, dims=1)
            roll[:, 0] = home_tex
            roll[:, 1:] = roll_vecs

            rolled_rel[i * rel_vec.size(0) : (i+1) * rel_vec.size(0), :] = roll

        pts = pts.repeat(n_roll+1, 1)

        return rolled_rel, pts



    def filter_configs(self, k_confs, pts, d_thresh_lo, d_thresh_hi, conf_thresh):

        # config_vecs = t.from_numpy( self.mem.coords.T )
        # values = t.from_numpy( self.mem.data )

        config_vecs = self.config_stack
        values = self.values

        ids_list, dists_list, pts_list = list(), list(), list()
        for i in range(len(k_confs)):
            ids, dists = k_confs[i]

            ids, dists = t.tensor(ids), t.tensor(dists)
            ids_list.append(ids); dists_list.append(dists); pts_list.append(pts[None,i,:].repeat(ids.size(0), 1))
        ids, dists, pts = t.cat(ids_list), t.cat(dists_list), t.cat(pts_list)

        ##
        dists_mask = dists.gt(d_thresh_lo) & dists.lt(d_thresh_hi)
        ids = ids[dists_mask].long()
        pts = pts[dists_mask]

        confidences = values[ids]
        confidences_mask = confidences.gt(conf_thresh)
        confidences_write = confidences[confidences_mask]

        configs = config_vecs[ids]
        configs_write = configs[confidences_mask]

        pts_write = pts[confidences_mask]

        return configs_write, confidences_write, pts_write

    def render_config(self, configs, confidences, pts, avg=False):

        pred_vals = configs[:, t.arange(0, configs.size(1), 3)]

        # pred_vals = pred_vals.mul(2).sub(1)
        pred_vals = pred_vals.mul(confidences[:,None])
        # pred_vals = pred_vals.mul(2).sub(1).mul(confidences[:,None])

        pred_vals = t.cat(pred_vals.split(1, dim=1))

        cols = t.arange(configs.size(1))
        vecs_0 = configs[:, cols[cols.fmod(3).ne(0)]]
        vecs_0 = t.cat(vecs_0.split(2, dim=1))

        vecs_0 = t.cat([t.ones_like(pts), vecs_0])

        vecs = vecs_0.float()
        vecs.div_(self.mem_width).mul_(2).sub_(1).mul_(self.lin_rad)

        locs = pts[:,:2].clone().repeat(self.n_nebs_hashed + 1, 1)
        pred_locs = locs.add(vecs)
        pred_locs[:pts.size(0), :] = pts

        ## drop padded vecs
        not_pad = vecs_0[:, 0].ne(0) | vecs_0[:, 1].ne(0)
        pred_locs = pred_locs[not_pad]
        pred_vals = pred_vals[not_pad]

        pred_locs.clamp_(0, self.img_size - 1)

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=avg)
        return comp


    def render_unroll(self, configs, confidences, pts, negative=False, weight=False, avg=False):

        locs = pts[...,None].repeat(1, 1,configs.size(1))

        D1 = t.arange(self.neb_width, dtype=t.int)
        D2 = t.arange(self.neb_width, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        offsets = t.stack([gridy, gridx], 2).reshape(-1,2)
        offsets = offsets.permute(1, 0).to(pts.device)

        locs = locs.add(offsets).permute(0,2,1).reshape(-1,2)
        tex = configs.reshape(-1, 1)
        tex = tex.float()

        if negative:
            tex.mul_(2)
            tex.sub_(1)

        if weight:
            confidences = confidences[:,None].repeat(self.neb_size, 1)
            confidences = confidences.float()
            confidences.div_(confidences.max())
            tex.mul_(confidences)

        locs.sub_(self.neb_width//2)

        bounds_mask = locs[:,0].lt(self.img_size) & locs[:,1].lt(self.img_size)  &  locs[:,0].ge(0) & locs[:,1].ge(0)

        locs = locs[bounds_mask]
        tex = tex[bounds_mask]

        imgid = t.zeros_like(locs[:, 0]).long()
        comp = regrid(tex, locs, imgid, self.img_size, avg=avg)
        return comp



    def query_ann_mp(self, queries, k):

        return_dict = self.manager.dict()

        query_list = queries.chunk(self.n_procs, dim=0)
        for i, chunk in enumerate(query_list):
            self.work_q.put( (str(i), chunk) )

        ann_path = os.path.join(self.opt.save_dir, self.opt.ann_filename)

        results = list()
        for i in range(self.n_procs):
            res = self.proc_pool.apply_async(query_ann_k_worker, args=(self.work_q, return_dict, k, self.neb_size, ann_path, self.opt.ann_type))
            results.append(res)
        [res.wait() for res in results]

        out = list()
        for i in range(len(return_dict)):
            out.extend( return_dict[str(i)] )
        return out


    def build_rel_mp(self, batch):

        tex,pts,imgid,batch_size = self.filter_sample(batch)

        n_procs = min(batch_size.item(), self.n_procs)
        im_per_proc = ((batch_size.item()-1) // n_procs) + 1
        split_size = 812 * im_per_proc
        tex, pts, imgid = tex.split(split_size), pts.split(split_size), imgid.split(split_size)

        for i in range(len(tex)):
            self.work_q.put( (tex[i], pts[i], imgid[i]) )

        results = list()
        for i in range(n_procs):
            res = self.proc_pool.apply_async(build_rel_mp_worker,
                                             args=(self.work_q, self.staged_q, self.rel_vec_width.item(), self.lin_rad.item(), self.mem_width.item(), self.n_rolls))
            results.append(res)

        [res.wait() for res in results]

        rel_out = list()
        pts_out = list()
        for i in range(self.staged_q.qsize()):
            rel, pts = self.staged_q.get()
            rel_out.append( rel )
            pts_out.append( pts )
        rel, pts = t.cat(rel_out), t.cat(pts_out)
        return rel, pts


    def gen_full_grid(self):
        D1 = t.arange(self.opt.img_size, dtype=t.int)
        D2 = t.arange(self.opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        gridy = gridy[None, None, ...].float()
        gridx = gridx[None, None, ...].float()

        pts = t.cat([gridy, gridx], 1)
        pts = pts.squeeze().permute(1,2,0)
        pts = pts.reshape(-1, pts.size(-1))
        pts = pts.repeat(self.opt.batch_size, 1)

        return pts




    def store_sparse_mp(self, batch):

        data = self.filter_sample(batch)
        tex, pts, imgid, batch_size = data

        n_procs = min(batch_size.item(), self.n_procs)
        im_per_proc = ((batch_size.item()-1) // n_procs) + 1
        split_size = 812 * im_per_proc
        tex, pts, imgid = tex.split(split_size), pts.split(split_size), imgid.split(split_size)

        for i in range(len(tex)):
            self.work_q.put( (tex[i], pts[i], imgid[i]) )

    def coalesce_MP(self):

        print('processing intermediates from images ...')
        self.proc_pool.close()
        self.proc_pool.join()

        print('coalescing memory ...')
        while not self.staged_q.empty():
            try: self.mem += self.staged_q.get_nowait()
            except: continue

    def store_torch_coo(self, batch):

        data, rel_vec0 = self.build_rel_one(batch)
        rolled_rel0 = self.roll_rel(rel_vec0, 5)

        data, rel_vec1 = self.build_rel_one(batch)
        rolled_rel1 = self.roll_rel(rel_vec1, 5)

        rel_vec = t.cat([rolled_rel0, rolled_rel1])

        # #### torch sparse coo
        mem_to_add = t.sparse_coo_tensor(indices=rel_vec.clone().T, values=t.ones_like(rel_vec[:, 0]).long(), size=tuple(self.mem_size))
        self.mem.add_(mem_to_add)

        self.calls += 1
        if self.calls % 10 == 0:
            self.mem = self.mem.coalesce()

            # assert self.mem.indices().min() >= 0

    def store_sparse_coo(self, batch):

        data, rel_vec0 = self.build_rel_one(batch)
        rolled_rel0 = self.roll_rel(rel_vec0, 3)

        data, rel_vec1 = self.build_rel_one(batch)
        rolled_rel1 = self.roll_rel(rel_vec1, 3)

        rel_vec = t.cat([rolled_rel0, rolled_rel1])

        ##
        mem_to_add = sparse.COO(coords=rel_vec.T.cpu().numpy(), data=1, shape=self.mem_size_tuple)
        self.mem_staged.append(mem_to_add)

        self.calls += 1
        if self.calls % 100 == 0:
            self.coalesce_staged()

    def forward_recall_torch_coo(self, batch):
        self.mem = self.mem.coalesce()

        # ann_index = AnnoyIndex(self.mem_size.size(0), 'euclidean')
        # path = '/home/chris/Documents/deep_mem/checkpoints/rel_sparse_annoy/mnist_ep1.ann'
        # ann_index.load(path, prefault=False)
        # self.ann_index = ann_index

        batch[0,0][t.arange(batch.size(2),device=batch.device).gt(self.img_size//2-2)] = 0
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1
        seed = batch
        # seed = t.zeros_like(batch)
        # seed = make_topbox(self.opt).to(batch.device)
        # seed = make_topbar(self.opt).to(batch.device)

        comp = self.recall_nn(seed, n_comp=20)

        oodl_utils.tensor_imshow(seed[0], dpi=150)
        oodl_utils.tensor_imshow(comp[0], dpi=150)
        comp_tmp = comp.cpu().numpy()[0, 0]

        comp.mean(), comp.max()

        comp_tmp = comp.clamp(0, comp.mean().item()*3)
        oodl_utils.tensor_imshow(comp_tmp[0], dpi=150)

        comp_tmp = comp.clone()
        comp_tmp[0,0][t.arange(comp.size(2)) > 26] = 0
        comp_tmp.clamp_(0,650)
        oodl_utils.tensor_imshow(comp_tmp[0], dpi=150)

        print("no")

    def recall_inc(self, batch, comp=None):

        (tex, pts, imgid), edges, vecs, rel_vec = self.build_rel_one(batch)

        pred_config = rel_vec.clone()
        init_conf = self.query_many(pred_config)

        ## if I can make a change to a single value in a config that increases the confidence, write just that change to the neighborhood
        ## the resulting prediction will not include the values from the seed - because changing them would not increase the conf of the neighborhood

        ## build configs - each with a single change
        configs = t.empty([pred_config.size(0) * self.n_nebs_hashed, pred_config.size(1)], dtype=pred_config.dtype, device=pred_config.device)

        for i, col in enumerate(range(3, self.rel_vec_width.item(), 3)):

            config_flipped = pred_config.clone()
            config_flipped[:, col] = t.where(config_flipped[:, col].eq(0), t.ones_like(config_flipped[:, col]), t.zeros_like(config_flipped[:, col]))

            configs[pred_config.size(0) * i : pred_config.size(0) * (i+1), :] = config_flipped

        confs = self.query_many(configs)

        good_flip = confs.gt( init_conf.repeat(self.n_nebs_hashed) )

        gflip_rows = t.arange(good_flip.size(0),device=good_flip.device)
        gflip_cols = t.arange(confs.size(0),device=confs.device).floor_divide(pred_config.size(0)).mul(3).add(1)[:,None].repeat(1,2)
        gflip_cols[:,1].add_(1)
        gflip_rows, gflip_cols = gflip_rows[good_flip], gflip_cols[good_flip]

        ## select vecs where a flip has increased the confidence, values that caused the increase
        vecs_0 = configs[(gflip_rows[:,None], gflip_cols)]

        ## TODO: these '0' vals are a vote for lower value at loc
        ## TODO: weight these votes by config confidence?
        vals = configs[(gflip_rows, gflip_cols[:,1].add(1))]

        ## transform relative vecs to locations
        locs = pts[:,:2].repeat(self.n_nebs_hashed, 1)[good_flip]

        vecs = vecs_0.float()
        vecs.div_(self.mem_width-1).mul_(2).sub_(1).mul_(self.lin_rad)

        pred_locs = locs.add(vecs)

        ## drop padded vecs
        not_pad = vecs_0[:,0].ne(0) | vecs_0[:,1].ne(0)
        pred_locs = pred_locs[not_pad]
        pred_vals = vals[not_pad]

        if pred_locs.size(0) > 0:
            assert pred_locs.min() >= -1e-5
            assert pred_locs.max() < self.img_size

        pred_locs.clamp_(0, self.img_size-1)

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=False, batch=comp)
        return comp

    def recall_hash(self, batch, comp=None):
        (tex, pts, imgid), edges, vecs, rel_vec = self.build_rel_one(batch)

        pred_config = rel_vec.clone()
        init_conf = self.query_ann(pred_config)

        ## build configs - each with a single change
        configs = t.empty([pred_config.size(0) * self.n_nebs_hashed, pred_config.size(1)], dtype=pred_config.dtype, device=pred_config.device)

        for i, col in enumerate(range(3, self.rel_vec_width.item(), 3)):
            config_flipped = pred_config.clone()
            config_flipped[:, col] = t.where(config_flipped[:, col].eq(0), t.ones_like(config_flipped[:, col]), t.zeros_like(config_flipped[:, col]))

            configs[pred_config.size(0) * i: pred_config.size(0) * (i + 1), :] = config_flipped

        confs = self.query_ann(configs)

        good_flip = confs.gt(init_conf.repeat(self.n_nebs_hashed))

        gflip_rows = t.arange(good_flip.size(0), device=good_flip.device)
        gflip_cols = t.arange(confs.size(0), device=confs.device).floor_divide(pred_config.size(0)).mul(3).add(1)[:, None].repeat(1, 2)
        gflip_cols[:, 1].add_(1)
        gflip_rows, gflip_cols = gflip_rows[good_flip], gflip_cols[good_flip]

        ## select vecs where a flip has increased the confidence, values that caused the increase
        vecs_0 = configs[(gflip_rows[:, None], gflip_cols)]

        vals = configs[(gflip_rows, gflip_cols[:, 1].add(1))]
        vals.mul_(2).sub_(1).mul_( init_conf.repeat(self.n_nebs_hashed)[gflip_rows] )
        vals[vals.lt(0)] = 0

        ## transform relative vecs to locations
        locs = pts[:, :2].repeat(self.n_nebs_hashed, 1)[good_flip]

        vecs = vecs_0.float()
        vecs.div_(self.mem_width - 1).mul_(2).sub_(1).mul_(self.lin_rad)

        pred_locs = locs.add(vecs)

        ## drop padded vecs
        not_pad = vecs_0[:, 0].ne(0) | vecs_0[:, 1].ne(0)
        pred_locs = pred_locs[not_pad]
        pred_vals = vals[not_pad]

        if pred_locs.size(0) > 0:
            assert pred_locs.min() >= -1e-5
            assert pred_locs.max() < self.img_size

        pred_locs.clamp_(0, self.img_size - 1)

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=False, batch=comp)
        return comp

    def query_ann_k_0(self, queries, k):

        queries = queries.cpu().numpy()

        k_nebs = list()
        for i,vec in enumerate(queries):

            k_nebs.append( self.ann_index.get_nns_by_vector(vec, k, include_distances=True) )

        return k_nebs



    def query_ann(self, queries, k):
        ann_path = os.path.join(self.opt.save_dir, self.opt.ann_filename)

        ann_index = AnnoyIndex(self.neb_size, self.opt.ann_type)
        ann_index.load(ann_path, prefault=False)

        k_nebs = list()
        for i, vec in enumerate(queries):
            k_nebs.append(ann_index.get_nns_by_vector(vec, k, include_distances=True))

        return k_nebs





    def query_hash(self, queries):

        dev = queries.device
        queries = queries.cpu().numpy()

        true_values = np.empty_like(queries[:,0]).astype( queries.dtype )
        hash_vals = np.empty_like(queries[:,0]).astype( queries.dtype )

        for i,vec in enumerate(queries):
            true_values[i] = self.mem[tuple(vec)]
            list_res = self.engine.neighbours(vec)

            if len(list_res) > 0:
                hash_vals[i] = list_res[0][1]
            else:
                hash_vals[i] = 0

        true_values = t.from_numpy( true_values ).to(dev)
        hash_vals = t.from_numpy( hash_vals ).to(dev)

        mask = true_values.ne(0)
        true_values[mask], hash_vals[mask]
        true_values.ne(0).sum(), hash_vals.ne(0).sum()

        return hash_vals

    def query_many(self, queries):

        # values = t.empty_like(queries[:,0]).to( self.mem.dtype )
        #
        # for i,vec in enumerate(queries):
        #     values[i] = self.mem[tuple(vec)]


        ## sparse lib
        dev = queries.device
        queries = queries.cpu().numpy()
        values = np.empty_like(queries[:,0]).astype( queries.dtype )

        for i,vec in enumerate(queries):
            values[i] = self.mem[tuple(vec)]
        values = t.from_numpy( values ).to(dev)

        return values































