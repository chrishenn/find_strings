import torch as t
import torch.nn as nn
import torch.nn.functional as F


## Adapted from oodl_local (or oodl_local_clone)
from string_finder import oodl_draw


class OONet_local(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.c_out = c_out = 512
        layer_1 = OOLayer(opt, 16, opt.c_init, 512, ptwise=True, bn=True, relu=True)
        # group_1 = OOPool(opt)
        # layer_2 = OOLayer(opt, 2, 32, 64, ptwise=True, bn=True, relu=True)
        # group_2 = OOPool(opt)
        # layer_3 = OOLayer(opt, 4, 64, 128, ptwise=True, bn=True, relu=True)
        # group_3 = OOPool(opt)
        # layer_4 = OOLayer(opt, 8, 128, 256, ptwise=True, bn=True, relu=True)
        #
        # layer_5 = OOLayer(opt, 16, 256, c_out, ptwise=True, bn=True, relu=True)
        # layer_6 = OOLayer(opt, 32, c_out, c_out, ptwise=True, bn=True, relu=True)

        self.layers = nn.Sequential()
        self.layers.add_module("layer_1", layer_1)
        # self.layers.add_module("group_1", group_1)
        #
        # self.layers.add_module("layer_2", layer_2)
        # self.layers.add_module("group_2", group_2)
        #
        # self.layers.add_module("layer_3", layer_3)
        # self.layers.add_module("group_3", group_3)
        #
        # self.layers.add_module("layer_4", layer_4)
        #
        # self.layers.add_module("layer_5", layer_5)
        # self.layers.add_module("layer_6", layer_6)

        bn = nn.BatchNorm1d(c_out, track_running_stats=False)
        fc = nn.Linear(c_out, opt.n_classes)
        relu = nn.ReLU()
        self.bn_fc_relu = nn.Sequential(bn,fc,relu)

    def avg_pool(self, data):
        tex, imgid, batch_size = data[0], data[1], data[-1]
        out = t.zeros([batch_size, tex.size(1)], dtype=tex.dtype, device=tex.device)
        out = out.index_add(0, imgid, tex)

        counts = t.zeros(batch_size,dtype=tex.dtype,device=tex.device).index_add_(0, imgid, t.ones_like(imgid).float())
        counts = t.where(counts.lt(1),t.ones_like(counts), counts)
        out = out.div(counts[:,None])
        return out

    def sum_pool(self, data):
        tex, imgid, batch_size = data[0], data[1], data[-1]
        out = t.zeros([batch_size, tex.size(1)], dtype=tex.dtype, device=tex.device)
        out = out.index_add(0, imgid, tex)
        return out

    def forward(self, data):

        tex, pts, imgid, batch_size = data
        textures = None,None,None,tex
        pts_geom = pts, imgid, None, batch_size
        leaf_geom = None,None,None,None
        data = textures, pts_geom, leaf_geom

        if self.opt.debug:
            for i,layer in enumerate(self.layers):
                data = layer(data)

                if i%2==0:
                    textures, pts_geom, leaf_geom = data
                    orig_tex, pred_tex, target_tex, class_tex = textures
                    pts, imgid, edges, batch_size = pts_geom

                    oodl_draw.oodl_draw(0, pts, imgid=imgid, edges=edges)

        else: data = self.layers(data)

        textures, pts_geom, leaf_geom = data
        orig_tex, pred_tex, target_tex, class_tex = textures
        pts, imgid, edges, batch_size = pts_geom

        data = self.avg_pool((class_tex,imgid,batch_size))
        # data = self.sum_pool((class_tex,imgid,batch_size))
        data = self.bn_fc_relu(data)
        return data





def find_geom(pts_geom, lin_radius, scale_radius):
    pts, imgid, edges, batch_size = pts_geom

    edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, lin_radius, scale_radius, batch_size)[0]
    edges = t.cat([edges, t.stack([edges[:,1],edges[:,0]], 1)])
    edges = t.cat([edges, t.arange(pts.size(0),device=edges.device)[:,None].repeat(1,2)])

    ## relative angles
    src_angles, dst_angles = pts[:,3][edges[:, 0]], pts[:,3][edges[:, 1]]
    diff_ang = dst_angles - src_angles
    diff_ang_leaf = t.stack([diff_ang.sin(), diff_ang.cos()], 1)

    ## relative distances
    locs_src, locs_dst = pts[:, :2][edges[:, 0]], pts[:, :2][edges[:, 1]]
    diff_yx = (locs_dst - locs_src)

    lin_ratio = pts[:, 4][edges[:, 1]].mul(pts[:, 4][edges[:, 0]])
    lin_ratio = t.where(lin_ratio.lt(1), t.ones_like(lin_ratio), lin_ratio)
    diff_mag = F.pairwise_distance(locs_dst, locs_src).div(lin_radius).div(lin_ratio)
    diff_mag_leaf = t.stack([diff_mag.sin(), diff_mag.cos()], 1).sub(0.5).mul(2)

    ## rotated relative vec-angles
    diff_rot = t.atan2(diff_yx[:,0], diff_yx[:,1] +1e-5)
    diff_rot = t.where(edges[:,0].eq(edges[:,1]), src_angles, diff_rot)  ## remove image-ref'd angles due to self-edges
    diff_rot = diff_rot - src_angles
    diff_rot_leaf = t.stack([diff_rot.sin(), diff_rot.cos()], 1)

    pts_geom = pts, imgid, edges, batch_size
    leaf_geom = diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, None
    return pts_geom, leaf_geom

class OOLayer(nn.Module):
    def __init__(self, opt, radius_factor, chan_in, chan_out, ptwise=True, bn=True, relu=True):
        super().__init__()
        self.chan_in, self.chan_out = chan_in, chan_out

        if radius_factor is not None:
            lin_radius = radius_factor + 0.05
            self.register_buffer('lin_radius',   t.tensor(lin_radius, dtype=t.float))
            self.register_buffer('scale_radius', t.tensor(1.0, dtype=t.float))
        if bn: self.bn = nn.BatchNorm1d(chan_out, track_running_stats=False)
        if relu: self.relu = nn.ReLU()

        std = 0.1
        self.register_parameter('center_kern', nn.Parameter(t.empty([1, chan_out])) )
        nn.init.normal_(self.center_kern.data, mean=0, std=std)
        self.register_parameter('kernel_bias', nn.Parameter(t.zeros([1, chan_out])))

        dtex_in = chan_in
        if ptwise:
            self.pt_wise = nn.Linear(chan_in,chan_out)
            nn.init.normal_(self.pt_wise.weight.data, mean=0, std=std)
            dtex_in = chan_out

        self.diff_tex = nn.Linear(dtex_in,chan_out)
        self.diff_mag = nn.Linear(2,chan_out)
        self.diff_ang = nn.Linear(2,chan_out)
        self.diff_rot = nn.Linear(2,chan_out)

        self.register_parameter('scale_1', nn.Parameter(t.ones(1,dtex_in)))
        self.register_parameter('scale_2', nn.Parameter(t.ones(1,2)))
        self.register_parameter('scale_3', nn.Parameter(t.ones(1,2)))
        self.register_parameter('scale_4', nn.Parameter(t.ones(1,2)))
        self.register_parameter('bias_1', nn.Parameter(t.zeros(1,dtex_in)))
        self.register_parameter('bias_2', nn.Parameter(t.zeros(1,2)))
        self.register_parameter('bias_3', nn.Parameter(t.zeros(1,2)))
        self.register_parameter('bias_4', nn.Parameter(t.zeros(1,2)))

        self.diff_1 = nn.Linear(chan_out, chan_out)
        self.diff_2 = nn.Linear(chan_out, chan_out)
        self.diff_3 = nn.Linear(chan_out, chan_out)
        self.diff_4 = nn.Linear(chan_out, chan_out)
        nn.init.normal_(self.diff_1.weight.data, mean=0, std=std)
        nn.init.normal_(self.diff_2.weight.data, mean=0, std=std)
        nn.init.normal_(self.diff_3.weight.data, mean=0, std=std)
        nn.init.normal_(self.diff_4.weight.data, mean=0, std=std)

    def forward(self, data):

        textures, pts_geom, leaf_geom = data

        if hasattr(self, 'lin_radius'):
            pts_geom, leaf_geom = find_geom(pts_geom, self.lin_radius, self.scale_radius)

        orig_tex, pred_tex, target_tex, class_tex = textures
        pts, imgid, edges, batch_size = pts_geom
        diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, _ = leaf_geom

        if hasattr(self, 'pt_wise'):
            class_tex = self.pt_wise(class_tex)

        ## src textures
        src_tex, dst_tex = class_tex[edges[:, 0]], class_tex[edges[:, 1]]
        diff_tex = dst_tex.sub(src_tex).mul(self.scale_1).add(self.bias_1)
        diff_tex = self.diff_tex(diff_tex.abs())

        ## relative angles
        diff_ang = diff_ang_leaf.mul(self.scale_2).add(self.bias_2)
        diff_ang = self.diff_ang(diff_ang)

        ## relative distances
        diff_mag = diff_mag_leaf.mul(self.scale_3).add(self.bias_3)
        diff_mag = self.diff_mag(diff_mag)

        ## rotated relative vec-angles
        diff_rot = diff_rot_leaf.mul(self.scale_4).add(self.bias_4)
        diff_rot = self.diff_rot(diff_rot)

        ## chain integration
        descr = self.diff_1(diff_ang)
        descr = self.diff_2(descr * diff_mag)
        descr = self.diff_3(descr * diff_rot)
        descr = self.diff_4(descr * diff_tex)
        diff_act = descr * src_tex

        ## center of kernel
        center_act = class_tex * self.center_kern

        ## pool
        diff_act = diff_act.to(center_act.dtype)
        tex_active = center_act.index_add(0, edges[:,1], diff_act)
        class_tex = tex_active + self.kernel_bias

        ## av_pool
        # if self.chan_in < 256:
        #     counts = t.ones_like(class_tex[:,0]).index_add_(0, edges[:,1], t.ones_like(edges[:,1]).float())
        #     class_tex = class_tex.div(counts[:,None])

        # counts = t.ones_like(class_tex[:, 0]).index_add_(0, edges[:, 1], t.ones_like(edges[:, 1]).float())
        # class_tex = class_tex.div(counts[:, None])

        ## bn + relu
        if hasattr(self, 'bn'): class_tex = self.bn(class_tex)
        if hasattr(self, 'relu'): class_tex = self.relu(class_tex)

        ## repack
        textures = orig_tex, pred_tex, target_tex, class_tex
        leaf_geom = diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, None
        return textures, pts_geom, leaf_geom

class OOPool(nn.Module):
    def __init__(self, opt, radius_factor=None, frac=1):
        super().__init__()
        self.frac = frac

        if radius_factor is not None:
            lin_radius = radius_factor + 0.05

            self.register_buffer('lin_radius', t.tensor(lin_radius, dtype=t.float))
            self.register_buffer('scale_radius', t.tensor(1.0, dtype=t.float))

    def forward(self, data):
        textures, pts_geom, leaf_geom = data
        orig_tex, pred_tex, target_tex, class_tex = textures
        pts, imgid, edges, batch_size = pts_geom

        ## filter out self-edges. Edges remaining are bi-directional
        if hasattr(self, 'lin_radius'):
            edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, self.lin_radius, self.scale_radius, batch_size)[0]
            edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])
        else:
            edges = edges[edges[:, 0].ne(edges[:, 1])]

        o_act = class_tex.norm(dim=1).clone().detach()

        ## avg rt->lf; rebalance by dividing sent textures by the # of times they've sent
        neb_avg = t.zeros_like(class_tex[:, 0]).index_add_(0, edges[:, 0], o_act[edges[:, 1]])
        counts = t.zeros_like(class_tex[:, 0]).index_add_(0, edges[:, 0], t.ones_like(edges[:, 0]).float())
        counts = t.where(counts.lt(1), t.ones_like(counts), counts)
        neb_avg.div_(counts)

        ## I survive if my act is greater than the avg of my neb's acts
        live_mask = o_act > (neb_avg * self.frac)

        ## edges with dying objects at left and survivors at right
        send_mask = (live_mask.logical_not()[edges[:, 0]]) & (live_mask[edges[:, 1]])
        send_edges = edges[send_mask]

        ## pool dying-object texture into their neighbors that survive
        class_tex = class_tex.index_add(0, send_edges[:, 1], class_tex[send_edges[:, 0]])
        class_tex = class_tex.div(counts[:, None])

        class_tex, pts, imgid = class_tex[live_mask], pts[live_mask], imgid[live_mask]

        return (orig_tex, pred_tex, target_tex, class_tex), (pts, imgid, edges, batch_size), leaf_geom