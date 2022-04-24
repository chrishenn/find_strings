import atexit, math, os, time, PIL
import matplotlib.pyplot as plt

import numpy as np
from numba import njit, prange

import torch as t
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as transforms

from datasets.artificial_dataset import make_topbox, make_topbox_plus, make_nine, make_vbar, make_blob, make_tetris, make_topbar, make_circle_seed
from string_finder import image_utils, object_draw
from unionfind.unionfind import UnionFind

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_opt_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/frnn_bipart_tree_brute/build/libfrnn_ts.so")




def mmm(data):
    return data.min(),data.max(),data.mean()



class String_Finder_Tree(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        thresh_lo, thresh_hi = 0.1, 0.1
        self.canny = Canny(opt, thresh_lo, thresh_hi)


    def forward(self, batch):
        dev = batch.device
        t.cuda.set_device(dev)

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
        oodl_draw.oodl_draw(0, draw_obj=False, strs=strs, max_size=self.opt.img_size)
        ################################



        ###### MERGING TREE

        ### PSEUDOCODE #############
        ## original strings become root nodes in search tree. These unmerged strings represent the choice to 'not merge', and thus should remain in the tree for subsequent scoring.

        ## for enough steps to reach an arbitrary tree depth:
            ## for each neighboring (qualified) pair of strings, generate a possible combined string.
            ## "Similarity Score": compute the similarity score for strings that could possibly merge; store in tree-node for possible new string

            ## a string cannot merge with any parent.
            ## strings that share a parent cannot merge.

            ## increase the search size for neighboring strings proportional to their size

        ## "Underlying Data Scores": generate scores for each node on the tree

        ## extract best-scored strings from pruned tree.
        ### ################

        depth = 3

        ## each row gives a str's inheritance. A '1' in a row gives a parent, whose id is the column-id in which the '1' appears
        # tree_table = t.zeros([strs.size(0), strs.size(0) * 6], dtype=t.uint8, device='cpu')

        tree_table = UnionFind()


        for i in range(depth):

            all_ends = t.cat([strs[:,[0,1]], strs[:,[2,3]]])
            pair_ids = t.arange(strs.size(0), device=strs.device).repeat(2)
            ## TODO: support batching
            imgid = t.zeros_like(all_ends[:,0]).long()


            ## find edges from all string-ends
            end_edges = t.ops.my_ops.frnn_ts_kernel(all_ends, imgid, t.tensor( 0.5 * (i+1) ).cuda(), t.tensor(1).cuda(), batch_size)[0]

            ## TODO: this end_Edges indexes into all_ends, which can be twice as large as pair_ids. we want str_ids here
            pairids_lf, pairids_rt = pair_ids[end_edges[:,0]], pair_ids[end_edges[:,1]]

            # filter ends that define the same string
            same_str = pairids_lf.eq( pairids_rt )

            # filter ends that share an ancestor-string
            tree_lf, tree_rt = tree_table[pairids_lf.cpu()], tree_table[pairids_rt.cpu()]
            share_ancester = (tree_lf & tree_rt).max(dim=1).values.bool()

            # filter ends that inherit from each other
            inherit_lfrt = tree_lf.gather(1, pairids_rt[:,None].cpu()).squeeze().bool()
            inherit_rtlf = tree_rt.gather(1, pairids_lf[:,None].cpu()).squeeze().bool()

            end_edges = end_edges[ (same_str | share_ancester.cuda() | inherit_lfrt.cuda() | inherit_rtlf.cuda()).logical_not() ]


            ###### generate combined strings from valid end_edges
            ## draw a "straight-line" between the farther endpoints (far_ends_lf/far_ends_rt) of the neighboring strings
            ## string deviation direction is referenced to string normal. normal points toward center of curvature.
            ## new string deviation is generated by avg perp dists from straight-line seg to other possible endpoints (not current deviations)
            ## new string normal is avg direction of the two contributing strings

            close_ends_lf, close_ends_rt = all_ends[end_edges[:,0]], all_ends[end_edges[:,1]]

            pairids_lf, pairids_rt = pair_ids[end_edges[:, 0]], pair_ids[end_edges[:, 1]]
            tree_lf, tree_rt = tree_table[pairids_lf.cpu()], tree_table[pairids_rt.cpu()]

            other_endpt_cols_lf = t.where( end_edges[:,0].ge(strs.size(0))[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
            other_endpt_cols_rt = t.where( end_edges[:,1].ge(strs.size(0))[:,None], t.tensor([[2,3]],device=dev).repeat(end_edges.size(0), 1), t.tensor([[0,1]],device=dev).repeat(end_edges.size(0), 1) )
            far_ends_lf = strs[pairids_lf].gather(1, other_endpt_cols_rt)
            far_ends_rt = strs[pairids_rt].gather(1, other_endpt_cols_lf)

            ## TEST ################################
            ## filter distance = 0
            # not_coincident = F.pairwise_distance(far_ends_lf, far_ends_rt).gt(1e-3)
            # other_ends_lf, other_ends_rt = other_ends_lf[not_coincident], other_ends_rt[not_coincident]

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

            ## filter tree_lf and tree_rt
            tree_lf, tree_rt = tree_lf[not_coinc.cpu()], tree_rt[not_coinc.cpu()]

            ## filter distances and endpoints
            seg_lens = seg_lens[not_coinc]
            close_ends_lf, close_ends_rt = close_ends_lf[not_coinc], close_ends_rt[not_coinc]
            far_ends_lf, far_ends_rt = far_ends_lf[not_coinc], far_ends_rt[not_coinc]

            ## filter pair_ids; needed for inheritance
            pairids_lf, pairids_rt = pairids_lf[not_coinc], pairids_rt[not_coinc]

            ## filtered string normals
            norm_cols = t.tensor([[4, 5]], device=dev).repeat(pairids_lf.size(0), 1)
            seg_norms_lf, seg_norms_rt = strs[pairids_lf].gather(1, norm_cols), strs[pairids_rt].gather(1, norm_cols)


            #### generate new possible strings
            ## deviation_pt describes the location of deviation from straight-line connection between string far-ends
            deviation_pt = close_ends_lf.add(close_ends_rt).div(2)

            ## new deviation frac of merged string will be the right-angle deviation of deviation_pt to straight-line connection, relative to straight-line seg length
            test_pts = deviation_pt
            dev_dist = ( (far_ends_rt[:,1] - far_ends_lf[:,1]) * (far_ends_lf[:,0] - test_pts[:,0]) )  -  ( (far_ends_lf[:,1] - test_pts[:,1]) * (far_ends_rt[:,0] - far_ends_lf[:,0]) )
            dev_dist.div_(seg_lens)

            new_dev_fracs = dev_dist.div(seg_lens)

            ## new string normals
            new_str_norms = seg_norms_lf.add(seg_norms_rt).div(2)
            new_str_norms.div_( new_str_norms.norm(dim=1, keepdim=True) )

            ## pack new strs datastrct
            new_strs = t.cat([ far_ends_lf, far_ends_rt, new_str_norms, new_dev_fracs[:,None] ], 1)
            strs = t.cat([strs, new_strs])

            ## TEST ################################
            oodl_draw.oodl_draw(0, draw_obj=False, strs=new_strs, max_size=self.opt.img_size, dot_locs=t.cat([far_ends_lf,far_ends_rt]))
            ################################


            #### update tree_table; each new string inherits ancestors from both parents; just elemwise add their rows from tree_table
            tree_table_append = tree_lf.add(tree_rt)
            assert max( pairids_lf.max(), pairids_rt.max()) < tree_table_append.size(1)

            pairids_lf.unsqueeze_(1), pairids_rt.unsqueeze_(1)
            tree_table_append.scatter_(1, pairids_lf.cpu(), t.ones([pairids_lf.size(0), 1], dtype=t.uint8))
            tree_table_append.scatter_(1, pairids_rt.cpu(), t.ones([pairids_rt.size(0), 1], dtype=t.uint8))

            tree_table = t.cat([tree_table, tree_table_append])
            assert tree_table.size(0) < tree_table.size(1)

            print("no")




        ## TODO: after tree is built, generate similarity scores and filter out bad strings


        return None























