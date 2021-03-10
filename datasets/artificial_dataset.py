import torch as t
from torchvision import transforms

import numpy as np

from oomodel import oodl_utils


############### Artificial Testing Images

class OneSquare_Dataset(t.utils.data.Dataset):
    def __init__(self, opt, transform):
        self.batch_size =   opt.batch_size
        self.img_size = size =    opt.img_size

        D1 = t.arange(opt.img_size, dtype=t.int)
        D2 = t.arange(opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)

        img = t.zeros([opt.img_size, opt.img_size, 3]).sub_(1)
        img[ ((3*size)//8-1 < gridy) & (gridy <= (5*size)//8-1) & ((3*size)//8-1 < gridx) & (gridx <= (5*size)//8-1) ] = 1
        img = img.permute(2,0,1)
        # oolayers_utils.tensor_imshow(img)

        topil = transforms.ToPILImage()
        self.img = topil( img )

        if transform is not None: self.transform = transform
        else: self.transform = lambda x: x

    def __getitem__(self, key):

        img = self.img.copy()
        img = self.transform(img)

        return (img, 0)

    def __len__(self):
        return 2000

class OneCurve_Dataset(t.utils.data.Dataset):
    def __init__(self, opt):
        self.batch_size =   opt.batch_size
        self.img_size = size =    opt.img_size

        D1 = t.arange(opt.img_size, dtype=t.int)
        D2 = t.arange(opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)

        gridy_p, gridx_p = gridy.sub(size//2), gridx.sub(size//2)
        radii = (gridy_p.pow(2) + gridx_p.pow(2)).float().sqrt()

        img = t.zeros([opt.img_size, opt.img_size, 3]).sub_(1)
        img[ (gridy < size//2) & (radii > 11) & (radii < 13) ] = 1
        img = img.permute(2,0,1)

        self.register_buffer('img', img)

    def __getitem__(self, key):

        img = self.img.clone()

        return (img, t.tensor(0))

    def __len__(self):
        return 2000


class OneCircle_Dataset(t.utils.data.Dataset):
    def __init__(self, opt):
        img = make_circle(opt)
        self.register_buffer('img', img)

    def __getitem__(self, key):
        img = self.img.clone()
        return (img, t.tensor(0))

    def __len__(self):
        return 2000


def make_circle(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)
    radii = (gridy_p.pow(2) + gridx_p.pow(2)).float().sqrt()

    img = t.zeros([opt.img_size, opt.img_size])
    img[(radii > 8) & (radii < 10)] = 1

    return img[None,None]

def make_arc(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub( (opt.img_size//2) + 4), gridx.sub( (opt.img_size//2) )
    radii = (gridy_p.pow(2) + gridx_p.pow(2)).float().sqrt()
    angles = t.atan2(gridy_p, gridx_p).sub(np.pi/8)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(radii > 3) & (radii < 7) & (angles > -np.pi/2) & (angles < np.pi/2)] = 1

    return img[None,None]

def make_topbar(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -10) & (gridy_p < -8)] = 1

    return img[None,None]


def make_vbar(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > -2) & (gridx_p < 2) & (gridy_p > -8) & (gridy_p < 8)] = 1

    return img[None,None]

def make_blob(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > -2) & (gridx_p < 2) & (gridy_p > -8) & (gridy_p < -4)] = 1

    return img[None,None]

def make_rtang(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[ ( (gridx_p > -5) & (gridx_p < 5) & (gridy_p > -10) & (gridy_p < -8) ) | ( (gridx_p > 5) & (gridx_p < 7) & (gridy_p > -10) & (gridy_p < 0) )] = 1

    return img[None,None]

def make_topbox(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -10) & (gridy_p < -8)] = 1
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -2) & (gridy_p < 0)] = 1

    img[(gridx_p > -6) & (gridx_p < -4) & (gridy_p > -10) & (gridy_p < 0)] = 1
    img[(gridx_p > 4) & (gridx_p < 6) & (gridy_p > -10) & (gridy_p < 0)] = 1

    return img[None,None]

def make_topbox_plus(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -10) & (gridy_p < -8)] = 1
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -2) & (gridy_p < 0)] = 1

    img[(gridx_p > -6) & (gridx_p < -4) & (gridy_p > -10) & (gridy_p < 0)] = 1
    img[(gridx_p > 4) & (gridx_p < 6) & (gridy_p > -10) & (gridy_p < 0)] = 1

    img[(gridx_p > -6) & (gridx_p < -4) & (gridy_p > -1) & (gridy_p < 6)] = 1

    return img[None,None]

def make_nine(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -10) & (gridy_p < -8)] = 1
    img[(gridx_p > -5) & (gridx_p < 5) & (gridy_p > -2) & (gridy_p < 0)] = 1

    img[(gridx_p > -6) & (gridx_p < -4) & (gridy_p > -10) & (gridy_p < 0)] = 1
    img[(gridx_p > 4) & (gridx_p < 6) & (gridy_p > -10) & (gridy_p < 0)] = 1

    img[(gridx_p > 4) & (gridx_p < 6) & (gridy_p > -1) & (gridy_p < 6)] = 1

    return img[None,None]

def make_tetris(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    gridy_p, gridx_p = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx_p > 4) & (gridx_p < 6) & (gridy_p > -4) & (gridy_p < 2)] = 1

    img[(gridx_p > 2) & (gridx_p < 6) & (gridy_p > -2) & (gridy_p < 0)] = 1

    return img[None,None]


def make_circle_seed(opt):
    D1 = t.arange(opt.img_size, dtype=t.int)
    D2 = t.arange(opt.img_size, dtype=t.int)
    gridy, gridx = t.meshgrid(D1, D2)

    height = 4

    gridy, gridx = gridy.sub(opt.img_size // 2), gridx.sub(opt.img_size // 2)

    img = t.zeros([opt.img_size, opt.img_size])
    img[(gridx > -height) & (gridx < height) & (gridy > -10) & (gridy < -8)] = 1
    img[(gridx > -height) & (gridx < height) & (gridy > 8) & (gridy < 10)] = 1

    img[(gridx > -10) & (gridx < -8) & (gridy > -height) & (gridy < height)] = 1
    img[(gridx > 8) & (gridx < 10) & (gridy > -height) & (gridy < height)] = 1

    return img[None,None]