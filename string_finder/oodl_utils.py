import matplotlib.pyplot as plt

import torch as t
from torchvision import transforms



def bool_version(tens):
    if t.__version__ < '1.2.0':
        return tens.type(t.ByteTensor).to(tens.device)
    else:
        return tens.bool()

def regrid(texture, pts, imgid, img_size, batch=None, avg=True):
    device = texture.device

    if len(texture.size()) < 2: texture.unsqueeze_(1)

    if batch is None:
        if imgid.size(0) > 1: imgs = imgid.max().add(1)
        else: imgs = 1
        batch = t.zeros( [imgs, texture.size(1), img_size, img_size], dtype=t.float, device=device )

    channels = t.arange(texture.size(1), dtype=t.long, device=device).unsqueeze(0)

    locs_x = pts[:, 1].round().long().unsqueeze(1).clamp(0, img_size-1)
    locs_y = pts[:, 0].round().long().unsqueeze(1).clamp(0, img_size-1)
    batch.index_put_([imgid.unsqueeze(1), channels, locs_y, locs_x], texture, accumulate=True)

    if avg:
        counts = t.ones_like(batch)
        counts.index_put_([imgid.unsqueeze(1), channels, locs_y, locs_x], t.ones_like(locs_x).float(), accumulate=True)
        batch.div_(counts)

    return batch

def tensor_imshow(img, dpi=300, axis='off'):
    img = img.cpu()
    img = img.sub( img.min() )
    img = img.div( img.max() ).clamp(0,1)
    topil = transforms.ToPILImage()

    plt.figure(dpi=dpi)
    plt.axis(axis)
    plt.imshow(topil(img))
    plt.show(block=False)
