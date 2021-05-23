import torch as t
import torchvision.transforms as transforms

import PIL



class Resizing_Dataset(t.utils.data.Dataset):
    def __init__(self, opt, scale_limits, wrapped_dataset):

        self.scale_limits = scale_limits
        self.scale_range = scale_limits[1] - scale_limits[0]
        self.dataset = wrapped_dataset
        self.classes = wrapped_dataset.classes
        self.img_size = opt.img_size

        assert scale_limits[0] <= (1 + 1e-6)
        max_pad = opt.img_size - int( opt.img_size * scale_limits[0] ) +1
        self.after = transforms.Compose([
            transforms.Pad(max_pad, fill=-1),
            transforms.CenterCrop(opt.img_size)
        ])

        if opt.debug:
            self.debug = True
            scales = t.linspace(*scale_limits, 4)
            self.sizes = scales.mul(opt.img_size).int()
            self.calls = 0
        else: self.debug = False

    def __getitem__(self, key):

        img, label = self.dataset.__getitem__(key)

        if self.debug:
            token = self.calls
            self.calls += 1
            token %= self.sizes.size(0)
            rand_size = self.sizes[token].item()
        else:
            rand_scale = t.rand(1).mul(self.scale_range).add( self.scale_limits[0] )
            rand_size = (self.img_size * rand_scale).round().int().item()
        if rand_size%2 != 0: rand_size -= 1

        img = transforms.functional.resize(img, rand_size, interpolation=PIL.Image.NEAREST)
        img = self.after(img)

        img.unsqueeze_(0)

        return [img, t.tensor(rand_size), t.tensor(label)]

    def __len__(self):
        return len(self.dataset)













