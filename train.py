import re, time, os
import matplotlib.pyplot as plt
import PIL

import torch as t
import torch.utils.data
import torch.nn as nn

import torchvision
from torchvision import transforms

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.cuda.amp as amp
import torchnet as tnt

from datasets.debug_dataset import Debug_Dataset
from datasets.oneim_dataset import OneIm_Dataset
from datasets.resizing_dataset import Resizing_Dataset

import string_finder.string_finder as string_finder


def scale_aware_collator(data):

    batch =   t.cat([te[0] for te in data], 0)
    sizes =  t.stack([te[1] for te in data], 0)

    labels = t.stack([te[-1] for te in data])

    return ((batch, sizes), labels)

def get_loader(opt, train):
    data_dir = opt.data_dir
    if not os.path.exists(data_dir): os.mkdir(data_dir)

    ## init
    if train:
        datamode_istrain = opt.train_datamode=="train"
        rand_rotate = opt.train_rotate
        datasize = opt.train_size
        shuffle = opt.train_shuffle
        scale = opt.train_scale
        normalize = opt.train_normalize
    else:
        datamode_istrain = opt.test_datamode=="train"
        rand_rotate = opt.test_rotate
        datasize = opt.test_size
        shuffle = opt.test_shuffle
        scale = opt.test_scale
        normalize = opt.test_normalize

    ## build transforms
    opt.base_dataset = opt.base_dataset.lower()
    if opt.base_dataset in ['cifar', 'tiny_imagenet']: opt.c_init = 3
    else: opt.c_init = 1

    tran_list = list()
    if opt.img_resize is not None:
        tran_list.append(transforms.Resize([opt.img_resize, opt.img_resize], interpolation=PIL.Image.NEAREST))
    if rand_rotate:
        tran_list.append(transforms.RandomRotation(180, expand=False, resample=PIL.Image.BICUBIC))

    tran_list.append(transforms.ToTensor())
    if normalize:
        norms = tuple(0.5 for _ in range(opt.c_init))
        normal_tran = transforms.Normalize(norms, norms)
        tran_list.append(normal_tran)
    transform = transforms.Compose(tran_list)

    ## Build Dataset
    if opt.base_dataset == 'cifar':
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=datamode_istrain, download=True, transform=transform)
    elif opt.base_dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(root=data_dir, train=datamode_istrain, download=True, transform=transform)
    elif opt.base_dataset == 'tiny-imagenet':
        if train: dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200/train'), transform)
        else: dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200/test'), transform)
    else:
        data_folder = os.path.join(data_dir, opt.base_dataset)
        if os.path.exists(data_folder):
            dataset = torchvision.datasets.ImageFolder(data_folder, transform)
        else:
            print('looked for folder', data_folder,' to load ImageFolder dataset; folder does not exist'); exit(1)

    if opt.oneim_dataset:
        dataset = OneIm_Dataset(opt, dataset)

    #################################################################
    opt.n_classes = len(dataset.classes)

    if 'world_size' in opt and opt.world_size > 1:
        partitions = dict()
        for i in range(opt.world_size):
            partitions[str(i)] = len(dataset)//opt.world_size
        dataset = tnt.dataset.SplitDataset(dataset, partitions)
        dataset.select(str(dist.get_rank()))

    elif datasize:  # split the dataset to at least 1/100th of its full size
        partitions = dict()
        multiple = len(dataset) // datasize
        n_parts = multiple if multiple < 100 else 100
        size = len(dataset) // n_parts
        for i in range(n_parts):
            partitions[str(i)] = size
        classes = dataset.classes
        dataset = tnt.dataset.SplitDataset(dataset, partitions)
        dataset.select('0')
        dataset.classes = classes

    collate_fn = None
    if scale:
        dataset = Resizing_Dataset(opt, scale, dataset)
        collate_fn = scale_aware_collator
    # if opt.debug:
    #     dataset = Debug_Dataset(opt, dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.n_threads, pin_memory=False, collate_fn=collate_fn, drop_last=True)
    return loader



def train(model, opt):

    total_batches = 0
    loader = opt.train_loader
    t.set_grad_enabled(False)
    loader_len = len(loader)

    for epoch in range( opt.n_epochs ):
        epoch_time = time.time()

        for i, data in enumerate(loader, 0):
            total_batches += 1

            inputs, _ = data
            inputs = inputs.to(opt.dev_ids[0])

            with amp.autocast(enabled=opt.amp):
                model(inputs)

            if i % opt.print_freq == 0 or (i+1) == len(loader):
                print('BATCH: ', i, '/', loader_len)

                print(model.count_avg / total_batches)
                print(model.n_ccpts / total_batches)
                print(model.maxes / total_batches)
                print(model.devs / total_batches)
                print()

        print('\n EPOCH: ', epoch)

        if opt.save_freq is not None and (epoch+1)%opt.save_freq == 0:
            opt.epoch, opt.total_batches = epoch, total_batches
            save_model(model, opt)

        print("epoch time: ", time.time() - epoch_time, "\n\n")



def save_model(model, opt):
    model.cpu()

    state_dict = model.state_dict()

    if 'sampler' in state_dict: del state_dict['sampler']
    if 'sampler.dists' in state_dict: del state_dict['sampler.dists']
    if 'sampler.img_filter' in state_dict: del state_dict['sampler.img_filter']
    if 'sampler.pts' in state_dict: del state_dict['sampler.pts']

    if 'canny' in state_dict: del state_dict['canny']

    filename = 'epoch_{}_batch_{}.pth'.format(opt.epoch, opt.total_batches)
    save_path = os.path.join(opt.save_dir, filename)
    print('saving model at ', save_path)

    t.save(state_dict, save_path)
    print("model save done")

    model.to(opt.dev_ids[0])



def init_environment(opt):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if opt.debug: os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:         os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    if opt.debug: opt.n_threads = 1

    t.cuda.set_device(opt.dev_ids[0])

    # torch.multiprocessing.set_start_method('forkserver', force=True)
    # torch.multiprocessing.set_start_method('spawn', force=True)

    # torch.cuda.manual_seed_all(7)
    # torch.manual_seed(7)

    # torch.set_num_threads(2)



##### Training Launchers
def train_single(opt):

    init_environment(opt)

    opt.train_loader = get_loader(opt, train=True)
    opt.test_loader = get_loader(opt, train=False)

    # model = string_finder.String_Finder(opt)
    # model = string_finder.Pyramid_Strings(opt)
    # model = string_finder.Seg_Strings(opt)
    model = string_finder.Interp_String(opt)
    print(model)
    model = model.to(opt.dev_ids[0])

    if opt.load_from is not None:
        print("loading from ", opt.load_from)

        model.load_state_dict( t.load(opt.load_from), strict=False )

        print("load done")

    tic = time.time()
    train(model, opt)
    print('ran ', opt.n_epochs, ' epochs in ', time.time() - tic)

##### Entry Point: Luanch Training Loop for Configuration
def train_oonet(opt):
    if opt.amp: print("AMP training enabled")

    print("single-device training on device: ", opt.dev_ids)
    train_single(opt)
    exit(0)



