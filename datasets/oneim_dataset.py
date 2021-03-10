import torch as t

from oomodel import oodl_utils




class OneIm_Dataset(t.utils.data.Dataset):
    '''
    Dataset that returns a batch of copies of a single image, where each example in the batch is transformed by the transform
    passed to constructor as argument "transform". For no transform, pass transform=None. Single image taken from CIFAR10.
    '''
    def __init__(self, opt, wrapped_dataset):

        self.wrapped_dataset = wrapped_dataset
        self.classes = wrapped_dataset.classes

        # self.key = 18803  ## bird in cifar10
        self.key = 17        ## from mnist | 0: '5', 2: '4', 3: '1', 4: '9', 7: '3', 17: '8'


    def __getitem__(self, _):

        img, _ = self.wrapped_dataset.__getitem__(self.key)
        return (img, t.tensor(0))

    def __len__(self):
        return len(self.wrapped_dataset)


