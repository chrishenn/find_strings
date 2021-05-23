import torch as t



class Debug_Dataset(t.utils.data.Dataset):
    def __init__(self, opt, wrapped_dataset):
        self.batch_size =   opt.batch_size
        self.img_size =     opt.img_size
        self.dataset = wrapped_dataset
        self.classes = wrapped_dataset.classes

        self.calls = 0
        self.ids = [18803, 28702, 11915, 27372]

    def __getitem__(self, key):

        token = self.calls % 4
        key = self.ids[token]

        self.calls += 1

        return self.dataset.__getitem__(key)

    def __len__(self):
        return len(self.dataset)