import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train

from options.options import Options_Parent


class Options(Options_Parent):

    def initialize(self, parser):

        parser.add_argument('--exp_name', type=str,   default='neb29_hamming', help='')

        parser.add_argument('--debug',    type=bool, default=False, help='debug ')

        # data options
        # parser.add_argument('--base_dataset', type=str, default='cifar', help='chooses what datasets are loaded.')
        parser.add_argument('--base_dataset', type=str, default='mnist', help='chooses what datasets are loaded.')
        # parser.add_argument('--base_dataset', type=str, default='tiny-imagenet', help='chooses what datasets are loaded.')

        parser.add_argument('--oneim_dataset', type=bool, default=True, help='')

        parser.add_argument('--data_dir',     type=str, default='/home/chris/Documents/oodl_local/data/', help='chooses what datasets are loaded.')
        parser.add_argument('--img_size',     type=int, default=32, help='square image; integer size for one side')

        parser.add_argument('--train_datamode', type=str,  default='train', help='train data mode: set to "train" for training-set; set to "test" for test set')
        parser.add_argument('--train_size',     type=int,  default=5000, help='specify dataset size in images - at least 1/30th of the full size. Set "None" for full size.')
        parser.add_argument('--train_rotate',   type=bool, default=False, help='set True to randomly-rotate train images')
        parser.add_argument('--train_scale',    type=float,default=False, help='randomly scale train images in given range')
        parser.add_argument('--train_normalize',type=bool, default=False, help='normalize train images to range=(-1,1)')
        parser.add_argument('--train_shuffle',  type=bool, default=True, help='random-shuffle train images')

        # training options
        parser.add_argument('--n_epochs',  type=int,   default=1, help='number of training epochs')

        parser.add_argument('--batch_size',   type=int, default=40, help='batch size')

        # compute options
        parser.add_argument('--gpu_ids',   type=str,  default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--n_threads', type=int,  default=2, help='data loader cpu threads for each process.')

        parser.add_argument('--amp',            type=bool, default=False, help='turn amp on or off')
        parser.add_argument('--profile',        type=bool, default=False, help='set True to run a profiling epoch')

        # I/O options
        parser.add_argument('--save_freq',   type=int, default=1, help='in epochs')
        parser.add_argument('--save_folder', type=str, default='/home/chris/Documents/deep_mem/checkpoints/', help='')

        parser.add_argument('--ann_filename',   type=str, default='neb29_hamming.ann', help='')
        parser.add_argument('--ann_type',   type=str, default='hamming', help='')
        parser.add_argument('--load_from',   type=str, default='/home/chris/Documents/deep_mem/checkpoints/neb29_hamming/epoch_0_batch_125.pth', help='')
        # parser.add_argument('--load_from',   type=str, default=None, help='')

        parser.add_argument('--print_freq',  type=int,  default=10, help='frequency to print progress (in batches)')
        parser.add_argument('--vis_network', type=bool, default=False, help='set True for loss visualization')
        parser.add_argument('--vis_file',    type=bool, default=False, help='write visualization to file. Run options/gen_graph_from_file.py with a graphs/*.log file.')

        self.parser = parser
        return self.parser

if __name__ == '__main__':

    opt = Options().parse()
    train.train_oonet(opt)
