import os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import warnings
warnings.filterwarnings("ignore")

class Options_Super():

    def parse(self):

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()

        if 'SINGULARITY_NAME' in os.environ:
            print('running in singularity container: ', os.environ['SINGULARITY_NAME'])

        opt = self.process_paths(opt)

        self.print_options(opt)

        str_ids = opt.dev_ids.split(',')
        opt.dev_ids = []

        if len(str_ids) == 1 and (str_ids[0] =='cpu' or str_ids[0] == '-1'):
            opt.dev_ids.append( 'cpu' )
        else:
            for str_id in str_ids:
                    id = int(str_id)
                    if id >= 0:
                        opt.dev_ids.append(id)

        self.opt = opt
        return self.opt

    def print_options(self, opt):

        message = ''
        message += ' ----------------- Options ----------------- \n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += ' ----------------- ------- ----------------- '
        print(message)

    def process_paths(self, opt):

        super_dir = opt.save_folder
        if not os.path.exists(super_dir): os.mkdir(super_dir)

        exp_name = re.sub('[/.]+', '_', opt.exp_name)
        save_dir = os.path.join(super_dir, exp_name)
        if not os.path.exists(save_dir): os.mkdir(save_dir)

        opt.save_dir = save_dir

        return opt