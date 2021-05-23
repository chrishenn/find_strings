import os
import time

import numpy as np

import visdom




class OODL_Vis:
    def __init__(self, opt):

        try:
            if opt.vis_file:
                filename = self.get_filename(opt)
            else: filename = None

            if opt.vis_network:
                offline = False
            else: offline = True

            self.vis = visdom.Visdom(offline=offline, log_to_filename=filename, raise_exceptions=True)
            if filename is not None: print("SUCCESS: visdom init write to file")
            if not offline: print("SUCCESS: visdom init connected to network server")

            self.names = ['Train Loss (*50)', 'Test Loss (*50)', 'Train Acc@1 (%)', 'Test Acc@1 (%)']
            self.win = None
            self.env = 'main'
            self.exp_name = opt.exp_name
            self.n_epochs = opt.n_epochs

        except ConnectionError:
            print("ERROR: visdom init failed to connect network server; attempting write to file")

            try:
                filename = self.get_filename(opt)
                self.vis = visdom.Visdom(offline=True, log_to_filename=filename, raise_exceptions=True)
                print("SUCCESS: visdom init; write to file")
            except:
                print("ERROR: visdom open save file failed")
                raise Exception

    def get_filename(self, opt):

        project_dir = os.path.split(os.path.split(__file__)[0])[0]
        graphs_dir = project_dir + '/graphs/'
        if not os.path.exists(graphs_dir): os.mkdir(graphs_dir)

        timestr = time.strftime("%Y.%m.%d-%H.%M.%S")

        filename = graphs_dir + opt.exp_name.replace(' ', '_').replace('/', '_') + '__' + timestr + '.log'
        return filename

    def vis_draw(self, epoch_stats):

        common_X = np.array([epoch_stats[0]]) + 1
        train_loss = np.array([epoch_stats[1]]) * 50
        test_loss = np.array([epoch_stats[3]]) * 50
        train_err = np.array([epoch_stats[2]])
        test_err = np.array([epoch_stats[4]])
        losses = [train_loss, test_loss, train_err, test_err]

        if self.win is None:
            vals = np.array([[train_loss[0], test_loss[0], train_err[0], test_err[0]]])

            self.win = self.vis.line( Y=vals,X=common_X, env=self.env,
                opts=dict(
                    xlabel='Epoch',
                    # xtickstep=1,
                    xtickmin=0,
                    ylabel='Loss and Error',
                    ytickstep=10,
                    title=self.exp_name,
                    legend=self.names,
                    dash=np.array(['dash', 'dash', 'solid', 'solid']),
                    ytickmin=0,ytickmax=120
                ))
        else:
            for i in range(len(losses)):
                self.vis.line(Y=losses[i], X=common_X, win=self.win, update='append', name=self.names[i], env=self.env)





