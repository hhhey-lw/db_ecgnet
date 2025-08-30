import random

import numpy as np
import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = None
        self._setup_seed()

    def _setup_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def _build_model(self, num_classes):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def train(self):
        pass

    def vali(self, phase, dataloader, criterion):
        pass

    def test(self):
        pass