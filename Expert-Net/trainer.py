"""
Network for Expert Network (Expert-Net)
Codes partically based on https://github.com/NVlabs/MUNIT
Created on 10/13/2020
@author DreamTale
"""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from networks import ExpertNet, MsImageDis
from utils import weights_init, get_model_list, get_scheduler, MultiScaleEdgeLoss

class ExpertNet_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(ExpertNet_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_i = ExpertNet(hyperparameters['input_dim_i'],  hyperparameters['gen'])   # auto-encoder for domain a
        self.gen_m = ExpertNet(hyperparameters['input_dim_m'],  hyperparameters['gen'])   # auto-encoder for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.detail_dim = hyperparameters['gen']['detail_dim']

        # fix the noise used in sampling detail
        display_size = int(hyperparameters['display_size'])
        self.d_i = torch.randn(display_size, self.detail_dim, 1, 1).cuda()
        self.d_m = torch.randn(display_size, self.detail_dim, 1, 1).cuda()

        
    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_in, x_im):
        self.eval()
        d_m = Variable(self.d_m)
        c_i, d_i_fake = self.gen_i.encode(x_in)
        c_m, _        = self.gen_m.encode(x_im)
        x_i2m = self.gen_m.decode(c_m, d_m)
        x_m2i = self.gen_i.decode(c_i, d_i_fake)
        # self.train()
        return x_i2m, x_m2i

    def sample(self, x_i, x_m):
        self.eval()
        d_m = Variable(self.d_m)
        x_i_recon, x_m_recon, x_m2i, x_i2m = [], [], [], []
        for i in range(x_i.size(0)):
            c_i, d_i_fake = self.gen_i.encode(x_i[i].unsqueeze(0))
            c_m, d_m_fake = self.gen_m.encode(x_m[i].unsqueeze(0))
            x_i_recon.append(self.gen_i.decode(c_i, d_i_fake))
            x_m_recon.append(self.gen_m.decode(c_m, d_m_fake))
            x_m2i.append(self.gen_i.decode(c_m, d_i_fake))
            x_i2m.append(self.gen_m.decode(c_i, d_m[i].unsqueeze(0)))
        x_i_recon, x_m_recon = torch.cat(x_i_recon), torch.cat(x_m_recon)
        x_m2i = torch.cat(x_m2i)
        x_i2m = torch.cat(x_i2m)
        self.train()
        return x_i, x_i_recon, x_i2m, x_m, x_m_recon, x_m2i

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_i.load_state_dict(state_dict['i'])
        self.gen_m.load_state_dict(state_dict['m'])
        iterations = int(last_model_name[-11:-3])
        
        print('Resume from iteration %d' % iterations)
        return iterations
