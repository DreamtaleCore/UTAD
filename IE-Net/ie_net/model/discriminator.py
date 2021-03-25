import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LatentDiscriminaor(nn.Module):
    def __init__(self, latent_dim, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.dim = 128 if latent_dim > 128 else latent_dim
        self.model = nn.Sequential(
            nn.Linear(feat_size**2 * latent_dim*2, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_real, feat_fake):
        """
        feat_comb can be generated as follow:
        ------
        z_map = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.feat_size, self.feat_size)
        idx = list(range(feat.shape[0]))
        random.shuffle(idx)
        feat_shuffle = feat[idx]
        z_f1 = torch.cat([z_map, feat], dim=-1)
        z_f2 = torch.cat([z_map, feat_shuffle], dim=1)
        """

        z_f1_scores = self.model(feat_real.view(feat_real.shape[0], -1))
        z_f2_scores = self.model(feat_fake.view(feat_fake.shape[0], -1))
        loss = -torch.mean(torch.log(z_f1_scores + 1e-6) + torch.log(1 - z_f2_scores + 1e-6))
        return loss
    
    def calc_gen_loss(self, feat_fake):
        z_f2_scores = self.model(feat_fake.view(feat_fake.shape[0], -1))
        loss = -torch.mean(torch.log(z_f2_scores + 1e-6))
        return loss


class ImageDiscriminator(nn.Module):
    def __init__(self, input_dim, gan_type, num_scales):
        super().__init__()
        self.input_dim = input_dim
        self.gan_type = gan_type
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())
    
    def _make_net(self):
        cnn_x = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, (4, 4), stride=2, padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(32,             32, (4, 4), stride=2, padding=(1, 1)), nn.InstanceNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,             32, (4, 4), stride=2, padding=(1, 1)), nn.InstanceNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,             32, (4, 4), stride=2, padding=(1, 1)),
        )
        return cnn_x
    
    def get_output(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs
    
    def forward(self, x_real, x_fake):
        # calculate the loss to train D
        outs0 = self.get_output(x_fake)
        outs1 = self.get_output(x_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
    
    def calc_gen_loss(self, x_fake):
        # calculate the loss to train G
        outs0 = self.get_output(x_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss