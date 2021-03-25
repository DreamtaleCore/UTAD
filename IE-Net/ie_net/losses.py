import torch
import torch.nn as nn
import torch.nn.functional as F
from ie_net.metrics import SSIM


class SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super().__init__()
        self.ssim = SSIM(window_size, size_average)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


class MultiScaleEdgeLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def compute_gradient_loss(self, img1, img2, level=1):
        gradx_loss = []
        grady_loss = []
        
        for l in range(level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            gradx_loss.append(torch.mean(torch.abs(gradx1 - gradx2)))
            grady_loss.append(torch.mean(torch.abs(grady1 - grady2)))

            img1 = F.interpolate(img1, scale_factor=0.5, mode='nearest')
            img2 = F.interpolate(img2, scale_factor=0.5, mode='nearest')

        gradx_loss = sum(gradx_loss) / len(gradx_loss)
        grady_loss = sum(grady_loss) / len(grady_loss)

        return gradx_loss, grady_loss

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

    def forward(self, img1, img2, level=3):
        gx, gy = self.compute_gradient_loss(img1, img2, level)
        return (gx + gy) / 2


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]