
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn
# import torch.optim as optim
# from models.DeblurNet import DeblurNet
# from models.DeblurNet import DeblurNet_ori
# from models.VGG19 import VGG19
# from config import config
import numpy as np
# from submodules import *
from VGG19 import VGG19


def angular_error(ilu_est, ilu_gt):             # angular error
    ilu_est_norm = torch.norm(ilu_est, 2, 1)
    ilu_gt_norm = torch.norm(ilu_gt, 2, 1)

    ilu_dot = torch.bmm(ilu_est.view(ilu_est.size(0), 1, 3), ilu_gt.view(ilu_gt.size(0), 3, 1))
    ilu_dot = ilu_dot.view(ilu_dot.size(0))

    ang_value = torch.clamp(ilu_dot/((ilu_est_norm*ilu_gt_norm) + epsilon), min=-1, max=1)
    ang_error = torch.acos(ang_value) * 180 / numpy.pi
    return ang_error

class color_angular_error(nn.Module):
    def __init__(self, color_weight):
        super(color_angular_error, self).__init__()
        self.color_weight = color_weight

    def forward(self, y, y_est):
        #print('y_nor size', y_nor.size())

        y_nor = y.view(y.size()[0], y.size()[1], -1)
        y_est_nor = y_est.view(y_est.size()[0], y_est.size()[1], -1)

        # inner_product = (y_nor * y_est_nor).sum(dim=1)
        # a_norm = y_nor.pow(2).sum(dim=1).pow(0.5)
        # b_norm = y_est_nor.pow(2).sum(dim=1).pow(0.5)
        # cos = inner_product / (a_norm * b_norm + 1e-12)
        # ang_value = torch.clamp(cos, min=-1, max=1)
        # ang_error = 1.0 - ang_value.mean()
        # ang_error = torch.acos(ang_value)
        # ang_error = ang_error.mean()

        norm_y = torch.sqrt(torch.pow(y_nor[:,0,:],2) + torch.pow(y_nor[:,1,:],2) + torch.pow(y_nor[:,2,:],2))
        norm_y_est = torch.sqrt(torch.pow(y_est_nor[:,0,:],2) + torch.pow(y_est_nor[:,1,:],2) + torch.pow(y_est_nor[:,2,:],2))

        nume = y_nor * y_est_nor
        #print('nume size', nume.size())
        nume = nume[:,0,:] + nume[:,1,:] + nume[:,2,:]
        #print('nume size', nume.size())

        cos = nume/(norm_y+1e-12)/(norm_y_est+1e-12)
        ang_value = torch.clamp(cos, min=-1, max=1)
        ang_error = 1.0 - ang_value.mean()
        #print('cos', cos)

        #loss = torch.acos(cos.sum())
        # loss = 1.0-cos.mean()
        #print('cos loss', loss)
        # ang_error = torch.acos(ang_value) * 180 / np.pi
        if not torch.isfinite(ang_error):
            # print(ang_error.mean())
            # print(y)
            # print(y_est)
            print(torch.max(y_est_nor), torch.min(y_est_nor))
            # print(torch.max(norm_y), torch.min(norm_y))
            # print(torch.max(norm_y_est), torch.min(norm_y_est))
            # print(torch.max(nume), torch.min(nume))
            print(torch.max(cos), torch.min(cos))
            print(7)
        # ang_error = torch.mean(ang_error)

        return ang_error* self.color_weight

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

'''def perceptualLoss(fakeIm, realIm, vggnet):

    #use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer


    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss'''

class perceptualLoss(nn.Module):

    def __init__(self):
        super(perceptualLoss, self).__init__()

    def forward(self, fakeIm, realIm, vggnet):
        weights = [1, 0.2, 0.04]
        features_fake = vggnet(fakeIm)
        features_real = vggnet(realIm)
        features_real_no_grad = [f_real.detach() for f_real in features_real]
        mse_loss = nn.MSELoss(reduction='mean')
        loss = 0
        for i in range(len(features_real)):
            loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
            loss = loss + loss_i * weights[i]

        return loss

class aveBrLoss(nn.Module):

    def __init__(self, aveBr_weight):
        super(aveBrLoss, self).__init__()
        self.aveBr_weight = aveBr_weight

    def forward(self, y, y_est):

        bs = y.size()[0]
        y_mean = y.view(bs, -1).mean()
        #print('y.view', y.view(bs, -1))
        #print('y_mean', y_mean)
        y_est_mean = y_est.view(bs, -1).mean()
        loss  = torch.pow((y_mean-y_est_mean), 2).sum()
        return loss * self.aveBr_weight

class stdLoss(nn.Module):

    def __init__(self, std_weight):
        super(stdLoss, self).__init__()
        self.std_weight = std_weight

    def forward(self, y, y_est):

        bs = y.size()[0]
        y_std = y.view(bs, -1).std()
        y_est_std = y_est.view(bs, -1).std()
        loss = torch.pow((y_std - y_est_std), 2).sum()
        return loss * self.std_weight

class histLoss(nn.Module):

    def __init__(self, hist_weight):
        super(histLoss, self).__init__()
        self.hist_weight = hist_weight

    def forward(self, y, y_est):

        #y_hist = torch.histc(y, bins=256, min=0, max=1)
        y_hist = torch.histc(y.view(y.size()[0], -1))
        #y_est_hist = torch.histc(y_est, bins=256, min=0, max=1)
        y_est_hist = torch.histc(y_est.view(y_est.size()[0], -1))
        loss  = torch.pow((y_hist-y_est_hist), 2).sum()
        return loss * self.hist_weight

class TVLoss(nn.Module):

    def __init__(self, tv_weight):
        super(TVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x):

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]),2).sum()
        return self.tv_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class colorLoss(nn.Module):

    def __init__(self, color_weight):
        super(colorLoss, self).__init__()
        self.color_weight = color_weight

    def forward(self, y, y_est):

        y_sum = y[:,0,:,:]+ y[:,1,:,:]+ y[:,2,:,:]
        y_sum = torch.unsqueeze(y_sum, dim=1)
        y_nor = torch.div(y, y_sum+1e-12)
        #print('y_nor size', y_nor.size())
        y_nor = y_nor.view(y_nor.size()[0], y_nor.size()[1], -1)
        #print('y_nor size', y_nor.size())
        y_nor = torch.mean(y_nor, dim=2)
        #print('y_nor', y_nor)
        '''y_nor = y_nor.permute(0,2,1)
        print('y_nor size', y_nor.size())'''

        y_est_sum = y_est[:,0,:,:]+ y_est[:,1,:,:]+ y_est[:,2,:,:]
        y_est_sum = torch.unsqueeze(y_est_sum, dim=1)
        y_est_nor = torch.div(y_est, y_est_sum+1e-12)
        y_est_nor = y_est_nor.view(y_est_nor.size()[0], y_est_nor.size()[1], -1)
        y_est_nor = torch.mean(y_est_nor, dim=2)
        #print('y_est_nor', y_est_nor)
        '''y_est_nor = y_est_nor.permute(0,2,1)
        print('y_est_nor size', y_est_nor.size())'''

        norm_y = torch.sqrt(torch.pow(y_nor[:,0],2) + torch.pow(y_nor[:,1],2) + torch.pow(y_nor[:,2],2))
        norm_y_est = torch.sqrt(torch.pow(y_est_nor[:,0],2) + torch.pow(y_est_nor[:,1],2) + torch.pow(y_est_nor[:,2],2))
        #norm_y = torch.norm(y_nor, p=1, dim = 1)
        #norm_y_est = torch.norm(y_est_nor, p=1, dim = 1)
        #print('norm_y size', norm_y.size())

        nume = y_nor * y_est_nor
        #print('nume size', nume.size())
        nume = nume[:,0] + nume[:,1] + nume[:,2]
        #print('nume size', nume.size())

        cos = nume/(norm_y+1e-12)/(norm_y_est+1e-12)
        #print('cos', cos)

        #loss = torch.acos(cos.sum())
        loss = 1.0-cos.mean()
        #print('cos loss', loss)

        return loss * self.color_weight

class IOU_calc(nn.Module):

    def __init__(self):
        super(IOU_calc, self).__init__()

    def forward(self, OUT, GT):

        # Union = torch.maximum(OUT, GT)
        Union = OUT + GT
        Union = Union > 0
        Union = Union.float()
        Intersect = OUT * GT
        Intersect = Intersect > 0
        Intersect = Intersect.float()
        IOU = (torch.sum(Intersect)).item() / (torch.sum(Union)).item()

        return IOU