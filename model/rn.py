import torch
import torch.nn as nn
import torch.nn.functional as F
from util.visual_map import visualize_feature_map


class RN_binarylabel(nn.Module):
    def __init__(self, feature_channels,norm_layer):
        super(RN_binarylabel, self).__init__()
    def forward(self, x, label):
        '''
        input:  x: (B,C,M,N), features
                label: (B,1,M,N), 1 for foreground regions, 0 for background regions
        output: _x: (B,C,M,N)
        '''
        label = label.detach()

        rn_foreground_region = self.rn(x * label, label)

        rn_background_region = self.rn(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()

        sum = torch.sum(region, dim=[2,3])  # (B, C) -> (C)
        Sr = torch.sum(mask, dim=[2,3])    # (B, 1) -> (1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)  # (B, C) -> (C)
        std =  torch.sqrt(torch.sum((region - mu[:,:,None,None]).pow(2), dim=[2,3]) / Sr + 1e-5 )
        return (region-mu[:,:,None,None])/std[:,:,None,None]*mask

class RN_B(nn.Module):
    def __init__(self, feature_channels,norm_layer=None):
        super(RN_B, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # RN
        self.rn = RN_binarylabel(feature_channels,norm_layer)    # need no external parameters

        # gamma and beta
        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)

    def forward(self, x, mask,index):
        # mask = F.adaptive_max_pool2d(mask, output_size=x.size()[2:])
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   # after down-sampling, there can be all-zero mask

        rn_x = self.rn(x, mask)

        rn_x_foreground = (rn_x * mask) * (1 + self.foreground_gamma[None,:,None,None]) + self.foreground_beta[None,:,None,None]
        rn_x_background = (rn_x * (1 - mask)) * (1 + self.background_gamma[None,:,None,None]) + self.background_beta[None,:,None,None]
        out = rn_x_foreground + rn_x_background

        # visualize_feature_map(x.chunk(2)[0],"feature_before0_"+str(index))
        # visualize_feature_map(out.chunk(2)[0], "feature_after0_" + str(index))
        # visualize_feature_map(x.chunk(2)[1],"feature_before1_"+str(index))
        # visualize_feature_map(out.chunk(2)[1], "feature_after1_" + str(index))
        return out

class SelfAware_Affine(nn.Module):
    def __init__(self, kernel_size=7):
        super(SelfAware_Affine, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)
        self.beta_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv1(x)
        importance_map = self.sigmoid(x)

        gamma = self.gamma_conv(importance_map)
        beta = self.beta_conv(importance_map)

        return importance_map, gamma, beta

class RN_L(nn.Module):
    def __init__(self, feature_channels,norm_layer=None,threshold=0.8):
        super(RN_L, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
        return: tensor RN_L(x): (B,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # SelfAware_Affine
        self.sa = SelfAware_Affine()
        self.threshold = threshold

        # RN
        self.rn = RN_binarylabel(feature_channels,norm_layer=norm_layer)    # need no external parameters


    def forward(self, x,mask_index=0):

        sa_map, gamma, beta = self.sa(x)     # (B,1,M,N)

        # m = sa_map.detach()
        if x.is_cuda:
            mask = torch.zeros_like(sa_map).cuda()
        else:
            mask = torch.zeros_like(sa_map)
        mask[sa_map.detach() >= self.threshold] = 1

        rn_x = self.rn(x, mask.expand(x.size()))

        rn_x = rn_x * (1 + gamma) + beta
        # temp = torch.mean(mask.chunk(2)[0])
        # temp2 = torch.std(mask.chunk(2)[0])
        #
        # visualize_feature_map(mask.chunk(2)[0],"mask"+str(mask_index))

        return rn_x

class SelfAware_Affine2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SelfAware_Affine2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        #
        # self.gamma_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)
        # self.beta_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv1(x)
        importance_map = self.sigmoid(x)

        # gamma = self.gamma_conv(importance_map)
        # beta = self.beta_conv(importance_map)

        return importance_map#, gamma, beta

class RN_L2(nn.Module):
    def __init__(self, feature_channels,norm_layer=None,threshold=0.8):
        super(RN_L2, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
        return: tensor RN_L(x): (B,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # SelfAware_Affine
        self.sa = SelfAware_Affine2()
        self.threshold = threshold

        # RN
        self.rn = RN_binarylabel(feature_channels,norm_layer=norm_layer)    # need no external parameters


    def forward(self, x,mean_style, std_style):

        sa_map = self.sa(x)     # (B,1,M,N)

        # m = sa_map.detach()
        if x.is_cuda:
            mask = torch.zeros_like(sa_map).cuda()
        else:
            mask = torch.zeros_like(sa_map)
        mask[sa_map.detach() >= self.threshold] = 1

        rn_x = self.rn(x, mask.expand(x.size()))
        B, C, H, W = rn_x.shape
        rn_x =  rn_x.view(B, C, -1)

        out = rn_x*std_style+mean_style# * (1 + gamma) + beta
        out = out.view(B, C, H, W)
        visualize_feature_map(mask,"mask")

        return out
