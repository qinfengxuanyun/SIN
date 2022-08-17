import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .external_function import SpectralNorm
from .rn import *
import numpy as np
import cv2
######################################################################################
# base function for network structure
######################################################################################


def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # elif classname.find('BatchNorm2d') != -1:
        #     init.normal_(m.weight.data, 1.0, 0.02)
        #     init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'instance2':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'lrn':
        norm_layer = functools.partial(nn.LocalResponseNorm,k=0.00005, alpha=1, beta=0.5)
        #norm_layer = nn.LocalResponseNorm(k=0.00005, alpha=1, beta=0.5, size=5)
    elif norm_type == 'lr':
        norm_layer = functools.partial(LayerNorm,affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params/1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

# class adaIN(nn.Module):
#     """
#     spectral normalization
#     code and idea originally from Takeru Miyato's work 'Spectral Normalization for GAN'
#     https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
#     """
#     def __init__(self, eps=1e-5):
#         super(adaIN, self).__init__()
#         self.eps=eps
#     def forward(self,feature, mean_style, std_style):
#         B, C, H, W = feature.shape
#
#         feature = feature.view(B, C, -1)
#
#         std_feat = (torch.std(feature, dim=2) + self.eps).view(B, C, 1)
#         mean_feat = torch.mean(feature, dim=2).view(B, C, 1)
#
#         adain = std_style * (feature - mean_feat) / std_feat + mean_style
#
#         adain = adain.view(B, C, H, W)
#         return adain

def adaIN(feature, mean_style, std_style, eps=1e-5):
    B, C, H, W = feature.shape

    feature = feature.view(B, C, -1)

    std_feat = (torch.std(feature, dim=2) + eps).view(B, C, 1)
    mean_feat = torch.mean(feature, dim=2).view(B, C, 1)

    adain = std_style * (feature - mean_feat) / std_feat + mean_style

    adain = adain.view(B, C, H, W)
    return adain


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            #output_nc = output_nc * 4
            #self.pool = nn.PixelShuffle(upscale_factor=2)
            self.pool = nn.Upsample(size=output_nc, scale_factor=2, mode='bilinear')
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass,)

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out

class ResBlock_Gate(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False):
        super(ResBlock_Gate, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            #output_nc = output_nc * 4
            #self.pool = nn.PixelShuffle(upscale_factor=2)
            self.pool = nn.Upsample(size=output_nc, scale_factor=2, mode='bilinear')
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.conv3 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv4 = coord_conv(hidden_nc, 1, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
        self.output_nc = output_nc
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
            self.model2 = nn.Sequential(nonlinearity, self.conv3, nonlinearity, self.conv4, nn.Sigmoid(),)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)
            self.model2 = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv3, norm_layer(hidden_nc), nonlinearity,
                                       self.conv4, nn.Sigmoid(),)

        self.shortcut = nn.Sequential(self.bypass,)

    def forward(self, x,mask=None):
        if self.sample:
            # out1,out2= self.model(x).split(self.output_nc,dim=1)
            # mask = F.sigmoid(out2)
            out = self.model(x)+self.shortcut(x)
            out = self.pool(out)

            mask_gate = self.model2(x)
            mask = mask + (1.0-mask)*mask_gate
            #out = out1 * mask
        else:
            # out1,out2= self.model(x).split(self.output_nc,dim=1)
            # mask = F.sigmoid(out2)
            out = self.model(x)+self.shortcut(x)

            mask_gate = self.model2(x)
            mask = mask + (1.0 - mask) * mask_gate
            #out = out1 * mask


        return out,mask

class ResBlock_Ada(nn.Module):
    def __init__(self, input_nc, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(ResBlock_Ada, self).__init__()

        # using no ReLU method

        # general
        self.nonlinearity = nonlinearity

        # left
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs_short)
        self.p = nn.Parameter(torch.rand(input_nc*4, 512).normal_(0.0, 0.02))
    def forward(self, x, style):
        p = self.p.unsqueeze(0)
        p = p.expand(style.shape[0],p.shape[1],p.shape[2])
        psi_slice = torch.bmm(p, style)
        C = psi_slice.shape[1]
        res = x

        out = adaIN(x, psi_slice[:, 0:C // 4, :], psi_slice[:, C // 4:C // 2, :])
        out = self.nonlinearity(out)
        out = self.conv1(out)
        out = adaIN(out, psi_slice[:, C // 2:3 * C // 4, :], psi_slice[:, 3 * C // 4:C, :])
        out = self.nonlinearity(out)
        out = self.conv2(out)

        out = out + self.bypass(res)

        return out

class ResBlock_Ada2(nn.Module):
    def __init__(self, input_nc, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(ResBlock_Ada2, self).__init__()

        # using no ReLU method

        # general
        self.nonlinearity = nonlinearity

        # left
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 2,'dilation':2}
        kwargs2 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs2)
        self.bypass = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs_short)
        self.p = nn.Parameter(torch.rand(input_nc*4, 512).normal_(0.0, 0.02))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, style):
        p = self.p.unsqueeze(0)
        p = p.expand(style.shape[0],p.shape[1],p.shape[2])
        psi_slice = torch.bmm(p, style)
        C = psi_slice.shape[1]
        res = self.bypass(x)

        out = adaIN(x, psi_slice[:, 0:C // 4, :], psi_slice[:, C // 4:C // 2, :])
        out = self.nonlinearity(out)
        out = self.conv1(out)
        out = adaIN(out, psi_slice[:, C // 2:3 * C // 4, :], psi_slice[:, 3 * C // 4:C, :])
        out = self.nonlinearity(out)
        out = self.conv2(out)

        # feature, mask = out.chunk(2, dim=1)
        # out = feature * self.sigmoid(mask) + res

        out = out + res

        return out#,self.sigmoid(mask)

class ResBlock_Ada3(nn.Module):
    def __init__(self, input_nc, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False,dilation=2):
        super(ResBlock_Ada3, self).__init__()

        # using no ReLU method

        # general
        self.nonlinearity = nonlinearity

        # left
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': dilation,'dilation':dilation}
        kwargs2 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs2)
        self.bypass = coord_conv(input_nc, input_nc, use_spect, use_coord, **kwargs_short)
        # self.p = nn.Parameter(torch.rand(input_nc*4, 512).normal_(0.0, 0.02))
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x, style):
        # p = self.p.unsqueeze(0)
        # p = p.expand(style.shape[0],p.shape[1],p.shape[2])
        # psi_slice = torch.bmm(p, style)
        # C = psi_slice.shape[1]
        res = self.bypass(x)

        #out = adaIN(x, psi_slice[:, 0:C // 4, :], psi_slice[:, C // 4:C // 2, :])
        out = self.nonlinearity(x)
        out = self.conv1(out)
        #out = adaIN(out, psi_slice[:, C // 2:3 * C // 4, :], psi_slice[:, 3 * C // 4:C, :])
        out = self.nonlinearity(out)
        out = self.conv2(out)

        # feature, mask = out.chunk(2, dim=1)
        # out = feature * self.sigmoid(mask) + res

        out = out + res

        return out#,self.sigmoid(mask)

class ResBlock_Ada_Up(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=None,norm_layer=None, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(ResBlock_Ada_Up, self).__init__()

        # using no ReLU method
        self.input_nc = input_nc
        self.hidden_nc = hidden_nc

        # general
        self.nonlinearity = nonlinearity

        # left
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs2 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs2)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
        self.p = nn.Parameter(torch.rand(input_nc*2+hidden_nc*2, 512).normal_(0.0, 0.02))
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.Upsample(size=None, scale_factor=2, mode='bilinear')
    def forward(self, x, style):
        p = self.p.unsqueeze(0)
        p = p.expand(style.shape[0],p.shape[1],p.shape[2])
        psi_slice = torch.bmm(p, style)
        #C = psi_slice.shape[1]
        res = self.bypass(self.pool(x))

        out = adaIN(x, psi_slice[:, 0:self.input_nc, :], psi_slice[:, self.input_nc:self.input_nc*2, :])
        out = self.nonlinearity(out)
        out = self.pool(out)
        out = self.conv1(out)
        out = adaIN(out, psi_slice[:,self.input_nc*2:self.input_nc*2+self.hidden_nc, :], psi_slice[:, self.input_nc*2+self.hidden_nc:self.input_nc*2+self.hidden_nc*2, :])
        out = self.nonlinearity(out)
        out = self.conv2(out)

        # feature, mask = out.chunk(2, dim=1)
        # out = feature * self.sigmoid(mask) + res

        out = out+res

        return out#,self.sigmoid(mask)


class ResBlockEncoderOptimized(nn.Module):
    """
    Define an Encoder block for the first layer of the discriminator and representation network
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False):
        super(ResBlockEncoderOptimized, self).__init__()

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
        self.nonlinearity = nonlinearity
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(self.conv1, nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.model = nn.Sequential(self.conv1, norm_layer(output_nc), nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))

        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), self.bypass)

    def forward(self, x):
        out = self.model(x) +  self.shortcut(x)
        return out

class ResBlockDecoder(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc

        self.pool = nn.UpsamplingNearest2d(scale_factor=2)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity,self.pool, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity,self.pool, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass,)

    def forward(self, x):
        out = self.model(x) + self.shortcut(self.pool(x))
        return out

'''
class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        self.bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out
'''

class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)
        #out = (1.0+out)/2

        return out


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention

class Auto_Attn2(nn.Module):
    def __init__(self, in_channel,norm_layer=None):
        super(Auto_Attn2, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 4, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 4, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        #self.conv_i = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.model = ResBlock(int(in_channel * 2), in_channel, in_channel, norm_layer=norm_layer, use_spect=True)

    def forward(self,x, pre=None, mask=None):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(f_projection.view(B, -1, H * W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H * W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H * W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1
        #4 256 1024 4 1024 1024
        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            p_projection = self.conv_h(pre).view(B, -1, W * H)
            context_flow = torch.bmm(p_projection, attention_map).view(B, C, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        # elif type(mask) != type(None):
        #     out = self.gamma * out + x*mask
        # else:
        #     out = self.gamma * out + x
        return out, attention_map

class Auto_Attn3(nn.Module):
    def __init__(self, in_channel, norm_layer=None):
        super(Auto_Attn3, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 4, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 4, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        self.conv_i = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.model = ResBlock(int(in_channel * 2), in_channel, in_channel, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x).view(B, -1, H * W)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x).view(B, -1, H * W)  # BxC'xHxW, C'=C//8
        x_projection = self.conv_h(x).view(B, -1, H * W)  # BxC'xHxW

        proj_key = g_projection
        proj_query = torch.transpose(f_projection, 1, 2)  # BxNxC', N=H*W
          # B X C x (N)

        attention_map = torch.bmm(proj_query, proj_key)  # transpose check
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        out = torch.bmm(x_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)
        out = self.alpha * out + x

        # using long distance attention layer to copy information from valid regions
        p_projection = self.conv_i(pre).view(B, -1, W * H)
        attn_pre = torch.bmm(p_projection, attention_map).view(B, C, W, H)
        attn_pre = self.gamma * (1 - mask) * attn_pre + (mask) * pre
        out = self.model(torch.cat([out, attn_pre], dim=1))
        return out, attention_map

class Auto_Attn4(nn.Module):
    def __init__(self, in_channel, norm_layer=None):
        super(Auto_Attn4, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 4, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 4, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        self.conv_i = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))


        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.model = ResBlock_Gate(int(in_channel * 2), in_channel, in_channel, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x).view(B, -1, H * W)  # BxC'xHxW, C'=C//8
        x_projection = self.conv_h(x).view(B, -1, H * W)  # BxC'xHxW

        proj_key = f_projection
        proj_query = torch.transpose(f_projection, 1, 2)  # BxNxC', N=H*W
        # B X C x (N)

        attention_map = torch.bmm(proj_query, proj_key)  # transpose check
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        out = torch.bmm(x_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)
        out = self.alpha * out + x

        # using long distance attention layer to copy information from valid regions
        p_projection = self.conv_i(pre).view(B, -1, W * H)
        attn_pre = torch.bmm(p_projection, attention_map).view(B, C, W, H)
        attn_pre = self.gamma * (1 - mask) * attn_pre + (mask) * pre
        #out = self.model(torch.cat([out, attn_pre], dim=1))

        attn_fuse,attn_mask = self.model(torch.cat([out, attn_pre], dim=1),mask)
        out = attn_mask * attn_fuse + (1.0-attn_mask)*x
        return out, attention_map, attn_mask

def extract_patches(x, kernel=3, stride=1):
  #x = nn.ZeroPad2d(kernel//4)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches

class AtnConv(nn.Module):
    def __init__(self, input_channels=128, output_channels=64, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10.,
                 fuse=True, rates=[1, 2, 4, 8],nonlinearity=nn.LeakyReLU(0.1),norm_layer=None):
        super(AtnConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.groups = groups
        self.fuse = fuse
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.fuse:
            for i in range(groups):
                self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
                    nonlinearity,
                    nn.Conv2d(input_channels, output_channels // groups, kernel_size=3, dilation=rates[i],
                              padding=rates[i])
                    )
                                 )
        else:
            self.__setattr__('conv_1*1', nn.Sequential(
                nonlinearity,
                nn.Conv2d(input_channels, output_channels, kernel_size=1,padding=0)
                )
                             )
        self.model = ResBlock(int(input_channels * 2), input_channels, input_channels, norm_layer=norm_layer, use_spect=True)
    def forward(self, x1, x2,x, mask=None,score=None):
        """ Attention Transfer Network (ATN) is first proposed in
            Learning Pyramid Context-Encoder Networks for High-Quality Image Inpainting. Yanhong Zeng et al. In CVPR 2019.
          inspired by
            Generative Image Inpainting with Contextual Attention, Yu et al. In CVPR 2018.
        Args:
            x1: low-level feature maps with larger resolution.
            x2: high-level feature maps with smaller resolution.
            mask: Input mask, 1 indicates holes.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.
        Returns:
            torch.Tensor, reconstructed feature map.
        """
        # get shapes
        x1s = list(x1.size())
        x2s = list(x2.size())
        mask = F.interpolate(mask, size=x1s[2:4], mode='bilinear', align_corners=True)
        if self.ksize == 1:
            kernel = 1
            rate = 1
        else:
            rate = x1s[-1]//x2s[-1]
            kernel = rate#*2
        # extract patches from low-level feature maps x1 with stride and rate
        raw_w = extract_patches(x1, kernel=kernel, stride=self.stride*rate)
        raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel)  # B*HW*C*K*K
        # split tensors by batch dimension; tuple is returned
        #4 1024 32 32 4 1024 128 4 4
        #raw_w =  raw_w.transpose([0,2,3,1])
        B,N, C, R_H, R_W = raw_w.shape
        raw_w = raw_w.permute(0,2,3,4,1)
        # score = torch.eye(1024, 1024).unsqueeze(0).float().cuda(0)
        # score = torch.cat([score,score,score,score],dim=0)
        y = torch.bmm(raw_w.view(B,C*R_H*R_W,N),score)

        y = y.view(B,C,R_H,R_W,32,32).permute(0, 1, 4, 2, 5, 3)
        y = y.reshape(B,C,32*R_H,32*R_W).contiguous()
        #temp = torch.mean(y - x1)


        # raw_w_groups = torch.split(raw_w, 1, dim=0)
        # y = []
        # for yi, raw_wi in zip(score,raw_w_groups):
        #     # attending
        #     wi_center = raw_wi[0]
        #     yi = F.conv_transpose2d(yi, wi_center, stride=rate, padding=rate // 2)
        #     if self.ksize!=1:
        #         yi = yi / 4.0#(2.0*rate)
        #     y.append(yi)
        # y = torch.cat(y, dim=0)
        # y.contiguous().view(x1s)

        y = x1*mask+self.alpha*y*(1.0-mask)
        # adjust after filling
        if self.fuse:
            tmp = []
            for i in range(self.groups):
                tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(y))
            y = torch.cat(tmp, dim=1)
        else:
            y = self.__getattr__('conv_1*1')(y)

        out = self.model(torch.cat([x, y], dim=1))
        attn_mask = 0
        # attn_fuse,attn_mask = self.model(torch.cat([x, y], dim=1),mask)
        # out = attn_mask * attn_fuse + (1.0-attn_mask)*x

        return out,attn_mask

class AtnScore(nn.Module):
    def __init__(self,ksize=3, stride=1, softmax_scale=10.):
        super(AtnScore, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

    def forward(self, x2, mask=None):
        x2s = list(x2.size())

        # split high-level feature maps x2 for matching
        f_groups = torch.split(x2, 1, dim=0)
        # extract patches from x2 as weights of filter
        w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
        w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)  # B*HW*C*K*K
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is not None:
            mask = F.interpolate(mask, size=x2s[2:4], mode='bilinear', align_corners=True)
        else:
            mask = torch.zeros([1, 1, x2s[2], x2s[3]])
            if torch.cuda.is_available():
                mask = mask.cuda()
        # extract patches from masks to mask out hole-patches for matching
        m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
        m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
        m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
        mm = m.eq(0.).float()  # (B, HW, 1, 1)
        mm_groups = torch.split(mm, 1, dim=0)

        y = []
        scale = self.softmax_scale
        padding = 0 if self.ksize == 1 else 1
        for xi, wi, mi in zip(f_groups, w_groups, mm_groups):
            # matching based on cosine-similarity
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4])
            if torch.cuda.is_available():
                escape_NaN = escape_NaN.cuda()
            # normalize
            wi_normed = wi / torch.max(torch.sqrt((wi * wi).sum([1, 2, 3], keepdim=True)), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
            yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3])

            # apply softmax to obtain
            yi = yi * mi
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mi
            yi = yi.clamp(min=1e-8)
            y.append(yi)
        #y = torch.cat(y, dim=0)
        return y


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.matmul(input_feature, self.weight)
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
