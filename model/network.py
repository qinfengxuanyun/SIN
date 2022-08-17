from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
from torch.nn import Parameter
import math
from .resnet_face import resnet50_ft_dag
import torchvision.models as models
import torchfile
import imp
import torchvision
from torchvision.models import vgg19
from util.visual_map import visualize_feature_map
from .rn import *
##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='instance', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[]):
    net = ResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)

def define_e2(input_nc=3, ngf=64, z_nc=512, img_f=512, layers=5, norm='instance', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[]):
    net = ResEncoder2(input_nc, ngf, z_nc, img_f, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)

def define_e3(input_nc=3, ngf=64, z_nc=512, img_f=512, layers=5, norm='instance', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[]):
    net = ResEncoder3(input_nc, ngf, z_nc, img_f, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)

def define_g(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU',
             output_scale=1,
             use_spect=False, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):
    net = ResGenerator(output_nc, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord,
                       use_attn)

    return init_net(net, init_type, activation, gpu_ids)


def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
             use_coord=False,
             use_attn=True, model_type='ResDis', init_type='orthogonal', gpu_ids=[]):
    if model_type == 'ResDis':
        net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)
    elif model_type == 'ResDis2':
        net = ResDiscriminator2(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)
    elif model_type == 'PatchDis':
        net = PatchDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)
    return init_net(net, init_type, activation, gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, L=6, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        rb_norm = get_norm_layer(norm_type='instance2')#functools.partial(nn.InstanceNorm2d, momentum=0.1, affine=False)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        #self.rb_start = RN_B(ngf,rb_norm)
        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            #if i<=3:
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect,
                             use_coord)
            setattr(self, 'encoder' + str(i), block)
            # if i>=2:
            #     block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down',
            #                      use_spect,
            #                      use_coord)
            #     setattr(self, 'encoder2' + str(i), block)

            # if i<=1:
            #     block = RN_B(ngf * mult,rb_norm)
            #     setattr(self, 'rn' + str(i), block)

        # attention part
        for i in range(layers - 1):
            mult = min(2 ** (i + 1), img_f // ngf)
            if i == 1:
                block = Auto_Attn2(ngf * mult,norm_layer=None)
                setattr(self, 'attn', block)

        # inference part
        block = ResBlock_Ada(ngf * mult, nonlinearity, use_spect)
        setattr(self, 'infer', block)
        # for i in range(self.L):
        #     #block = ResBlock_Ada3(ngf * mult, nonlinearity, use_spect)
        #     block = ResBlock_Ada2(ngf * mult, nonlinearity, use_spect)
        #     setattr(self, 'infer' + str(i), block)

        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 512*1*1
        self.fc = nn.Sequential(nn.Conv2d(ngf * mult,ngf * mult,4,bias=False))
        self.nonlinearity = nonlinearity
    def forward(self, mask, img_m, landmark1=None,style=None):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """

        mask_use = 1.0-mask
        #input =  torch.cat([img_m,landmark1,mask_use],dim=1)
        input = torch.cat([img_m, mask_use], dim=1)
        #img = torch.cat([img_m, img_a], dim=0)
        # encoder part
        out = self.block0(input)
        feature = [out]

        # mask_index = 0
        # out = self.rb_start(out, mask_use,mask_index)

        for i in range(self.layers - 1):
            # if i==2:
            #     atten_model = getattr(self, 'attn')
            #     out_style, _ = atten_model(out)
            #     model = getattr(self, 'encoder2' + str(i))
            #     out_style = model(out_style)
            # if i>2:
            #     model = getattr(self, 'encoder2' + str(i))
            #     out_style = model(out_style)

            #if i <= 2:
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)


        #out_style = self.sum_pooling(out_style)
        # out_style = self.fc(out_style)
        # out_style = self.nonlinearity(out_style) #out 512*1*1
        # style = out_style.view(-1,self.z_nc,1) #out B*512*1
        #style = torch.ones([1])
        infer_prior = getattr(self, 'infer')
        out = infer_prior(out,style)
        # infer state
        # for i in range(self.L):
        #     infer_prior = getattr(self, 'infer' + str(i))
        #     #visualize_feature_map(out,"before"+str(i))
        #     out = infer_prior(out,style)
        #     #visualize_feature_map(out, "after" + str(i))

        return out, feature,style

class ResEncoder2(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(ResEncoder2, self).__init__()

        self.layers = layers
        self.z_nc = z_nc

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect,
                             use_coord)
            setattr(self, 'encoder' + str(i), block)

        # attention part
        for i in range(layers - 1):
            if i==1:
                mult = min(2 ** (i + 1), img_f // ngf)
                block = Auto_Attn2(ngf * mult,norm_layer=norm_layer)
                setattr(self, 'attn' + str(i), block)

        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 512*1*1
        self.nonlinearity = nonlinearity

    def forward(self, img,landmark):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """
        # encoder part
        #input =  torch.cat([img,landmark],dim=1)
        out = self.block0(img)

        feature = [out]

        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)
            if i == 1:
                atten_model = getattr(self, 'attn' + str(i))
                out, _ = atten_model(out)
        out = self.sum_pooling(out)
        out = self.nonlinearity(out) #out 512*1*1
        style = out.view(-1,self.z_nc,1) #out B*512*1
        return style,feature

class ResEncoder3(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(ResEncoder3, self).__init__()

        self.layers = layers
        self.z_nc = z_nc

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect,
                             use_coord)
            setattr(self, 'encoder' + str(i), block)

        # attention part
        for i in range(layers - 1):
            if i==1:
                mult = min(2 ** (i + 1), img_f // ngf)
                block = Auto_Attn2(ngf * mult,norm_layer=norm_layer)
                setattr(self, 'attn' + str(i), block)

        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 512*1*1
        self.nonlinearity = nonlinearity
        #self.gcn = GcnNet(512)

    def forward(self, img, mask):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """
        # encoder part
        input =  torch.cat([img,mask],dim=1)
        out = self.block0(input)

        feature = [out]

        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)
            if i == 1:
                atten_model = getattr(self, 'attn' + str(i))
                out, _ = atten_model(out)
        out = self.sum_pooling(out)
        out = self.nonlinearity(out) #out 512*1*1
        style = out.view(-1,self.z_nc,1) #out B*512*1
        #style = self.gcn(style)
        return style#,feature


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=512,hidden_dim=1,use_bias=True):
        super(GcnNet, self).__init__()
        self.use_bias = use_bias
        self.gcn1 = GraphConvolution(1, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, 1)
        self.softmax = nn.Softmax(-1)
        self.nonlinearity = nn.LeakyReLU(0.2)
        self.w11 = nn.Parameter(torch.ones(1,input_dim,1))
        self.w12 = nn.Parameter(torch.ones(1,input_dim,1))
        self.w21 = nn.Parameter(torch.ones(1,input_dim,1))
        self.w22 = nn.Parameter(torch.ones(1,input_dim,1))
        self.w31 = nn.Parameter(torch.ones(1,input_dim,1))
        self.w32 = nn.Parameter(torch.ones(1,input_dim,1))
        if self.use_bias:
            self.b11 = nn.Parameter(torch.zeros(1,input_dim,1))
            self.b12 = nn.Parameter(torch.zeros(1,input_dim,1))
            self.b21 = nn.Parameter(torch.zeros(1,input_dim,1))
            self.b22 = nn.Parameter(torch.zeros(1,input_dim,1))
            self.b31 = nn.Parameter(torch.zeros(1,input_dim,1))
            self.b32 = nn.Parameter(torch.zeros(1,input_dim,1))
        else:
            self.register_parameter('b11', None)
            self.register_parameter('b12', None)
            self.register_parameter('b21', None)
            self.register_parameter('b22', None)
            self.register_parameter('b31', None)
            self.register_parameter('b32', None)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, feature):
        if self.use_bias:
            p_projection = feature * self.w11+self.b11#) * self.w12+self.b12
            proj_key = feature * self.w21+self.b21#) * self.w22+self.b22
        else:
            p_projection = feature * self.w11#) * self.w12
            proj_key = feature * self.w21#) * self.w22
        proj_query = torch.transpose(proj_key, 1, 2)  # BxNxC', N=H*W
        adjacency = torch.bmm(proj_key, proj_query)  # transpose check
        adjacency = self.softmax(adjacency)
        h = self.nonlinearity(self.gcn1(adjacency, p_projection))
        h2 = self.gcn2(adjacency, h)
        if self.use_bias:
            output = h2 * self.w31+self.b31#) * self.w32+self.b32
        else:
            output = h2 * self.w31#) * self.w32
        output = self.alpha*output + feature
        return output

        # if self.use_bias:
        #     p_projection = self.nonlinearity(feature * self.w11+self.b11) * self.w12+self.b12
        #     proj_key = self.nonlinearity(feature * self.w21+self.b21) * self.w22+self.b22
        # else:
        #     p_projection = self.nonlinearity(feature * self.w11) * self.w12
        #     proj_key = self.nonlinearity(feature * self.w21) * self.w22
        # proj_query = torch.transpose(proj_key, 1, 2)  # BxNxC', N=H*W
        # adjacency = torch.bmm(proj_key, proj_query)  # transpose check
        # adjacency = self.softmax(adjacency)
        # h = self.nonlinearity(self.gcn1(adjacency, p_projection))
        # h2 = self.nonlinearity(self.gcn2(adjacency, h))
        # if self.use_bias:
        #     output = self.nonlinearity(h2 * self.w31+self.b31) * self.w32+self.b32
        # else:
        #     output = self.nonlinearity(h2 * self.w31) * self.w32
        # return output

class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """

    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # latent z to feature
        mult = min(2 ** (layers - 1), img_f // ngf)
        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 2), img_f // ngf)
            # if i == 0:
            #     mult = 4
            input_n = int(ngf * mult_prev)
            output_n = int(ngf * mult)


            if i > layers - output_scale:
                upconv = ResBlockDecoder(input_n + output_nc, output_n, output_n, norm_layer, nonlinearity,
                                        use_spect, use_coord)
                # upconv = ResBlockDecoder(input_n * 2 + output_nc, output_n, output_n, norm_layer, nonlinearity,
                #                          use_spect, use_coord)
                # if i > layers - output_scale + 1:
                #     upconv = ResBlockDecoder(input_n*2 + output_nc, output_n, output_n, norm_layer, nonlinearity,
                #                              use_spect, use_coord)
                # else:
                #     upconv = ResBlockDecoder(input_n + output_nc, output_n, output_n, norm_layer, nonlinearity,
                #                              use_spect, use_coord)
            else:
                upconv = ResBlockDecoder(input_n, output_n, output_n, norm_layer, nonlinearity, use_spect,
                                         use_coord)

            setattr(self, 'decoder' + str(i), upconv)

            block = ResBlock_Ada(output_n, nonlinearity, use_spect)
            setattr(self, 'infer' + str(i), block)

            # block = AtnScore()
            # setattr(self, 'at_score', block)
            # output part
            if i > self.layers - self.output_scale - 1:
                if i<self.layers-1:
                    block = AtnConv(output_n,output_n)#,fuse=False)
                    setattr(self, 'at_conv' + str(i), block)
                    # block = ResBlock(int(output_n * 2), output_n, output_n, norm_layer=None,
                    #                       use_spect=True)
                    # setattr(self, 'at_fuse' + str(i), block)

                    #outconv = Output(output_n*2, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                    outconv = Output(output_n, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                else:
                    outconv = Output(output_n, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)

            # short+long term attention part
            if i == self.layers - self.output_scale and use_attn:
                attn = Auto_Attn3(output_n, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, mask, f=None,style=None):
        # def forward(self, z, f_m=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """
        results = []
        out = z
        # attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i <= self.layers - self.output_scale:
                model = getattr(self, 'infer' + str(i))
                out = model(out,style)
            if i == self.layers - self.output_scale:
                model = getattr(self, 'attn' + str(i))
                out, attn_score= model(out, f[2],mask[0])
                #out_attn = out
            if i > self.layers - self.output_scale - 1:
                # if i > self.layers - self.output_scale and i<self.layers-1:
                #     model = getattr(self, 'at_conv' + str(i))
                #     out,attn_mask = model(f[4-i],out_attn,out,mask[-1],attn_score)

                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)



        return results

class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn2(ndf * mult_prev, None)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect,
                             use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect,
                               use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x,landmark):
        #out = self.block0(x)
        #out = self.block0(torch.cat([x[-1], landmark], dim=1))
        out = self.block0(x[-1])
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out, out

class ResDiscriminator2(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True):
        super(ResDiscriminator2, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf//2, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn2(ndf * mult_prev, None)
                setattr(self, 'attn' + str(i), attn)
            if i<=1:
                block = ResBlock(ndf * mult_prev, ndf * mult // 2, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect,
                                 use_coord)
                setattr(self, 'encoder' + str(i), block)
            else:
                block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect,
                                 use_coord)
                setattr(self, 'encoder' + str(i), block)

            if i <= 2:
                block = nn.Conv2d(3, ndf * mult_prev // 2, (1, 1), bias=True)
                setattr(self, 'encoder2' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect,
                               use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x,landmark):
        #out = self.block0(torch.cat([x[-1],landmark],dim=1))
        out = self.block0(x[-1])
        for i in range(self.layers - 1):
            if i <= 2:
                model = getattr(self, 'encoder2' + str(i))
                input_part=model(x[-(i+2)])
                out = torch.cat([out,input_part],dim=1)
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)

        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out, out

class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """

    def __init__(self, input_nc=3, ndf=64, img_f=256, layers=6, norm='batch', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=False):
        super(PatchDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 5, 'stride': 2, 'padding': 2}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]
        self.block0 = nn.Sequential(*sequence)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn2(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)

            sequence = [
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]
            setattr(self, 'encoder' + str(i), nn.Sequential(*sequence))
            setattr(self, 'out', nn.Flatten())

        '''
        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)
        '''

    def forward(self, x):
        out = self.block0(x)

        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        model = getattr(self, 'out')
        out = model(out)
        # out = self.model(x)
        return out, out

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path="./checkpoints/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        conv_1_1 = F.relu(self.conv_1_1(x))
        conv_1_2 = F.relu(self.conv_1_2(conv_1_1))
        max_pool_1 = F.max_pool2d(conv_1_2, 2, 2)
        conv_2_1 = F.relu(self.conv_2_1(max_pool_1))
        conv_2_2 = F.relu(self.conv_2_2(conv_2_1))

        features = {'conv_2_2': conv_2_2}

        # max_pool_2 = F.max_pool2d(conv_2_2, 2, 2)
        # conv_3_1 = F.relu(self.conv_3_1(max_pool_2))
        # conv_3_2 = F.relu(self.conv_3_2(conv_3_1))
        # conv_3_3 = F.relu(self.conv_3_3(conv_3_2))
        # max_pool_3 = F.max_pool2d(conv_3_3, 2, 2)
        # conv_4_1 = F.relu(self.conv_4_1(max_pool_3))
        # conv_4_2 = F.relu(self.conv_4_2(conv_4_1))
        # conv_4_3 = F.relu(self.conv_4_3(conv_4_2))
        # max_pool_4 = F.max_pool2d(conv_4_3, 2, 2)
        # conv_5_1 = F.relu(self.conv_5_1(max_pool_4))
        # conv_5_2 = F.relu(self.conv_5_2(conv_5_1))
        # conv_5_3 = F.relu(self.conv_5_3(conv_5_2))
        # max_pool_5 = F.max_pool2d(conv_5_3, 2, 2)

        # features = {
        #     'conv_1_1': conv_1_1,#
        #     'conv_1_2': conv_1_2,
        #     'max_pool_1': max_pool_1,
        #     'conv_2_1': conv_2_1,
        #     'conv_2_2': conv_2_2,
        #     'max_pool_2': max_pool_2,#
        #     'conv_3_1': conv_3_1,
        #     'conv_3_2': conv_3_2,
        #     'conv_3_3': conv_3_3,
        #     'max_pool_3': max_pool_3,
        #     'conv_4_1': conv_4_1,#
        #     'conv_4_2': conv_4_2,
        #     'conv_4_3': conv_4_3,
        #     'max_pool_4': max_pool_4,
        #     'conv_5_1': conv_5_1,
        #     'conv_5_2': conv_5_2,
        #     'conv_5_3': conv_5_3,
        #     'max_pool_5': max_pool_5}
        return features
        # x = max_pool_5.view(max_pool_5.size(0), -1)
        # x = F.relu(self.fc6(x))
        # x = F.dropout(x, 0.5, self.training)
        # x = F.relu(self.fc7(x))
        # x = F.dropout(x, 0.5, self.training)
        # return self.fc8(x),features

class PerceptualLoss2(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss2, self).__init__()
        model = VGG_16()#.double()
        model.load_weights()
        self.add_module('vgg', model)
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['conv_2_2'], y_vgg['conv_2_2'])
        # content_loss += self.weights[1] * self.criterion(x_vgg['max_pool_2'], y_vgg['max_pool_2'])
        # content_loss += self.weights[2] * self.criterion(x_vgg['max_pool_3'], y_vgg['max_pool_3'])
        # content_loss += self.weights[3] * self.criterion(x_vgg['max_pool_4'], y_vgg['max_pool_4'])
        # content_loss += self.weights[4] * self.criterion(x_vgg['max_pool_5'], y_vgg['max_pool_5'])


        return content_loss

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



