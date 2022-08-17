import torch
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task
import itertools
import numpy as np
import torch.nn.functional as F
from .vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19

class SIN(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""

    def name(self):
        return "Pluralistic Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        parser.add_argument('--lambda_sigma', type=float, default=5.0, help='weight for generation loss')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two',
                                help='training strategies with two path or three paths')
            parser.add_argument('--use_pc', type=bool, default=True, help='use perceptual loss')
            parser.add_argument('--use_tv', type=bool, default=False, help='use perceptual loss')
            parser.add_argument('--pc_mode', type=str, default='model2', choices=['model1', 'model2'])
            parser.add_argument('--res_layers_p', type=int, nargs='+', default=[5])  # [5],[2,6,9,14,17,20,23,26]
            parser.add_argument('--res_layers_p_weight', type=int, nargs='+', default=[1])  # [1],[1,1,1,1,1,1,1,1]
            parser.add_argument('--use_style', type=bool, default=True, help='use style loss')

            parser.add_argument('--lambda_rec', type=float, default=1.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=1.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_valid', type=float, default=1.0, help='weight for mask  loss')
            parser.add_argument('--lamda_r_scale', type=float, default=0.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_g', type=float, default=0.1, help='weight for generation loss')
            parser.add_argument('--lambda_si', type=float, default=0.5, help='weight for generation loss')

            parser.add_argument('--lambda_cos', type=float, default=1.0, help='weight for cos loss')
            parser.add_argument('--lambda_a_cls', type=float, default=20.0, help='weight for cls loss')
            parser.add_argument('--lambda_o_cls', type=float, default=20.0, help='weight for cls loss')
            parser.add_argument('--lambda_g_cls', type=float, default=20.0, help='weight for cls loss')

            parser.add_argument('--lambda_pc', type=float, default=1.0, help='weight for pc loss')
            parser.add_argument('--lambda_fm', type=float, default=5.0, help='weight for pc loss')
            parser.add_argument('--lambda_style', type=float, default=0, help='weight for style loss')
            parser.add_argument('--lambda_tv', type=float, default=0.0, help='weight for tv loss')
        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)
        if opt.mode == '1':
            self.loss_names = ['app_g', 'ad_g', 'img_g_d','pc_g','pc_g2','style_g']
        elif opt.mode == '2':
            self.loss_names = ['app_g', 'ad_g', 'img_g_d', 'style_s']
        self.model_names = ['E', 'E2', 'G', 'D','E3']

        # define the inpainting model
        self.net_E = network.define_e(input_nc=4,ngf=64, z_nc=512, img_f=512, layers=6, L=6, norm='none', activation='LeakyReLU',
                                      init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_E2 = network.define_e2(input_nc=3,ngf=64, z_nc=512, img_f=512, layers=6, norm='instance', activation='LeakyReLU',
                                      init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G = network.define_g(ngf=64, z_nc=512, img_f=512, L=0, layers=6, output_scale=opt.output_scale,
                                      norm='instance', activation='LeakyReLU', init_type='orthogonal',
                                      gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(input_nc=3,ndf=64, img_f=512, layers=5, model_type='ResDis2', norm='none',
                                      init_type='orthogonal', gpu_ids=opt.gpu_ids)
        
        self.net_E3 = network.define_e3(input_nc=4,ngf=64, z_nc=512, img_f=512, layers=6, norm='instance', activation='LeakyReLU',
                                        init_type='orthogonal', gpu_ids=opt.gpu_ids)


        self.net_G.load_state_dict(torch.load('./checkpoints/celeba_random/14/'+'epoch_88_net_G.pth'))
        self.net_E.load_state_dict(torch.load('./checkpoints/celeba_random/14/'+'epoch_88_net_E.pth'))
        self.net_E2.load_state_dict(torch.load('./checkpoints/celeba_random/14/'+'epoch_88_net_E2.pth'))
        self.net_E3.load_state_dict(torch.load('./checkpoints/celeba_random/23/'+'epoch_98_net_E3.pth'))
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        # if self.opt.isTrain:
        #     self.input = input
        #     self.image_paths = self.input['img_path'] + self.input['img_path2']
        #     self.image_paths2 = self.input['img_path2'] + self.input['img_path']
        #     self.img = torch.cat([input['img'],input['img2']],dim=0)
        #     self.img2 = torch.cat([input['img2'],input['img']],dim=0)
        #     self.landmark1 = torch.cat([input['landmark1'],input['landmark2']],dim=0)
        #     self.landmark2 = torch.cat([input['landmark2'],input['landmark1']],dim=0)
        #     mask = input['mask'].split(1, dim=1)[0]
        #     self.mask = torch.cat([mask,mask],dim=0)
        # else:
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.img2 = input['img']
        # self.landmark1 = input['landmark1']
        # self.landmark2 = input['landmark1']
        self.mask = input['mask'].split(1, dim=1)[0]
        #self.dilate_mask = input['dilate_mask'].split(1, dim=1)[0]
        #self.style_grad = input['style_grad']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])  # ,async=True)
            self.img2 = self.img2.cuda(self.gpu_ids[0])  # , async=True)
            self.mask = self.mask.cuda(self.gpu_ids[0])  # , async=True)
            #self.dilate_mask = self.dilate_mask.cuda(self.gpu_ids[0])  # , async=True)
            self.landmark1 = None#self.landmark1.cuda(self.gpu_ids[0])
            self.landmark2 = None#self.landmark2.cuda(self.gpu_ids[0])
            self.style_grad = None#self.style_grad.cuda(self.gpu_ids[0])
        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_a = self.img2 * 2 - 1
        self.img_m = self.mask * self.img_truth + (1.0-self.mask)
        #self.img_a_m = self.mask * self.img_a
        #self.sd_mask = torch.from_numpy(self.spatial_discounting_mask()).cuda(self.gpu_ids[0])

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_img_a = task.scale_pyramid(self.img_a, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)
        self.scale_mask = [(scaled_img >= 0.5).float() for scaled_img in self.scale_mask] 
        #self.scale_dilate_mask = task.scale_pyramid(self.dilate_mask, self.opt.output_scale)

        self.net_G.load_state_dict(torch.load('./checkpoints/celeba_random/14/' + 'epoch_88_net_G.pth'))
        self.net_E.load_state_dict(torch.load('./checkpoints/celeba_random/14/' + 'epoch_88_net_E.pth'))
        self.net_E2.load_state_dict(torch.load('./checkpoints/celeba_random/14/' + 'epoch_88_net_E2.pth'))
        self.net_E3.load_state_dict(torch.load('./checkpoints/celeba_random/23/' + 'epoch_98_net_E3.pth'))

    def test(self):
        self.save_results(self.img_truth,data_name='truth')
        #self.save_results(self.mask*self.img_truth+(1.0-self.mask)*(self.landmark1*2-1), data_name='ld_g')
        self.save_results(self.mask*self.img_truth, data_name='ld_g')
        if self.opt.mode == '1':
            # encoder process
            style,_ = self.net_E2(self.img_a, self.landmark2)
            distribution, f, style = self.net_E(self.mask, self.img_m, self.landmark1, style)
            results = self.net_G(distribution,self.scale_mask, f,style)
        elif self.opt.mode == '2':
            input = torch.cat([self.img_a,self.img_m],dim=0)
            style_all,f = self.net_E2(input,self.landmark2)
            style,_ = style_all.chunk(2)
            f2,f3 = f[2].chunk(2)
            self.get_uncertainty(f2,f3,style)
            style_adj = self.net_E3(self.img_m, self.mask)
            distributions, f, style_adj = self.net_E(self.mask, self.img_m,self.landmark1,style_adj)
            results = self.net_G(distributions,self.scale_mask, f, style_adj)

        img = results[-1]
        self.img_out = (1 - self.mask) * img.detach() + self.mask * self.img_m
        self.save_results(self.img_out, 0, data_name='out')
        if self.opt.mode == '2':
            alpha = (1 / (1 - self.opt.lambda_sigma*torch.log(self.attention_map_max[3]))).detach()
            #self.save_results(self.mask*self.img_truth+(1.0-self.mask)*(self.landmark1*2*alpha-1), 0, data_name='uncertainty')
            self.save_results(self.mask*self.img_truth+(1.0-self.mask)*(2*alpha-1), 0, data_name='uncertainty')

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        if self.opt.mode == '1':
            style,_ = self.net_E2(self.img_a,self.landmark2)
            distributions, f, style = self.net_E(self.mask, self.img_m,self.landmark1,style)
            results = self.net_G(distributions,self.scale_mask, f, style)
            self.img_g = results
            self.img_out = (1 - self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth
        elif self.opt.mode == '2':
            #style,f2 = self.net_E2(self.img_a, self.landmark2).detach()
            #input = torch.cat([self.img_a,self.img_m],dim=0)
            style,f = self.net_E2(self.img_a,self.landmark2)
            f2 = f[2]
            #style,_ = style_all.chunk(2)
            #f2,f3 = f[2].chunk(2)
            style_adj = self.net_E3(self.img_m,self.mask)
            distributions, f, style_adj = self.net_E(self.mask, self.img_m,self.landmark1,style_adj)
            results = self.net_G(distributions,self.scale_mask, f, style_adj)
            self.img_g = results
            self.img_out = (1 - self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth
            self.style = style
            self.style_adj = style_adj
            self.get_uncertainty(f2)#,f3,style)
    
    def get_uncertainty(self,f2,f3=None,style=None):
        B, C, H, W = f2.shape
        
        x = f2.view(B, -1, H * W)  # BxCx(HxW)
        x_T = torch.transpose(x,1,2)
        y = (f2* self.scale_mask[0]).view(B, -1, H * W)  # BxCx(HxW)
        y_T = torch.transpose(y,1,2)

        #grad = [torch.autograd.grad(style[0,i,0],y[0,:,:],allow_unused=True) for i in range(512)]

        m = self.scale_mask[0].view(B, -1, H * W)
        x_norm = torch.norm(x_T,2,2,keepdim=True)
        y_norm = torch.norm(y_T,2,2,keepdim=True)
        attention_map =  torch.bmm(x_T, y) / torch.clamp(torch.bmm(x_norm, torch.transpose(y_norm,1,2)),min=1e-6) * m
        attention_map_max = torch.max(attention_map,dim=2)[0]
        #self.uncertainty =  torch.sum(attention_map_max * (self.style_grad.view(B, -1, H * W)),dim=2)

        attention_map_max_scale = task.scale_up_pyramid(attention_map_max.view(B,1,H,W),4)
        attention_map_max_scale.reverse()
        self.attention_map_max = attention_map_max_scale