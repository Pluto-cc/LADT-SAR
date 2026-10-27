import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_improved_PTM1
# from util import util
import models.external_function as external_function
class GuidedPix2PixModel(BaseModel):
    def name(self):
        return 'GuidedPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.Vggloss = external_function.VGGLoss().to(self.device)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.use_GAN = opt.use_GAN
        if self.use_GAN:
            self.loss_names = ['G', 'G_GAN', 'G_L1', 'G_L2', 'D_real', 'D_fake']
        else:
            self.loss_names = ['G', 'G_L1', 'G_L2']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            if self.use_GAN:
                self.model_names = ['G', 'D']
            else:
                self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks ############### MODIFIIED
        guide_nc = opt.guide_nc + opt.z_nc if opt.use_z else opt.guide_nc
        self.netG = networks_improved_PTM1.define_G(
            input_nc=opt.input_nc,
            guide_nc=guide_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            netG=opt.netG,
            n_layers=opt.n_layers,
            norm=opt.norm,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            ftm_mode=opt.ftm_mode,
            use_gmtm=not opt.no_gmtm,
            independent_decoder=opt.independent_decoder,
            gpu_ids=self.gpu_ids,
        )

        if self.isTrain & self.use_GAN:
            use_sigmoid = opt.no_lsgan
            self.netD = networks_improved_PTM1.define_D(input_nc=opt.input_nc + opt.guide_nc + opt.output_nc, ndf=opt.ndf, netD=opt.netD, n_layers_D=opt.n_layers_D,
                                                norm=opt.norm, use_sigmoid=use_sigmoid, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(0)

            # define loss functions
            if self.use_GAN:
                self.criterionGAN = networks_improved_PTM1.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.use_GAN:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
#########################MODIFIED
    def set_input(self, input):
        self.real_A = input['Xs'].to(self.device)
        self.real_B = input['Xt'].to(self.device)
        self.guide_A = input['Ps'].to(self.device)
        self.guide = input['Pt'].to(self.device)
        if self.opt.use_z:
            self.z = input['z'].to(self.device)
########################MODIFIED
    def forward(self):
        if self.opt.use_z:
            guide = torch.cat([self.guide, self.z], dim=1)
            guide_A = torch.cat([self.guide_A, torch.zeros_like(self.z)], dim=1)
        else:
            guide = self.guide
            guide_A = self.guide_A

        self.fake_B, self.fake_A = self.netG(self.real_A, guide, guide_A)

    def get_output(self):
        return self.fake_B

    def get_class(self):
        return self.class_B

    def get_current_data(self):
        return {'A':self.real_A, 'guide':self.guide, 'B':self.real_B, 'output':self.fake_B}

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.guide, self.fake_B), 1))

        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.guide, self.real_B), 1)

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        if self.use_GAN:
            # GAN loss for target SAR image generation
            fake_AB = torch.cat((self.real_A, self.guide, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            # Target SAR image generation losses: fake_B -> real_B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_content_gen, self.loss_style_gen = self.Vggloss(self.fake_B, self.real_B)

            self.loss_G = (
                    self.loss_G_GAN
                    + self.loss_G_L1
                    + self.loss_content_gen * self.opt.lambda_L1
                    + self.loss_style_gen * self.opt.lambda_L1
            )

            # Auxiliary source SAR image reconstruction losses: fake_A -> real_A
            if not self.opt.no_reconstruct_loss:
                self.loss_G_L2 = self.criterionL2(self.fake_A, self.real_A) * self.opt.lambda_L1
                self.loss_content_gen1, self.loss_style_gen1 = self.Vggloss(self.fake_A, self.real_A)

                self.loss_G = (
                        self.loss_G
                        + self.loss_G_L2
                        + self.loss_content_gen1 * self.opt.lambda_L1
                        + self.loss_style_gen1 * self.opt.lambda_L1
                )
            else:
                self.loss_G_L2 = torch.tensor(0.0, device=self.device)
                self.loss_content_gen1 = torch.tensor(0.0, device=self.device)
                self.loss_style_gen1 = torch.tensor(0.0, device=self.device)

        else:
            # Target SAR image generation losses: fake_B -> real_B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.loss_content_gen, self.loss_style_gen = self.Vggloss(self.fake_B, self.real_B)

            self.loss_G = (
                    self.loss_G_L1
                    + self.loss_content_gen
                    + self.loss_style_gen
            )

            # Auxiliary source SAR image reconstruction losses: fake_A -> real_A
            if not self.opt.no_reconstruct_loss:
                self.loss_G_L2 = self.criterionL2(self.fake_A, self.real_A)
                self.loss_content_gen1, self.loss_style_gen1 = self.Vggloss(self.fake_A, self.real_A)

                self.loss_G = (
                        self.loss_G
                        + self.loss_G_L2
                        + self.loss_content_gen1
                        + self.loss_style_gen1
                )
            else:
                self.loss_G_L2 = torch.tensor(0.0, device=self.device)
                self.loss_content_gen1 = torch.tensor(0.0, device=self.device)
                self.loss_style_gen1 = torch.tensor(0.0, device=self.device)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        if self.use_GAN:
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            # update G
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            # update G
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
