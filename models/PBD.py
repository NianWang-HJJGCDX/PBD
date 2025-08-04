import itertools
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_PSNR

class PBD(BaseModel):
    def name(self):
        return 'PBD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # initialize tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.guided_filter = networks.GuidedFilter(r=10, eps=1e-3)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout,opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDepth = networks.define_G(opt.input_nc, 1, opt.ngf, opt.which_model_netDepth, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netBeta = networks.define_G(opt.input_nc, 1, opt.ngf, opt.which_model_netBeta, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netA = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netA, opt.norm, not opt.no_dropout,opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
           self.netD = networks.define_D(3, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if  opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netDepth, 'Depth', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netDepth.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netDepth)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        # input_A: haze image; input_B: haze-free image in another scene
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if self.real_A.shape[1] > 1:
            # rgb2gray
            guidance = 0.2989 * self.real_A[:, 0, :, :] + 0.5870 * self.real_A[:, 1, :, :] + 0.1140 * self.real_A[:, 2, :, :]
        else:
            guidance = self.real_A
            # rescale to [0,1]
        guidance = (guidance + 1) / 2
        guidance = torch.unsqueeze(guidance, dim=1)
        self.fake_B = self.netG.forward(self.real_A)
        self.depth = self.netDepth.forward(self.real_A)
        self.Beta = self.netBeta.forward(self.real_A)
        self.T = torch.exp(-self.Beta*self.depth)

        if self.T.shape[2:4] != guidance.shape[2:4]:
            self.T = F.interpolate(self.T, size=guidance.shape[2:4], mode='nearest')

        self.refineT=self.guided_filter(guidance,self.T)
        self.refineT= self.refineT.repeat(1, 3, 1, 1)
        shape = self.fake_B.shape

        self.map_A = self.netA.forward(self.real_A)
        # recover B according to depth
        # refine_T_map = self.pre_filter.repeat(1, 3, 1, 1)
        self.fake_B2 = util.reverse_matting(self.real_A, self.refineT,self.map_A)

        # reconstruct A based on optical model
        self.fake_A = util.synthesize_matting(self.fake_B, self.refineT,self.map_A)

    # test images during traning, note: no backprop gradients

    def validation(self, haze):

        fake_gt = self.netG.forward(haze)
        return  fake_gt

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # stop backprop to the generator by detaching fake_B
        if self.opt.which_model_netD == 'multi':
            # Fake
            pred1_fake, pred2_fake = netD.forward(fake.detach())
            loss_D_fake = 0.5 * (self.criterionGAN(pred1_fake, False) + self.criterionGAN(pred2_fake, False))
            # Real
            pred1_real, pred2_real = netD.forward(real)
            loss_D_real = 0.5*(self.criterionGAN(pred1_real, True) + self.criterionGAN(pred2_real, True))
        else:
            # Fake
            pred1_fake = netD.forward(fake.detach())
            loss_D_fake = 0.5 * (self.criterionGAN(pred1_fake, False))
            # Real
            pred1_real = netD.forward(real)
            loss_D_real = 0.5 * (self.criterionGAN(pred1_real, True))

        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.which_model_netD == 'multi':
            pred1_fake_B, pred2_fake_B = self.netD.forward(self.fake_B)
            self.loss_G_B = 0.5*(self.criterionGAN(pred1_fake_B, True) + self.criterionGAN(pred2_fake_B, True))
        else:
            pred1_fake_B = self.netD.forward(self.fake_B)
            self.loss_G_B = 0.5 * (self.criterionGAN(pred1_fake_B, True) )
        # Second, reconstruction loss
        self.loss_G_L1 = self.criterionL1(self.fake_A, self.real_A)  # L1 loss for reconstruction
        self.loss_G_REC = 10 * self.loss_G_L1
        # Third, total variance loss
        self.loss_TV = self.criterionTV(self.T)

        self.loss_G = self.loss_G_REC + self.loss_G_B + self.loss_TV

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_B', self.loss_G_B.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('D', self.loss_D.data),
                            ('TV', self.loss_TV.data)
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        fake_B2 = util.tensor2im(self.fake_B2.data)

        return OrderedDict(
            [('Hazy', real_A), ('Haze-free1', fake_B),('Haze-free2', fake_B2),
             ('recover', fake_A),  ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def batch_PSNR(slef,img, imclean):
        Img = img.data.cpu().numpy().astype(np.float32)
        Iclean = imclean.data.cpu().numpy().astype(np.float32)
        PSNR = 0
        for i in range(Img.shape[0]):
            PSNR += compare_PSNR(Iclean[i, :, :, :], Img[i, :, :, :])
        return PSNR

