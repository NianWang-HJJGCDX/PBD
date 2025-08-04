from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain,
                                      self.gpu_ids)
        self.netA = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netA, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDepth = networks.define_G(opt.input_nc, 1, opt.ngf, opt.which_model_netDepth, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netBeta = networks.define_G(opt.input_nc, 1, opt.ngf, opt.which_model_netBeta, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        which_epoch = opt.which_epoch

        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):

        self.input_A = input['A']
        self.input_A.resize_(self.input_A.size()).copy_(self.input_A)
        self.image_paths = input['A_paths']
        if self.opt.dataset_mode == 'paired':
            self.input_B = input['B']
            self.input_B.resize_(self.input_B.size()).copy_(self.input_B)

    def test(self):
        self.netG.eval()
        self.netA.eval()
        self.netDepth.eval()
        self.netBeta.eval()
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        if self.opt.dataset_mode == 'paired':
            self.real_B = Variable(self.input_B)
        fake_B  = torch.clamp(self.fake_B, -1., 1.)
        real_B = torch.clamp(self.real_B, -1., 1.)
        fake_B = np.uint8(127.5 * (fake_B.data.cpu().numpy().squeeze() + 1))
        real_B = np.uint8(127.5 * (real_B.data.cpu().numpy().squeeze() + 1))
        fake_B = fake_B.astype(float) / 255.0
        real_B = real_B.astype(float) / 255.0
        psnr0 = PSNR(real_B, fake_B)
        ssim0 = SSIM(real_B, fake_B, win_size = 3, multichannel=True)
        print("psnr:", psnr0, "ssim:", ssim0)
        return psnr0, ssim0


    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('fake_B', fake_B)])
        # return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
