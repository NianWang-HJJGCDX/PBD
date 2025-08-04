import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.dataset_mode = 'paired'
    opt.resize_or_crop = 'none'
    # 'single' for testing on real-world haze images; 'paired' for testing on synthetic haze images, which calculates PSNR and SSIM during testing
    opt.model = 'test'
    opt.serial_batches = True  # no shufflemodel
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    ssim_dict = []
    psnr_dict = []
    for i, data in enumerate(dataset):
        model.set_input(data)
        psnr0, ssim0 = model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
        psnr_dict.append(psnr0)
        ssim_dict.append(ssim0)
    print("mean ssim is ", np.mean(ssim_dict, 0))
    print("mean psnr is ", np.mean(psnr_dict, 0))
    webpage.save()
