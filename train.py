import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from data.OTS_loader import OTSDataset
from torch.utils.data import DataLoader
import os
import torch
from util import util

import warnings

warnings.filterwarnings("ignore")
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
test_data = OTSDataset('D:\DataSet\dehazing\Reside\SOTS\outdoor\hazy\\',
                           'D:\DataSet\dehazing\Reside\SOTS\outdoor\gt\\', istrain=False)
testing_data_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=1,
                                 shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
avg_psnr_dict = {}
avg_psnr_list  = []
for epoch in range(1, opt.total_epoch+1):
    epoch_start_time = time.time()

    # if (epoch-1) % 5 == 0:
    #     # reinitialize data_load at each epoch for random pairing
    #     data_loader.initialize(opt)
    #     dataset = data_loader.load_data()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.PSNR_freq == 0:
        avg_psnr = 0
        psnr_sum = 0
        for batch in testing_data_loader:
            haze, gt = batch[0].to(device), batch[1].to(device)
            fake_gt = model.validation(haze)
            fake_gt = torch.clamp(fake_gt, -1, 1)
            psnr = model.batch_PSNR(fake_gt, gt)
            psnr_sum += psnr
        avg_psnr = psnr_sum / len(testing_data_loader)
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
        message = '%i: %.4f' % (epoch, avg_psnr)
        avg_psnr_list.append(message)

        # avg_psnr_dict.update({epoch: avg_psnr})
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'PSNR_log.txt')

        with open(file_name, 'wt') as PSNR_file:
            PSNR_file.write('------------ Epoch and PSNR -------------\n')
            lenght = len(avg_psnr_list)
            for i in range (lenght):
                PSNR_file.write('%s\n' % avg_psnr_list[i])
            PSNR_file.write('-------------- End ----------------\n')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.total_epoch, time.time() - epoch_start_time))


    model.update_learning_rate()
