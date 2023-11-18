# PBD model for single image dehazing
A novel weakly supervised image dehazing model via physics-based decomposition. The code will be updated after paper is published.

# Our Configuration
All the experiments were conducted by a PyTorch framework on a PC with one RTX 3090 GPU. 

# Our Environment
Python (3.7)
PyTorch (1.12.0) with CUDA (11.6)
torchvision (0.13.0)
numpy (1.21.6)

# Testing
## Download the pretrained models.
1. Get the model on [Google drive]. It's trained on RESIDE-unpaired.

2. Create a folder named `checkpoints`, and unzip `PBD.zip` in `./checkpoints`.
Now, your directory tree should look like
```
<PBD_root>
├── checkpoints
│   ├── PBD
│   │   ├── web (for visual images during training)
│   │   ├── 60_net_G.pth
│   │   ├── opt.txt
│   │   ├── loss_log.txt
│   │   └── PSNR_log.txt
│   ...
...
```
(60_net_G.pth) is the trained model, (opt.txt) records the options for training, (loss_log.txt) records the loss after per 100 iteration,  (PSNR_log.txt) records the PSNR value after per 2 Epoch.

## For visual results
1. Download the pretrained model (see above).

2. Run the following command from <PBD_root>.
```
Run test.py before setting --dataroot(in configuration): [change to your own root of testing images] --model(in base_options.py): test --name(in base_options.py): PBD --resize_or_crop(in base_options.py): none --which_epoch(in test_options.py): 60.
```
The results will be saved in the folder `./results/PBD/test_60/images`.

3. When testing on synthetic images, you may need to quantitatively compare  with GTs.
   
*  For PSNR and SSIM, you can run Eval.py before setting the image root and choosing datasets (We has  preliminarily set up the name correspondence between dehazing results and the corresponding GTs in different data sets):

```
   imgs_dehaze = glob('D:\Results\dehazing\BeDDE\whole\DCP\\*.png')
   imgs_gt = 'D:\Results\dehazing\BeDDE\whole\gt\\'
```
```
    if __name__=='__main__':
        eval_SOTS_outdoor()
        # eval_OHAZE()
        # eval_BeDDE()
```

*  For CIEDE, you should run SOTS_eval.m or OHAZE_eval.m in Matlab before changing the root of GTs of SOTS and OHAZE data set, respectively.

*  For VI,RI,VSI,  you should run my_eval.m in Matlab before changing the root of GTs of BeDDE data set. Note that the main procedure was writen by ourselves. We cancel the city index of BeDDE set by original authors. For convenience, for this  procedure, you only need put the dehazing results and GTs of BeDDE respectively in one folder like a common way.

# Training
## Train PBD on RESIDE-unpaired 
1. Download RESIDE-unpaired on the links of original paper: [Google drive](https://drive.google.com/file/d/1SjQwESy8nwVO7pC3JRW7vXvJ6Qqk6Et4/view?usp=sharing) or [BaiduYun Disk](https://pan.baidu.com/s/1pqy-Ka9b9xVaeumdNSZAWQ) (Key: bswu).
2. Open visdom by `python -m visdom.server` (Optional)
3. Run the following command  from <PBD_root>.
```
train.py before setting --dataroot(in configuration): [change to your own root of training images] --model(in base_options.py): PBD --name(in base_options.py): PBD --resize_or_crop(in base_options.py): resize_and_crop --total_epoch(in train_options.py): 60 --niter_decay(in train_options.py): 60 --load_size(in base_options.py): 286 --fineSize(in base_options.py): 256 --num_threads:  0 --display_freq:  100 --print_freq: 100 --PSNR: 2 --which_model_netD: DWD --which_model_netG: resnet_9blocks --which_model_netDepth: unet_256 --which_model_netBeta: unet_256 --which_model_netA: AGenerator --gan_mode: one of [vanilla| lsgan | wgangp], wgangp will lead to gradient explosion.
```
Moreover, you should set the image root of validation data set during training in train.py. This is used to compute PSNR and record it during training.

    test_data = OTSDataset('D:\DataSet\dehazing\Reside\SOTS\outdoor\hazy\\','D:\DataSet\dehazing\Reside\SOTS\outdoor\gt\\', istrain=False)
    testing_data_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=1,shuffle=False)

Duirng the traing, your directory tree should look like
```
<PBD_root>
├── checkpoints
│   ├── PBD
│   │   ├── web (for visual images during training)
│   │   ├── 5_net_G (the model of generator of the 5th epoch)
│   │   ├── 5_net_D (the model of discriminator of the 5th epoch)
│   │   ├── ...
│   │   ├── latest_net_G
│   │   ├── latest_net_D
│   │   ├── opt.txt
│   │   ├── loss_log.txt
│   │   └── PSNR_log.txt
│   ...
...
```
License：MIT

For any questions, please do not hesitate to contact me at nianwang04@outlook.com
