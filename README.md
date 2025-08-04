# PBD [IEEE TCSVT 2025]
This is the  Pytorch implementation of our paper "Weakly Supervised Image Dehazing via Physics-Based Decomposition",  which is accepted by [IEEE Transactions on Circuits and Systems for Video Technology] in 2025. If you are interested at this work, you can star the repository. Thanks! 

# Our Configuration
All the experiments were conducted by a PyTorch framework on a PC with one RTX 3090 GPU. 

# Our Environment
Python (3.7)
PyTorch (1.12.0) with CUDA (11.6)
torchvision (0.13.0)
numpy (1.21.6)

# Testing
1. Download the pretrained model (60_net_G.pth) at [Link1](https://pan.baidu.com/s/1GS5l3rMNTGpjMnqPsRrwzA?pwd=1234).

2. Find the path `checkpoints\\PBD` and put the "60_net_G.pth" in it. Then the directory tree should look like
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
where (60_net_G.pth) is the our pretrained model, (opt.txt) records the options for training, (loss_log.txt) records the loss after per 100 iteration,  (PSNR_log.txt) records the PSNR value after per 2 Epoch.

3. Find `options\\base_options.py` and set "--dataroot" to your own root of testing images. Note that the testing images should include two folders named "testA" and "testB", which respectively store the haze images and their pixel-aligned clear images. If the test dataset does not have clear images, simply create an empty folder named "testB".

4. Run "test.py" and the results will be saved in the folder `./results/PBD/test_60/images`.



# Training
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

