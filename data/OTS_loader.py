import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
from glob2 import glob
import os
EPS=1e-12

class OTSDataset(data.Dataset):
    def __init__(self, data_dir, data_dir_clean, istrain=True, flip=True):
        super(OTSDataset, self).__init__()
        self.scale_size = 286
        self.size = 256
        self.hazy_img_list = glob(data_dir+'*.jpg')
        self.clean_img_list = []
        self.isTrain = istrain
        self.Flip = flip
        if self.isTrain:
            for img in self.hazy_img_list:
                gt_name = img.split('\\')[-1].split('_')[0]+'.jpg'
                if os.path.exists(data_dir_clean + gt_name):
                    self.clean_img_list.append(data_dir_clean + gt_name)
        else:

            for img in self.hazy_img_list:
                img_name = img.split('\\')[-1]
                gt_name = img_name
                # name = clean_data_dir+img.split("_")[0]+'.jpg'
                self.clean_img_list.append(data_dir_clean + gt_name)


    def name(self):
        return 'ITSDataset'


    def initialize(self, opt):
        pass


    def __getitem__(self, index):
        if self.isTrain:
            hazy_img = Image.open(self.hazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')


            if random.uniform(0, 1.0) > 0.5:
                hazy_img = hazy_img.transpose(Image.FLIP_LEFT_RIGHT)
                clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)


            w_s, h_s = hazy_img.size

            w_offset_s = random.randint(0, max(0, w_s - self.size  - 1))
            h_offset_s = random.randint(0, max(0, h_s - self.size - 1))

            hazy_img = transforms.ToTensor()(hazy_img)
            clean_img = transforms.ToTensor()(clean_img)


            hazy_img = hazy_img[:, h_offset_s:h_offset_s + self.size, w_offset_s:w_offset_s + self.size]
            clean_img = clean_img[:, h_offset_s:h_offset_s + self.size, w_offset_s:w_offset_s + self.size ]

            hazy_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hazy_img)
            clean_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(clean_img)

            return hazy_img,clean_img

        else:
            hazy_img = Image.open(self.hazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')

            w_s = 512
            h_s = 512
            # clean_img = clean_img.crop((10,10,630,470))
            hazy_img = hazy_img.resize((w_s, h_s), Image.BICUBIC)
            clean_img = clean_img.resize((w_s, h_s), Image.BICUBIC)

            transform_list = []
            transform_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
            trans = transforms.Compose(transform_list)
            img_name = self.hazy_img_list[index].split('\\')[-1]

            return trans(hazy_img), trans(clean_img),img_name

    def __len__(self):
        return len(self.hazy_img_list)