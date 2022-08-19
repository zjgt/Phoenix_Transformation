from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torchvision.transforms.functional as TF

import numpy as np
from torchvision import transforms, datasets, models
import torch

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, root_dir1, root_dir2, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames1 = [join(root_dir1, x) for x in listdir(root_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(root_dir2, x) for x in listdir(root_dir2) if is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor


    def __getitem__(self, index):
        image_LR = Image.open(self.image_filenames1[index])
        image_HR = Image.open(self.image_filenames2[index])

        self.resize1 = transforms.Resize(size=(512, 512))
        self.resize2 = transforms.Resize(size=(1024, 1024))
        image_LR = self.resize1(image_LR)
        image_HR = self.resize2(image_HR)

        i, j, h, w = transforms.RandomCrop.get_params(
            image_LR, output_size=(self.crop_size, self.crop_size))
        image_LR = TF.crop(image_LR, i, j, h, w)
        image_HR = TF.crop(image_HR, i * self.upscale_factor, j * self.upscale_factor, h * self.upscale_factor, w * self.upscale_factor)
        image_LR = TF.to_tensor(image_LR)
        image_HR = TF.to_tensor(image_HR)
        return image_LR, image_HR

    def __len__(self):
        return len(self.image_filenames1)


class ValDatasetFromFolder(Dataset):
    def __init__(self, root_dir1, root_dir2, crop_size, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames1 = [join(root_dir1, x) for x in listdir(root_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(root_dir2, x) for x in listdir(root_dir2) if is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        image_LR = Image.open(self.image_filenames1[index])
        image_HR = Image.open(self.image_filenames2[index])

        self.resize1 = transforms.Resize(size=(512, 512))
        self.resize2 = transforms.Resize(size=(1024, 1024))
        image_LR = self.resize1(image_LR)
        image_HR = self.resize2(image_HR)

        i, j, h, w = transforms.RandomCrop.get_params(
            image_LR, output_size=(self.crop_size, self.crop_size))
        image_LR = TF.crop(image_LR, i, j, h, w)
        image_HR = TF.crop(image_HR, i * self.upscale_factor, j * self.upscale_factor, h * self.upscale_factor, w * self.upscale_factor)
        hr_scale = Resize(self.crop_size * self.upscale_factor, interpolation=Image.BICUBIC)
        image_restore = hr_scale(image_LR)
        image_LR = TF.to_tensor(image_LR)
        image_HR = TF.to_tensor(image_HR)
        image_restore = TF.to_tensor(image_restore)

        return image_LR, image_restore, image_HR

    def __len__(self):
        return len(self.image_filenames1)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)