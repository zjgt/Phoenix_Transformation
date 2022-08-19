import argparse
import time
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator
cuda_visible_devices = 0

parser = argparse.ArgumentParser(description='Test Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--src_folder', type=str, help='folder name for the images to be expanded')
parser.add_argument('--dest_folder', type=str, help='folder name for storing the expanded images')
parser.add_argument('--model_name', default='netG_epoch_4_76.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
SRC_FOLDER = opt.src_folder
DEST_FOLDER = opt.dest_folder
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    cuda_visible_devices = 1
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

names = os.listdir(SRC_FOLDER)
for IMAGE_NAME in names:
    NAME = os.path.join(SRC_FOLDER, IMAGE_NAME)
    image = Image.open(NAME)
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.clock()
    out = model(image)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(os.path.join(DEST_FOLDER, IMAGE_NAME))



