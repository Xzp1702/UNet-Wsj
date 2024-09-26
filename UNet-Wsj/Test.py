import time
from operator import add
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.UNet import Unet
from data_loader import test_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./data/TestDataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#load the model
model = Unet()
model.load_state_dict(torch.load('./cpts/PRNet_epoch_final.pth'))
model.cuda()
model.eval()
scores = []
test_datasets = ['external test set','local test set']
for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/images/'
    gt_root = dataset_path + dataset + '/masks/'
    test_loader = test_dataset(image_root, gt_root, 352)
    total_time = 0
    count = 0
    for i in tqdm(range(test_loader.size)):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        res = model(image)
        start_time = time.perf_counter()
        end_time = time.perf_counter()
        count += 1
        total_time += end_time-start_time
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ',save_path+name)
        cv2.imwrite(os.path.join(save_path, name), res*255)
    fps = count/total_time
    print('FPS:', fps)
    print('Test Done!')

