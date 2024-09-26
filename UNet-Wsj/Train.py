import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.UNet import Unet
from data_loader import get_loader, test_dataset
from utils import *
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options_cod import opt

cudnn.benchmark = True
image_root = opt.rgb_root
gt_root = opt.gt_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'MyNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("MyNet-Train_4_pairs")

model = Unet()
num_parms = 0

for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=352)
test_loader = test_dataset(test_image_root, test_gt_root, testsize=352)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
step = 0
writer = SummaryWriter(save_path + 'summary')
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.cuda()
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()

            y1 = model(images)
            y1 = F.interpolate(y1, size=352, mode='bilinear', align_corners=False)
            bce_iou_res = WiouWbceLoss(y1, gts)
            loss = bce_iou_res
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0 or epoch == 50:
            torch.save(model.state_dict(), save_path + 'UNet_epoch_{}.pth'.format(epoch))
        if epoch == 99:
            torch.save(model.state_dict(), save_path + 'UNet_epoch_final.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'PRNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)