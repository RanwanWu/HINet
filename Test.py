import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.HINet import HINet
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--test_path',type=str,default='./Datasets/RGBD_for_test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path


## ----------------------------    load the model     ----------------------------

model = HINet()
model.load_state_dict(torch.load('./pre/HINet/HINet_epoch_best.pth'), False)
model.cuda()
model.eval()


## ----------------------------    set test  dataset   ----------------------------

test_datasets = ['NJUD', 'STERE', 'NLPR', 'SSD', 'LFSD']
for dataset in test_datasets:
    save_path = './Salmaps/HINet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        _,res = model(image,depth)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
