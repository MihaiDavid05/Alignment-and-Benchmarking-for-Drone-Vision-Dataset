r""" PF-PASCAL dataset """

import os
import numpy as np
import torch
import cv2
from .dataset import CorrespondenceDataset
from PIL import Image


class DroneDataset(CorrespondenceDataset):

    def __init__(self, benchmark, datapath, thres, device, split, cam, ann_dir, img_dir, drone_dir):
        r""" Drone dataset constructor """
        super(DroneDataset, self).__init__(benchmark, datapath, thres, device, split)

        self.metadata['drone'] = ('', 'splits', '120m', '', 'img', img_dir, drone_dir, ann_dir)
        self.src_img_path = os.path.join(self.base_path, self.metadata['drone'][6])
        self.trg_img_path = os.path.join(self.base_path, self.metadata['drone'][5])
        self.label_img_path = os.path.join(self.base_path, self.metadata['drone'][7])

        with open(self.spt_path, 'r') as f:
            self.train_data = f.read().splitlines()

        self.trg_imnames = [img_name + '.jpg' for img_name in self.train_data]
        self.src_imnames = [img_name + '.jpg' for img_name in self.train_data]
        self.label_imnames = [img_name + '.png' for img_name in self.train_data]
        self.cls_ids = []
        self.cls = []
        self.cam = cam

    def __getitem__(self, idx):
        r"""Construct and return a batch for PF-WILLOW dataset"""

        # Image names
        sample = dict()
        sample['src_imname'] = self.src_imnames[idx]
        sample['trg_imname'] = self.trg_imnames[idx]
        sample['label_imname'] = self.label_imnames[idx]

        # Image tensors
        sample['src_img'] = self.get_image(self.src_imnames, idx, src=True)
        sample['trg_img'] = self.get_image(self.trg_imnames, idx, src=False)

        # Key-points
        sample['src_kps'] = torch.tensor(0.0).to(self.device)
        sample['trg_kps'] = torch.tensor(0.0).to(self.device)

        # The number of pairs in training split
        sample['datalen'] = len(self.train_data)

        # Transform source & target images (normalization)
        if self.transform:
            sample = self.transform(sample)
        sample['src_img'] = sample['src_img'].to(self.device)
        sample['trg_img'] = sample['trg_img'].to(self.device)
        
        # Threshold on number of points
        sample['pckthres'] = self.get_pckthres(sample).to(self.device)

        # # src_bbox
        sample['src_bbox'] = torch.zeros(4).to(self.device)

        # # trg_bbox
        sample['trg_bbox'] = torch.zeros(4).to(self.device)

        return sample

    def get_image(self, img_names, idx, src=True):
        r"""Return image tensor"""
        if src:
            img_name = os.path.join(self.src_img_path, img_names[idx])
        else:
            img_name = os.path.join(self.trg_img_path, img_names[idx])
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        if self.thres == 'bbox':
            return max(sample['trg_kps'].max(1)[0] - sample['trg_kps'].min(1)[0]).clone()
        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size()[1], sample['trg_img'].size()[2]))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts, idx):
        r"""Return key-points of an image"""
        point_coords = pts[idx, :].reshape(2, 10)
        point_coords = torch.tensor(point_coords.astype(np.float32))

        return point_coords

    @staticmethod
    def threshold_img(img_hsv):
        # define range of blue color in HSV
        lower_blue = np.array([115, 50, 50])
        upper_blue = np.array([125, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

        # define range of green color in HSV
        lower_green = np.array([55, 50, 50])
        upper_green = np.array([65, 255, 255])
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([5, 255, 255])
        mask0_red = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([175, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1_red = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask_red = mask0_red + mask1_red

        # set my output img to zero everywhere except my mask
        output_hsv = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))
        output_hsv[np.where(mask_red == 255)] = 1
        output_hsv[np.where(mask_blue == 255)] = 2
        output_hsv[np.where(mask_green == 255)] = 3

        return output_hsv.astype(np.uint8)
