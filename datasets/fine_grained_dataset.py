from PIL import Image
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
import json
import pdb
import os


class FineGrainedDataset(Dataset):
    def __init__(self, opt, stage):
        self.opt = opt
        json_path = None
        if 'train' in stage:
            json_path = opt.train_json
        elif 'val' in stage:
            json_path = opt.val_json
        elif 'test' in stage:
            json_path = opt.test_json
        
        with open(json_path, 'r') as outfile:
            self.data_list = json.load(outfile)
        
        self.path_prefix = "/".join(json_path.split('/')[:-2])
        self.image_prefix = self.path_prefix + "/images/"
        annotation_path = self.path_prefix + "/annotations/annotations.json"
        self.annotation_path = annotation_path
        with open(annotation_path) as annotation_file:
            self.annotations = json.load(annotation_file)

        
        # print("\n\n\n=======")
        # print("inited with stage", stage, self.path_prefix)
        # print("=======\n\n\n")
        # print(self.data_list)

    def __getitem__(self, index):

        data = OrderedDict()
        file_name = self.data_list[index]
        img_path = self.image_prefix + file_name
        # read image
        img = Image.open(img_path).convert('RGB')
        if self.opt.gray:
            img = img.convert('L')
        ratio = min(1080/img.size[0], 1080/img.size[1])
        l = 32
        w, h = int(ratio*img.size[0]/l)*l, int(ratio*img.size[1]/l)*l
        o_w, o_h = img.size
        img = img.resize([w, h])

        #img = img.resize([int(ratio*img.size[0]), int(ratio*img.size[1])])

        ## read density map
        # get ground-truth path, dot, fix4, fix16, or adapt
        if 'fix4' in self.opt.dmap_type:
            temp = '_fix4.h5'
        elif 'fix16' in self.opt.dmap_type:
            temp = '_fix16.h5'
        elif 'adapt' in self.opt.dmap_type:
            temp = '_adapt.h5'
        elif 'dot' in self.opt.dmap_type:
            temp = '_dot.h5'
        else:
            print('dmap type error!')
        suffix = img_path[-4:]
        # suppose the ground-truth density maps are stored in ground-truth folder
        # gt_path = img_path.replace(suffix, temp).replace('images', 'ground-truths')
        # gt_file = h5py.File(gt_path, 'r')
        den = np.array(self.annotations[file_name])
        # reshape the dot map
        if 'dot' in self.opt.dmap_type:
            idx = den.nonzero()
            for i in range(len(idx[1])):
                idx[1][i] = int(idx[1][i] * h / o_h)
            for i in range(len(idx[2])):
                idx[2][i] = int(idx[2][i] * w / o_w)
            den = torch.zeros(2,h,w)
            den = np.array(den)
            den[idx] = 1

        # read roi mask
        if self.opt.roi:
            # get mask path
            # For Towards_vs_Away change the mask path
            mask_path = 'mask.h5'
            gt_file = h5py.File(mask_path, 'r')
            mask = np.asarray(gt_file['mask'])
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask[mask > 0] = 1
            data['mask'] = mask

        # read semantic map
        if self.opt.smap:
            # get segmentation map
            seg_path = img_path.replace('.png', '_seg.h5')
            gt_file = h5py.File(seg_path, 'r')
            smap = np.asarray(gt_file['seg'])
            smap = torch.from_numpy(smap)#.unsqueeze(0)
            data['smap'] = smap


        # transformation
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)
        den = torch.from_numpy(den)
        
        # return
        data['img'] = img
        data['den'] = den
        return data

    def __len__(self):
        return len(self.data_list)
