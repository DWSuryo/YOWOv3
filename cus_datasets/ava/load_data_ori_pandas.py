import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from cus_datasets.ucf.transforms_ori import Augmentation, UCF_transform
from PIL import Image
import csv
import pandas as pd
            
class AVA_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, transform=Augmentation(), phase='train'):
       self.root_path     = root_path
       self.split_path    = os.path.join(root_path, 'annotations', 'ava_v2.2', split_path)
    #    self.split_path    = fr"{root_path} + '\' + 'annotations\' + split_path"  # manual
       self.data_path     = os.path.join(root_path, data_path)
    #    self.data_path     = fr"{root_path} + '\' + data_path"   # manual
       self.clip_length   = clip_length
       self.sampling_rate = sampling_rate
       self.transform     = transform
       self.valid_frame   = range(902, 1799) 
       self.num_classes   = 80
       self.phase         = phase
       self.img_size      = img_size

       self.read_ann_csv()
    
    def read_ann_csv(self):
        """
        Reads the AVA CSV annotation file using pandas with explicit data types
        to prevent inference errors.
        """
        print(f"Reading split path with pandas: {self.split_path}")
        
        # 1. Define column names
        col_names = ['video_id', 'sec', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']
        
        # 2. Define explicit data types for each column
        #    This is the key fix. We read everything as a string,
        #    just like the original csv.reader did.
        col_types = {
            'video_id': str,
            'sec': int,       # Read '902' as a string
            'x1': str,        # Read '0.372' as a string
            'y1': str,
            'x2': str,
            'y2': str,
            'action_id': int,  # Read '80' as a string
            'person_id': int
        }

        # 3. Read the CSV using the explicit types
        df = pd.read_csv(
            self.split_path, 
            header=None, 
            names=col_names,
            dtype=col_types  # Apply the explicit data types
        )

        # 4. Create the 'key' and 'subkey'
        #    No .astype(str) is needed because everything is already a string.
        df['key'] = df['video_id'] + '/' + df['sec'].astype(str)
        df['subkey'] = df['x1'] + '/' + df['y1'] + '/' + df['x2'] + '/' + df['y2']

        # 5. Group by key and subkey, and aggregate action_ids into a list.
        #    We must convert action_id to int *here*, before grouping.
        df['action_id'] = df['action_id'].astype(int)
        agg_actions = df.groupby(['key', 'subkey'])['action_id'].apply(list)
        
        # 6. Convert the grouped data into the final nested dictionary.
        my_dict = {}
        for key, group_df in agg_actions.reset_index().groupby('key'):
            sub_dict = group_df.set_index('subkey')['action_id'].to_dict()
            my_dict[key] = sub_dict

        self.data_dict = my_dict
        self.data_list = list(my_dict.keys())
        self.data_len  = len(self.data_list)
        
        print(f"Successfully loaded {self.data_len} unique keyframes.")

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index, get_origin_image=False):

        video_name, sec = self.data_list[index].split('/')  # linux
        # video_name, sec = self.data_list[index].split('\\') # windows
        str_sec = sec
        sec = int(sec)
        key_frame_idx = (sec - 900) * 30 + 1
        video_path = os.path.join(self.data_path, video_name)

        clip        = []
        boxes       = []
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i*self.sampling_rate

            if cur_frame_idx < 1:
                cur_frame_idx = 1

            # get frame
            cur_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            cur_frame = Image.open(cur_frame_path).convert('RGB')
            clip.append(cur_frame)

        if get_origin_image == True:
            key_frame_path     = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
            original_image     = cv2.imread(key_frame_path)

        boxes  = []
        labels = [] 

        W, H =  clip[-1].size   # extract image dimension

        cur_frame_dict = self.data_dict[self.data_list[index]]
        for raw_bboxes in cur_frame_dict.keys():
            box = list(raw_bboxes.split('/'))
            box = [float(x) for x in box]
            box[0] *= W
            box[1] *= H
            box[2] *= W
            box[3] *= H

            label = np.zeros(self.num_classes)
            for x in cur_frame_dict[raw_bboxes]:
                label[x - 1] = 1

            boxes.append(box)
            labels.append(label)

        boxes = np.array(boxes)
        labels = np.array(labels)
        
        # clip   : list of (num_frame) PIL image (RGB order, 0 .. 255)
        # boxes  : np array [nbox, 4], absolute coordinate (tl-br)
        # labels : np array [nbox, nclass]


        targets = np.concatenate((boxes, labels), axis=1)
        clip, targets = self.transform(clip, targets)
        
        boxes = targets[:, :4]
        labels = targets[:, 4:]

        # clip   : tensor [C, numframe, H, W] (RBG order)
        # boxes  : tensor [nbox, 4], relative coordinate (tl-br)
        # labels : tensor [nbox, nclass]
        if get_origin_image == True : 
            return original_image, clip, boxes, labels
        elif self.phase == 'test':
            return clip, boxes, labels, video_name, str_sec
        else:
            return clip, boxes, labels
        

def build_ava_dataset(config, phase):
    root_path     = config['data_root']
    data_path     = "frames"
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size      = config['img_size']

    # windows directory management
    print(root_path)
    root_path = root_path.replace('/', '\\')
    # print(os.path.exists(root_path))
    # dir_components = root_path.split('/')
    # print(dir_components)
    # joined_dir = os.path.join(*dir_components)
    # print(joined_dir)

    if phase == 'train':
        split_path    = "ava_train_v2.2.csv"
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=Augmentation(img_size=img_size), phase=phase)
    elif phase == 'test':
        split_path    = "ava_val_v2.2.csv"
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=UCF_transform(img_size=img_size), phase=phase)
        
