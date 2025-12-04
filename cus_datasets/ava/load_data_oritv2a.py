import torch
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import csv
import torchvision.io as io
import torchvision.transforms.functional as F

# Make sure this import path is correct for your project
from cus_datasets.ucf.transforms_oritv2 import Augmentation, UCF_transform
            
class AVA_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, transform=None, phase='train'):
        self.root_path     = root_path
        self.split_path    = os.path.join(root_path, 'annotations', 'ava_v2.2', split_path)
        self.data_path     = os.path.join(root_path, data_path)
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
        Reads the AVA CSV annotation file using pandas with explicit data types.
        """
        print(f"Reading split path with pandas: {self.split_path}")
        
        # 1. Define column names
        col_names = ['video_id', 'sec', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']
        
        # 2. Define explicit data types
        #    Reading 'sec' as a string is the safest, matching original CSV logic.
        col_types = {
            'video_id': str,
            'sec': int,       # Read '902' as a string
            'x1': str,        # Read '0.372' as a string
            'y1': str,
            'x2': str,
            'y2': str,
            'action_id': int, # Read '80' as an int
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
        #    No .astype(str) is needed for 'sec' as it's already a string.
        df['key'] = df['video_id'] + '/' + df['sec'].astype(str)
        df['subkey'] = df['x1'] + '/' + df['y1'] + '/' + df['x2'] + '/' + df['y2']

        # 5. Group by key and subkey, and aggregate action_ids into a list.
        #    'action_id' is already int, so no .astype(int) needed here.
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

        video_name, sec = self.data_list[index].split('/')
        str_sec = sec
        sec = int(sec)
        key_frame_idx = (sec - 900) * 30 + 1
        video_path = os.path.join(self.data_path, video_name)

        # --- Torchvision Loading ---
        
        # 1. Get keyframe path and read it to establish H, W
        key_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
        try:
            key_frame_img = io.read_image(key_frame_path)
            H_orig, W_orig = key_frame_img.shape[1:]
        except Exception as e:
            # print(f"Warning: Failed to load keyframe {key_frame_path}. Skipping sample.")
            # Return a different sample
            return self.__getitem__((index + 1) % len(self)) 

        clip_tensors = []
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i * self.sampling_rate
            if cur_frame_idx < 1:
                cur_frame_idx = 1
            
            cur_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            
            try:
                cur_frame = io.read_image(cur_frame_path)
                
                if cur_frame.shape[0] == 1:
                    cur_frame = cur_frame.repeat(3, 1, 1)
                
                if cur_frame.shape[1:] != (H_orig, W_orig):
                    cur_frame = F.resize(cur_frame, (H_orig, W_orig), antialias=False)
                    
            except Exception as e:
                cur_frame = torch.zeros((3, H_orig, W_orig), dtype=torch.uint8)
                
            clip_tensors.append(cur_frame)

        # 'clip' is created as a [C, F, H, W] TENSOR here
        clip = torch.stack(clip_tensors, dim=1) 
        # --- End Torchvision Loading ---

        # --- Annotation Loading ---
        boxes = []
        labels = [] 
        cur_frame_dict = self.data_dict[self.data_list[index]]
        for raw_bboxes in cur_frame_dict.keys():
            box = list(raw_bboxes.split('/'))
            box = [float(x) for x in box]
            box[0] *= W_orig
            box[1] *= H_orig
            box[2] *= W_orig
            box[3] *= H_orig

            label = np.zeros(self.num_classes)
            for x in cur_frame_dict[raw_bboxes]:
                label[x - 1] = 1

            boxes.append(box)
            labels.append(label)

        boxes = np.array(boxes)
        labels = np.array(labels)
        
        targets = np.concatenate((boxes, labels), axis=1)

        # --- Apply Transform ---
        
        # **** NEW DEBUGGING BLOCK ****
        # This will catch the error if 'clip' is not a tensor
        if self.transform:
            if not isinstance(clip, torch.Tensor):
                print(f"\n\n[DEBUG ERROR] 'clip' is NOT a torch.Tensor!")
                print(f"Instead, 'clip' is a: {type(clip)}")
                print(f"Value of 'clip': {clip}\n\n")
            # This is the line from your traceback
            clip, targets = self.transform(clip, targets)
        
        # The transform will return torch tensors
        boxes = targets[:, :4]
        labels = targets[:, 4:]

        if get_origin_image: 
            original_image = io.read_image(key_frame_path).permute(1, 2, 0).numpy() # to HWC
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

    # This part seems specific to Windows, it's fine.
    print(root_path)
    root_path = root_path.replace('/', '\\')

    if phase == 'train':
        split_path = "ava_train_v2.2.csv"
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=Augmentation(img_size=img_size), phase=phase)
    elif phase == 'test':
        split_path = "ava_val_v2.2.csv"
        # This is where the UCF_transform is correctly passed
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=UCF_transform(img_size=img_size), phase=phase)