import torch
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import cv2  # Use OpenCV for loading
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AVA_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, phase='train'):
        self.root_path     = root_path
        self.split_path    = os.path.join(root_path, 'annotations', 'ava_v2.2', split_path)
        self.data_path     = os.path.join(root_path, data_path)
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate
        self.phase         = phase
        self.img_size      = img_size
        self.num_classes   = 80
        
        # --- Modern Albumentations Pipelines ---
        
        # Dynamically create keys for all frames in the clip
        # We need a key for the main image ("image")
        # and additional keys for all other frames ("frame1", "frame2", ..., "frame15")
        self.frame_keys = ["image"] + [f"frame{i}" for i in range(1, self.clip_length)]
        
        # This tells Albumentations to apply the same transform to all frames
        additional_targets = {key: 'image' for key in self.frame_keys if key != "image"}

        if phase == 'train':
            # 1. We define ONE pipeline for all transforms
            self.transform = A.Compose([
                # --- Spatial Transforms ---
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                
                # --- Color & Tensor Transforms ---
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2() # Converts HWC -> CHW
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc', # [x_min, y_min, x_max, y_max]
                min_visibility=0.1,
                min_area=1.0,
                # Link our custom multi-hot labels to the boxes
                label_fields=['box_indices']
            ),
            # This is the magic: apply the same spatial transform
            # to all frames in the 'additional_targets' list.
            additional_targets=additional_targets
            )
            
        else: # 'test' or 'val' phase
            # Test transform is simpler
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            additional_targets=additional_targets # Still need this to process all frames
            )

        self.read_ann_csv()

    def read_ann_csv(self):
        # (This function is identical to the previous one)
        print(f"Reading split path with pandas: {self.split_path}")
        col_names = ['video_id', 'sec', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']
        col_types = {'video_id': str, 'sec': str, 'x1': str, 'y1': str, 
                     'x2': str, 'y2': str, 'action_id': int, 'person_id': int}
        df = pd.read_csv(self.split_path, header=None, names=col_names, dtype=col_types)
        df['key'] = df['video_id'] + '/' + df['sec']
        df['subkey'] = df['x1'] + '/' + df['y1'] + '/' + df['x2'] + '/' + df['y2']
        agg_actions = df.groupby(['key', 'subkey'])['action_id'].apply(list)
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

        # --- 1. Load Clip Frames ---
        clip_frames = [] # This will be a list of HWC RGB np.arrays
        
        key_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
        try:
            key_frame_img = cv2.imread(key_frame_path)
            if key_frame_img is None: raise FileNotFoundError
            H_orig, W_orig, _ = key_frame_img.shape
        except:
            return self.__getitem__((index + 1) % len(self))
        
        original_image = cv2.cvtColor(key_frame_img, cv2.COLOR_BGR2RGB)

        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i * self.sampling_rate
            if cur_frame_idx < 1: cur_frame_idx = 1
            path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            try:
                cur_frame = cv2.imread(path)
                if cur_frame is None: raise FileNotFoundError
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
                if cur_frame.shape[0] != H_orig or cur_frame.shape[1] != W_orig:
                    cur_frame = cv2.resize(cur_frame, (W_orig, H_orig))
            except:
                cur_frame = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
            clip_frames.append(cur_frame)

        # --- 2. Load Annotations ---
        boxes_list = []
        labels_list = [] 
        cur_frame_dict = self.data_dict[self.data_list[index]]
        
        for raw_bboxes in cur_frame_dict.keys():
            box = [float(x) for x in raw_bboxes.split('/')]
            box[0] *= W_orig
            box[1] *= H_orig
            box[2] *= W_orig
            box[3] *= H_orig
            
            label_vec = np.zeros(self.num_classes)
            for x in cur_frame_dict[raw_bboxes]:
                label_vec[x - 1] = 1
            
            boxes_list.append(box)
            labels_list.append(label_vec)
            
        boxes_np = np.array(boxes_list)
        labels_np = np.array(labels_list) # Our (N, 80) multi-hot labels

        # --- THIS IS THE FIX ---
        # Create a simple list of indices [0, 1, 2, ..., N-1]
        box_indices = list(range(len(boxes_np)))

        # --- 3. Apply Transforms (The Modern Way) ---
        
        # A. Build the input dictionary for the transform
        # e.g., {"image": frame0, "frame1": frame1, ..., "bboxes": ..., "action_labels": ...}
        transform_input = {key: frame for key, frame in zip(self.frame_keys, clip_frames)}
        transform_input['bboxes'] = boxes_np

        # Pass the simple indices list to be filtered
        transform_input['box_indices'] = box_indices
        
        # B. Call the transform ONCE
        data = self.transform(**transform_input)
        
        # C. Re-pack the clip
        transformed_clip = []
        for key in self.frame_keys:
            transformed_clip.append(data[key])

        final_boxes = np.array(data['bboxes'])
            
        # --- THIS IS THE FIX (Part 2) ---
        # Get the indices that survived the transform
        kept_indices = data['box_indices']
        
        # Use these indices to select the correct multi-hot labels
        if len(kept_indices) > 0:
            kept_indices = np.array(kept_indices).astype(int)
            final_labels = labels_np[kept_indices]
        else:
            final_labels = np.empty((0, self.num_classes)) # Handle case where all boxes are filtered
        
        # --- 4. Stack & Post-Process ---
        clip_tensor = torch.stack(transformed_clip, dim=1)
        
        if len(final_boxes) > 0:
            H_new, W_new = self.img_size, self.img_size
            final_boxes[:, [0, 2]] /= W_new
            final_boxes[:, [1, 3]] /= H_new
            final_boxes = np.clip(final_boxes, 0, 1)

        boxes_tensor = torch.from_numpy(final_boxes).float()
        labels_tensor = torch.from_numpy(final_labels).float()

        # --- 5. Return Logic ---
        if get_origin_image:
            return original_image, clip_tensor, boxes_tensor, labels_tensor
        elif self.phase == 'test':
            return clip_tensor, boxes_tensor, labels_tensor, video_name, str_sec
        else: # 'train'
            return clip_tensor, boxes_tensor, labels_tensor

def build_ava_dataset(config, phase):
    # This builder function remains the same
    root_path = config['data_root'].replace('/', '\\')
    split = "ava_train_v2.2.csv" if phase=='train' else "ava_val_v2.2.csv"
    
    return AVA_dataset(root_path=root_path, split_path=split,
                       data_path="frames", clip_length=config['clip_length'], 
                       sampling_rate=config['sampling_rate'], 
                       img_size=config['img_size'], phase=phase)