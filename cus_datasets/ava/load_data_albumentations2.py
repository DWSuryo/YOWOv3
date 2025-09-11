import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from PIL import Image
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AVA_dataset(data.Dataset):
    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, transform=None, phase='train'):
        self.root_path = root_path
        self.split_path = os.path.join(root_path, 'annotations', 'ava_v2.2', split_path)
        self.data_path = os.path.join(root_path, data_path)
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.valid_frame = range(902, 1799)
        self.num_classes = 80
        self.phase = phase
        self.img_size = img_size
        self.read_ann_csv()

    def read_ann_csv(self):
        my_dict = dict()
        print(f"split path: {self.split_path}")
        with open(self.split_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                key = '/'.join([row[0], row[1]])
                subkey = '/'.join([row[2], row[3], row[4], row[5]])
                sub_dict = my_dict.setdefault(key, dict())
                sub_list = sub_dict.setdefault(subkey, [])
                sub_list.append(int(row[6]))
        self.data_dict = my_dict
        self.data_list = list(my_dict.keys())
        self.data_len = len(self.data_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index, get_origin_image=False):
        video_name, sec = self.data_list[index].split('/')
        str_sec = sec
        sec = int(sec)
        key_frame_idx = (sec - 900) * 30 + 1
        video_path = os.path.join(self.data_path, video_name)

        clip = []
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i * self.sampling_rate
            if cur_frame_idx < 1:
                cur_frame_idx = 1

            cur_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            cur_frame = cv2.imread(cur_frame_path)
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
            clip.append(cur_frame)

        if get_origin_image:
            key_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
            original_image = cv2.imread(key_frame_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        H, W, _ = clip[-1].shape
        cur_frame_dict = self.data_dict[self.data_list[index]]

        for raw_bboxes in cur_frame_dict.keys():
            box = list(raw_bboxes.split('/'))
            box = [float(x) for x in box]
            # box[0] *= W
            # box[1] *= H
            # box[2] *= W
            # box[3] *= H
            
            class_indices = cur_frame_dict[raw_bboxes]
            
            boxes.append(box)
            labels.append(class_indices)
        
        boxes = np.array(boxes)
        
        bboxes_list = boxes.tolist()
        
        multi_label_bboxes = []
        multi_label_indices = []
        for box, class_indices in zip(bboxes_list, labels):
            for class_idx in class_indices:
                multi_label_bboxes.append(box)
                multi_label_indices.append(class_idx - 1)

        # Apply transformations to the key frame and its targets
        transformed = self.transform(image=clip[-1], bboxes=multi_label_bboxes, labels=multi_label_indices)
        
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['labels']

        img_H, img_W = transformed_image.shape[1:]
        transformed_bboxes_pixel = []
        for bbox in transformed_bboxes:
            bbox_pixel = [bbox[0] * img_W, bbox[1] * img_H, bbox[2] * img_W, bbox[3] * img_H]
            transformed_bboxes_pixel.append(bbox_pixel)
        
        # Define a simplified transform for non-key frames.
        # This pipeline should contain only non-geometric transforms (e.g., normalization)
        # that are applied consistently to all frames.
        # Reusing the existing transform but applying it to each frame is incorrect
        # because the geometric transforms would be different for each frame.
        other_frames_transform = build_transform_pipeline(self.img_size, self.phase, exclude_geo=True)

        # Apply the same non-geometric transformations to the rest of the frames in the clip
        clip_tensors = []
        for i, frame in enumerate(clip):
            if i == len(clip) - 1:
                # Use the already transformed key frame
                clip_tensors.append(transformed_image)
            else:
                # Apply non-geometric transforms to other frames
                other_frames_transform = build_transform_pipeline(self.img_size, self.phase, exclude_geo=True)
                transformed_frame = other_frames_transform(image=frame)['image']
                clip_tensors.append(transformed_frame)

        # Stack the list of tensors into a single tensor
        # Shape will be [C, T, H, W]
        # clip_stacked = torch.stack(clip_tensors, dim=1)

        # Convert the list of tensors to a single numpy array
        clip_array = np.array([t.cpu().numpy() for t in clip_tensors])

        # Convert the numpy array to a tensor
        clip_stacked = torch.from_numpy(clip_array).permute(1, 0, 2, 3)
        
        grouped_targets = {}
        for bbox, label in zip(transformed_bboxes, transformed_labels):
            bbox_tuple = tuple(bbox)
            if bbox_tuple not in grouped_targets:
                grouped_targets[bbox_tuple] = np.zeros(self.num_classes, dtype=np.float32)
            # CRITICAL FIX: Convert label to an integer
            label_int = int(label)
            grouped_targets[bbox_tuple][label_int] = 1.0

        boxes_tensor = torch.as_tensor(list(grouped_targets.keys()), dtype=torch.float32)
        labels_tensor = torch.as_tensor(list(grouped_targets.values()), dtype=torch.float32)

        if get_origin_image:
            return original_image, clip_stacked, boxes_tensor, labels_tensor
        elif self.phase == 'test':
            return clip_stacked, boxes_tensor, labels_tensor, video_name, str_sec
        else:
            return clip_stacked, boxes_tensor, labels_tensor


def build_transform_pipeline(img_size, phase, exclude_geo=False):
    if phase == 'train' and not exclude_geo:
        # This is for the key frame with bboxes and labels
        return A.Compose([
            A.RandomSizedCrop(min_max_height=(int(img_size*0.8), img_size), size=(img_size, img_size), p=0.5),
            A.Resize(height=img_size, width=img_size),
            A.ColorJitter(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
    
    elif phase == 'train' and exclude_geo:
        # This is the pipeline for the other frames, which have no bboxes.
        # It must be a separate A.Compose without bbox_params.
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    else: # For the test/validation phase
        # This pipeline should also have bbox_params for the key frame.
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))

def build_ava_dataset(config, phase):
    root_path = config['data_root']
    data_path = "frames"
    clip_length = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size = config['img_size']

    root_path = root_path.replace('/', os.path.sep)

    if phase == 'train':
        split_path = "ava_train_v2.2.csv"
        transform_pipeline = build_transform_pipeline(img_size, phase)
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=transform_pipeline, phase=phase)
    elif phase == 'test':
        split_path = "ava_val_v2.2.csv"
        transform_pipeline = build_transform_pipeline(img_size, phase)
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=transform_pipeline, phase=phase)