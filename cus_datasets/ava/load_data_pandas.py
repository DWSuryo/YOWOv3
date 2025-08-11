import pandas as pd
import albumentations as A
import torch
import torch.utils.data as data
import os
import cv2
import numpy as np
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
        
        # Use Pandas to read the entire CSV file at once
        self.annotations = pd.read_csv(self.split_path, header=None, names=['video_id', 'middle_frame_timestamp', 'person_box_x1', 'person_box_y1', 'person_box_x2', 'person_box_y2', 'action_id', 'person_id'])
        
        # Group annotations for quick lookup in __getitem__
        self.annotations_by_frame = self.annotations.groupby(['video_id', 'middle_frame_timestamp'])
        
        # Get a list of unique frames to iterate over
        self.frames = list(self.annotations_by_frame.groups.keys())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Retrieve annotations using Pandas
        video_id, frame_timestamp = self.frames[idx]
        frame_annotations = self.annotations_by_frame.get_group((video_id, frame_timestamp)).reset_index(drop=True)
        
        # Load image frame
        frame_path = os.path.join(self.data_path, video_id, f"{frame_timestamp:05d}.jpg")
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract bboxes and labels from the DataFrame
        bboxes = frame_annotations[['person_box_x1', 'person_box_y1', 'person_box_x2', 'person_box_y2']].values
        labels = frame_annotations['action_id'].values

        # Prepare annotations for Albumentations (x1, y1, x2, y2, label)
        alb_bboxes = [(b[0], b[1], b[2], b[3], l) for b, l in zip(bboxes, labels)]

        # Apply Albumentations transform
        augmented = self.transform(image=image, bboxes=alb_bboxes)
        image_transformed = augmented['image']
        bboxes_transformed = augmented['bboxes']

        # Convert back to PyTorch-friendly format
        if bboxes_transformed:
            boxes = torch.tensor([b[:4] for b in bboxes_transformed], dtype=torch.float32)
            labels = torch.tensor([b[4] for b in bboxes_transformed], dtype=torch.long)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)

        return image_transformed, boxes, labels