import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2 # Use OpenCV for image loading
import pickle
import numpy as np
from PIL import Image
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2 # To convert NumPy array to PyTorch Tensor

class AVA_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, transform=None, phase='train'):
        self.root_path = root_path
        self.split_path = os.path.join(root_path, 'annotations', 'ava_v2.2', split_path)
        self.data_path = os.path.join(root_path, data_path)
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.transform = transform # This will now be an Albumentations Compose object
        self.valid_frame = range(902, 1799)
        self.num_classes = 80
        self.phase = phase
        self.img_size = img_size

        self.read_ann_csv()

    def read_ann_csv(self):
        my_dict = dict()
        print(f"split path: {self.split_path}")

        try:
            with open(self.split_path, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    key = '/'.join([row[0], row[1]])
                    subkey = '/'.join([row[2], row[3], row[4], row[5]])
                    sub_dict = my_dict.setdefault(key, dict())
                    sub_list = sub_dict.setdefault(subkey, [])
                    sub_list.append(int(row[6]))
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {self.split_path}")
            raise
        except Exception as e:
            print(f"An error occurred while reading CSV: {e}")
            raise

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

        clip_images = []
        
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i * self.sampling_rate
            if cur_frame_idx < 1:
                cur_frame_idx = 1

            cur_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            
            image = cv2.imread(cur_frame_path)
            if image is None:
                print(f"Warning: Could not load image {cur_frame_path}. Skipping this clip.")
                # Recursively call __getitem__ with a new index.
                # Ensure this doesn't lead to infinite loops if many images are missing.
                # A more robust solution might be to filter out problematic samples during CSV reading.
                return self.__getitem__((index + 1) % len(self), get_origin_image)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            clip_images.append(image)

        if get_origin_image:
            key_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
            original_image = cv2.imread(key_frame_path)
            if original_image is None:
                 original_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        boxes_normalized = [] # Store normalized 0-1 boxes for Albumentations
        multi_hot_labels = [] # Store multi-hot labels separately

        # Get the original image dimensions from the first frame for Albumentations
        original_H, original_W, _ = clip_images[0].shape

        cur_frame_dict = self.data_dict[self.data_list[index]]
        for raw_bboxes_str in cur_frame_dict.keys():
            box_coords = [float(x) for x in raw_bboxes_str.split('/')] # These are already 0-1 normalized
            
            action_ids = cur_frame_dict[raw_bboxes_str]
            multi_hot_label = np.zeros(self.num_classes, dtype=np.float32)
            for action_id in action_ids:
                multi_hot_label[action_id - 1] = 1

            boxes_normalized.append(box_coords)
            multi_hot_labels.append(multi_hot_label)

        # Load and denormalize bounding box annotations
        boxes_normalized = [] # Store normalized 0-1 boxes for Albumentations
        multi_hot_labels = [] # Store multi-hot labels separately

        original_H, original_W, _ = clip_images[0].shape

        cur_frame_dict = self.data_dict[self.data_list[index]]
        for raw_bboxes_str in cur_frame_dict.keys():
            box_coords = [float(x) for x in raw_bboxes_str.split('/')]
            
            action_ids = cur_frame_dict[raw_bboxes_str]
            multi_hot_label = np.zeros(self.num_classes, dtype=np.float32)
            for action_id in action_ids:
                multi_hot_label[action_id - 1] = 1

            boxes_normalized.append(box_coords)
            multi_hot_labels.append(multi_hot_label)

        # Convert to numpy arrays
        boxes_normalized_np = np.array(boxes_normalized, dtype=np.float32)
        multi_hot_labels_np = np.array(multi_hot_labels, dtype=np.float32)


        transformed_clip = []
        
        if self.transform:
            # Albumentations expects bboxes as a list of lists.
            # If `boxes_normalized_np` is empty, `.tolist()` will return `[]`.
            
            # Apply to the first image. Pass bboxes_normalized directly.
            # Albumentations will handle cases where bboxes is an empty list.
            first_frame_transformed = self.transform(
                image=clip_images[0],
                bboxes=boxes_normalized_np.tolist(), # Pass the list of lists
                original_height=original_H,
                original_width=original_W
            )
            
            transformed_clip.append(first_frame_transformed['image'])
            
            # Re-apply for subsequent frames for spatial consistency
            if hasattr(self.transform, 'replay') and first_frame_transformed.get('replay'):
                replay_data = first_frame_transformed['replay']
                for i in range(1, len(clip_images)):
                    replayed_transform = A.ReplayCompose.replay(replay_data, image=clip_images[i])
                    transformed_clip.append(replayed_transform['image'])
            else:
                # Fallback: Apply transform independently to each frame.
                for i in range(1, len(clip_images)):
                    frame_transformed = self.transform(
                        image=clip_images[i],
                        bboxes=boxes_normalized_np.tolist(), # Pass the list of lists again
                        original_height=original_H,
                        original_width=original_W
                    )
                    transformed_clip.append(frame_transformed['image'])
            
            # Recover transformed bounding boxes.
            # Ensure transformed_boxes is [0, 4] if no boxes were found or kept.
            if len(first_frame_transformed['bboxes']) > 0:
                transformed_boxes = torch.tensor(first_frame_transformed['bboxes'], dtype=torch.float32)
                # Get the indices of the original bboxes that were kept by Albumentations
                original_indices = first_frame_transformed.get('bboxes_original_idx', range(len(multi_hot_labels_np)))
                transformed_labels = torch.tensor(multi_hot_labels_np[original_indices], dtype=torch.float32)
            else:
                # If no bounding boxes are returned by Albumentations (e.g., filtered out)
                transformed_boxes = torch.empty((0, 4), dtype=torch.float32)
                transformed_labels = torch.empty((0, self.num_classes), dtype=torch.float32)


        else: # No transforms defined, just convert to tensor
            transformed_clip = [ToTensorV2()(image=img)['image'] for img in clip_images]
            # Ensure transformed_boxes and transformed_labels have correct shapes even if empty
            if len(boxes_normalized_np) > 0:
                transformed_boxes = torch.tensor(boxes_normalized_np, dtype=torch.float32)
                transformed_labels = torch.tensor(multi_hot_labels_np, dtype=torch.float32)
            else:
                transformed_boxes = torch.empty((0, 4), dtype=torch.float32)
                transformed_labels = torch.empty((0, self.num_classes), dtype=torch.float32)


        clip = torch.stack(transformed_clip, dim=1) # Results in [C, num_frames, H, W]

        if get_origin_image:
            return original_image, clip, transformed_boxes, transformed_labels
        elif self.phase == 'test':
            return clip, transformed_boxes, transformed_labels, video_name, str_sec
        else:
            return clip, transformed_boxes, transformed_labels

def build_ava_dataset(config, phase):
    root_path = config['data_root']
    data_path = "frames"
    clip_length = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size = config['img_size']

    if os.name == 'nt':
        root_path = root_path.replace('/', '\\')
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define common bbox parameters.
    # We DO NOT use `label_fields` for multi-hot labels to avoid the TypeError.
    # Instead, we will manually filter `multi_hot_labels` based on filtered `bboxes`.
    bbox_params_common = A.BboxParams(
        format='pascal_voc', # [x_min, y_min, x_max, y_max] normalized (0-1)
        min_area=1.0,        # Minimum area in pixels for a box to be kept (after transform)
        min_visibility=0.1,  # Minimum visibility of a box to be kept (after transform)
        # No `label_fields` here for multi-hot labels!
    )

    if phase == 'train':
        split_path = "ava_train_v2.2.csv"
        # Use A.ReplayCompose to ensure spatial consistency across frames
        transform = A.ReplayCompose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=config.get('hue', 0.1),
                                 sat_shift_limit=config.get('saturation', 0.2), # Original was 1.5, which is very high. 0.2-0.3 is more common for sat/val.
                                 val_shift_limit=config.get('exposure', 0.2),   # Original was 1.5.
                                 p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ], bbox_params=bbox_params_common) # Pass the common bbox parameters
        
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=transform, phase=phase)

    elif phase == 'test':
        split_path = "ava_val_v2.2.csv"
        transform = A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ], bbox_params=bbox_params_common) # Pass the common bbox parameters

        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=transform, phase=phase)