import torch
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import torchvision.io as io
from torchvision.transforms import v2
from torchvision import tv_tensors

class AVA_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, phase='train'):
        self.root_path     = root_path
        self.split_path    = os.path.join(root_path, 'annotations', 'ava_v2.2', split_path)
        self.data_path     = os.path.join(root_path, data_path)
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate
        self.phase         = phase
        self.img_size      = img_size
        self.valid_frame   = range(902, 1799) 
        self.num_classes   = 80

        # --- DEFINING TRANSFORMS (The Magic of v2) ---
        # These handle Image, Video, AND Bounding Boxes simultaneously
        if phase == 'train':
            self.transform = v2.Compose([
                # 1. Random Crop & Resize (replaces your manual random_crop)
                v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), antialias=True),
                
                # 2. Random Flip (replaces your manual flip)
                v2.RandomHorizontalFlip(p=0.5),
                
                # 3. Color Jitter (Only applies to video/image, ignores boxes)
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                
                # 4. Convert to Float [0.0-1.0]
                v2.ToDtype(torch.float32, scale=True),
                
                # 5. Normalize
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
                # 6. Sanitize (Removes boxes that became invalid/empty after cropping)
                v2.SanitizeBoundingBoxes()
            ])
        else:
            # Validation Transform
            self.transform = v2.Compose([
                v2.Resize(size=(img_size, img_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.read_ann_csv()

    def read_ann_csv(self):
        # (Use the exact same Pandas code from the previous robust solution)
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

        # --- 1. Load Video Clip ---
        clip_tensors = []
        
        # Read keyframe first for dimensions
        key_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
        try:
            key_frame_img = io.read_image(key_frame_path)
            H_orig, W_orig = key_frame_img.shape[1:]
        except:
            return self.__getitem__((index + 1) % len(self))

        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i * self.sampling_rate
            if cur_frame_idx < 1: cur_frame_idx = 1
            path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            try:
                cur_frame = io.read_image(path)
                if cur_frame.shape[0] == 1: cur_frame = cur_frame.repeat(3, 1, 1)
                if cur_frame.shape[1:] != (H_orig, W_orig):
                    # Use simple resize for raw loading
                    cur_frame = v2.functional.resize(cur_frame, (H_orig, W_orig), antialias=False)
            except:
                cur_frame = torch.zeros((3, H_orig, W_orig), dtype=torch.uint8)
            clip_tensors.append(cur_frame)
        
        # --- FIX STEP 1: Stack as [T, C, H, W] ---
        # Changing dim=1 to dim=0 creates [16, 3, H, W]
        clip = torch.stack(clip_tensors, dim=0) 

        # --- FIX STEP 2: Wrap in tv_tensors.Video ---
        # This tells v2 "The first dimension is Time, ignore it for ColorJitter"
        clip = tv_tensors.Video(clip)
        
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
            
            label = np.zeros(self.num_classes)
            for x in cur_frame_dict[raw_bboxes]:
                label[x - 1] = 1
            
            boxes_list.append(box)
            labels_list.append(label)
            
        boxes_np = np.array(boxes_list)
        labels_np = np.array(labels_list)

        # --- 3. WRAP IN TV_TENSORS ---
        if len(boxes_np) > 0:
            tv_boxes = tv_tensors.BoundingBoxes(
                torch.from_numpy(boxes_np).float(), 
                format="XYXY", 
                canvas_size=(H_orig, W_orig)
            )
            tv_labels = torch.from_numpy(labels_np)
        else:
            tv_boxes = tv_tensors.BoundingBoxes(torch.zeros((0,4)), format="XYXY", canvas_size=(H_orig, W_orig))
            tv_labels = torch.zeros((0, self.num_classes))

        # --- FIX: Wrap targets in a Dictionary ---
        # This creates the structure: (Video, Dictionary)
        # SanitizeBoundingBoxes knows exactly how to handle this.
        target = {
            "boxes": tv_boxes,
            "labels": tv_labels
        }

        # --- 4. APPLY TRANSFORM ---
        # Pass as a tuple of (Input, TargetDict)
        clip, target = self.transform(clip, target)

        # Unpack the dictionary back into variables
        tv_boxes = target["boxes"]
        tv_labels = target["labels"]

        # --- 5. Post-Process ---
        # clip is currently [T, C, H, W] (e.g. 16, 3, 224, 224)
        
        # Permute back to [C, T, H, W] for YOWO
        clip = clip.permute(1, 0, 2, 3) 
        
        # Convert boxes back to Relative (0-1)
        H_new, W_new = clip.shape[-2:] 
        
        if tv_boxes.shape[0] > 0:
            tv_boxes[:, 0] /= W_new
            tv_boxes[:, 1] /= H_new
            tv_boxes[:, 2] /= W_new
            tv_boxes[:, 3] /= H_new
            tv_boxes = tv_boxes.clamp(0, 1)

        # --- NEW RETURN LOGIC ---
        # The 'get_origin_image' check MUST come first
        if get_origin_image:
            # We must load the original_image as a numpy array (H, W, C)
            # which is what your old cv2.imread did
            original_image = key_frame_img.permute(1, 2, 0).numpy()
            return original_image, clip, tv_boxes, tv_labels

        elif self.phase == 'test':
            return clip, tv_boxes, tv_labels, video_name, str_sec

        elif self.phase == 'train':
            return clip, tv_boxes, tv_labels

def build_ava_dataset(config, phase):
    root_path = config['data_root'].replace('/', '\\')
    # Note: We no longer pass 'transform=...' because it's defined inside __init__ now
    return AVA_dataset(root_path=root_path, split_path="ava_train_v2.2.csv" if phase=='train' else "ava_val_v2.2.csv",
                       data_path="frames", clip_length=config['clip_length'], 
                       sampling_rate=config['sampling_rate'], img_size=config['img_size'], phase=phase)