from . import ucf_config
import torch
import numpy as np

class UCF_transform():
    """
    Args:
        clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
        boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    Return:
        clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
        boxes : not change
    """

    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def to_tensor(self, video_clip):
        return [F.to_tensor(image) for image in video_clip]

    def normalize(self, clip, mean=ucf_config.MEAN, std=ucf_config.STD):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, clip, targets):
        W, H = clip[-1].size
        targets[:, :4] /= np.array([W, H, W, H])
        clip = [img.resize([self.img_size, self.img_size]) for img in clip]
        clip = self.to_tensor(clip)
        clip = torch.stack(clip, dim=1)
        clip = self.normalize(clip)
        targets = torch.as_tensor(targets).float()
        return clip, targets

import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from PIL import Image
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
import torchvision.transforms.functional as F

class Augmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, video_clip, target):
        video_clip_np = [np.array(img.convert('RGB')) for img in video_clip]

        # Determine original dimensions (all frames should have the same initial size)
        oh, ow, _ = video_clip_np[0].shape

        # Removed the unused 'transform_pipeline' A.Compose block

        # The `target` is expected to be a NumPy array of bounding boxes (N, 4).
        if target is not None and target.shape[0] > 0:
            bboxes = target.tolist() # Assuming target is (N, 4)
            category_ids = [0] * len(bboxes) # Dummy category ID if not provided
        else:
            bboxes = []
            category_ids = []

        # Generate the crop parameters
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        pleft = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pright = random.randint(-dw, dw) 
        pbot = random.randint(-dh, dh)
        
        # Calculate crop region and padding needs
        x_min_crop_region = pleft
        y_min_crop_region = ptop
        x_max_crop_region = ow - pright
        y_max_crop_region = oh - pbot
        
        pad_left_needed = max(0, -x_min_crop_region)
        pad_top_needed = max(0, -y_min_crop_region)
        pad_right_needed = max(0, x_max_crop_region - ow)
        pad_bottom_needed = max(0, y_max_crop_region - oh)

        # Updated crop coordinates on the *padded* image
        crop_x_min = x_min_crop_region + pad_left_needed
        crop_y_min = y_min_crop_region + pad_top_needed
        crop_x_max = x_max_crop_region + pad_left_needed
        crop_y_max = y_max_crop_region + pad_top_needed

        # Removed unused swidth_cropped and sheight_cropped

        # Define CustomColorDistortion class (kept inline as per your structure)
        class CustomColorDistortion(A.ImageOnlyTransform):
            def __init__(self, hue_range, sat_range, exp_range, always_apply=False, p=1.0):
                super().__init__(always_apply, p)
                self.hue_range = hue_range
                self.sat_range = sat_range
                self.exp_range = exp_range

            def apply(self, img, **params):
                image_pil = Image.fromarray(img)
                
                dhue = random.uniform(-self.hue_range, self.hue_range)
                dsat = self._rand_scale(self.sat_range)
                dexp = self._rand_scale(self.exp_range)

                image_pil = image_pil.convert('HSV')
                cs = list(image_pil.split())
                cs[1] = cs[1].point(lambda i: i * dsat)
                cs[2] = cs[2].point(lambda i: i * dexp)
                
                def change_hue(x):
                    x += dhue * 255
                    if x > 255:
                        x -= 255
                    if x < 0:
                        x += 255
                    return x

                cs[0] = cs[0].point(change_hue)
                image_pil = Image.merge(image_pil.mode, tuple(cs))
                image_pil = image_pil.convert('RGB')
                return np.array(image_pil)
            
            def _rand_scale(self, s):
                scale = random.uniform(1, s)
                if random.randint(0, 1):
                    return scale
                return 1./scale
        
        # --- Albumentations Pipeline Construction ---
        spatial_transform = A.ReplayCompose([
            A.Pad((pad_left_needed,
                         pad_top_needed,
                         pad_right_needed,
                         pad_bottom_needed),
                # padding=(pad_left_needed,
                #          pad_top_needed,
                #          pad_right_needed,
                #          pad_bottom_needed
                #         ),
                p=1.0
            ),
            A.Crop(
                x_min=crop_x_min,
                y_min=crop_y_min,
                x_max=crop_x_max,
                y_max=crop_y_max,
                p=1.0
            ),
            A.Resize(height=self.img_size, width=self.img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.0, label_fields=['category_ids']))
        # print(spatial_transform.__dir__())

        color_and_final_transform = A.Compose([
            CustomColorDistortion(
                hue_range=self.hue,
                sat_range=self.saturation,
                exp_range=self.exposure,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0)
        ])

        transformed_video_clip = []
        # Removed unused `transformed_bboxes = []`
                
        # First frame for replay
        replayed_spatial_transform_result = spatial_transform(
            image=video_clip_np[0],
            bboxes=bboxes,
            category_ids=category_ids
        )
        # print(dir(replayed_spatial_transform_result))
        
        first_frame_spatial = replayed_spatial_transform_result['image']
        transformed_bboxes_first_frame = replayed_spatial_transform_result['bboxes']
        
        transformed_video_clip.append(first_frame_spatial)

        # Apply the same "replay" to the rest of the frames
        for i in range(1, len(video_clip_np)):
            replayed = A.ReplayCompose.replay(
                replayed_spatial_transform_result['replay'],
                image=video_clip_np[i],
                bboxes=[], # Bboxes not passed here as per current logic
            )
            transformed_video_clip.append(replayed['image'])

        final_video_clip = []
        for img_np in transformed_video_clip:
            processed_frame = color_and_final_transform(image=img_np)['image']
            final_video_clip.append(processed_frame)

        video_clip_out = torch.stack(final_video_clip, dim=1)

        final_target = []
        if transformed_bboxes_first_frame:
            for bbox in transformed_bboxes_first_frame:
                xmin, ymin, xmax, ymax = bbox[0:4]
                final_target.append(list(bbox[0:4]))

        final_target = torch.as_tensor(np.array(final_target)).float()

        return video_clip_out, final_target

def UCF_collate_fn(batch_data):
    clips  = []
    boxes  = []
    labels = []
    for b in batch_data:
        clips.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
    
    clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
    return clips, boxes, labels


