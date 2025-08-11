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
from albumentations.pytorch import ToTensorV2 # For converting to PyTorch tensor

import random
# torchvision.transforms.functional is still useful for some tensor operations if needed, but Albumentations aims to replace much of it.
# import torchvision.transforms.functional as F

class AlbumentationsAugmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

        # Albumentations transforms
        # We'll construct the pipeline in the __call__ method to ensure random states are consistent per clip.
        # Some transforms need to be applied with the same parameters across frames,
        # so we'll either apply them to the list of images or use A.Compose with a seed.

    def __call__(self, video_clip, target):
        # video_clip is a list of PIL Images
        # target is a numpy array of shape (N, 4) or (N, 5) if class is included, using xmin, ymin, xmax, ymax

        # Convert PIL Images to NumPy arrays (Albumentations works with NumPy arrays)
        # Assuming input PIL Images are RGB, convert to HWC (Height, Width, Channel)
        video_clip_np = [np.array(img.convert('RGB')) for img in video_clip]

        # Determine original dimensions (all frames should have the same initial size)
        oh, ow, _ = video_clip_np[0].shape

        # Define the augmentation pipeline.
        # Important: For video augmentation, you need to ensure that random parameters (e.g., crop coordinates, flip)
        # are the same for all frames within a single video clip.
        # One way is to create a strong transform and apply it to the first image to get params,
        # then re-apply using A.Lambda, or use A.Compose with a fixed seed, or simply apply one by one.
        # A simpler way with Albumentations for consistent video augmentation is to generate the parameters once.

        # The jitter in original code is essentially a random crop.
        # Albumentations' RandomResizedCrop or RandomCrop can simulate this.
        # For simplicity and to directly map the jitter, we will use RandomCrop.
        # The original code's random crop can add padding, which is not directly a "crop" in the
        # traditional sense but rather cropping from an inflated canvas.
        # Let's try to replicate the effect.

        # The `jitter` in the original code allows for cropping "outside" the image boundaries
        # by creating a larger effective canvas from which to crop.
        # Albumentations' `RandomCrop` or `RandomResizedCrop` generally operates within the image bounds.
        # To replicate the original's random crop (which can effectively zoom out or in slightly),
        # we might need a custom approach or a combination of Pad and RandomCrop.

        # Let's simplify and use RandomResizedCrop as it handles both cropping and resizing.
        # The `scale` parameter in RandomResizedCrop is analogous to the 'jitter' effect.
        # Or, we can use Pad and then Crop.

        # Let's try to maintain the original logic of calculating pleft, pright etc.
        # This requires custom Lambda transforms in Albumentations.

        # Step 1: Random Crop (and get parameters for bbox adjustment)
        # This is the most complex part to directly map with a single Albumentations transform
        # because the original "crop" can actually expand the image virtually.
        # A.Pad and A.RandomCrop might achieve a similar effect.
        # The original code takes plef/prig/ptop/pbot and then resizes to img_size.
        # A.RandomSizedCrop(min_max_height=(int(oh*(1-self.jitter)), oh),
        #                   min_max_width=(int(ow*(1-self.jitter)), ow),
        #                   height=self.img_size, width=self.img_size, p=1.0)
        # This would perform a random crop and resize.

        # Let's try to stick closer to the original logic for crop coordinates.
        # Albumentations can accept coordinates for cropping.
        # We can calculate the crop parameters once and then apply to all frames.

        # Original crop logic:
        # dw =int(width * self.jitter)
        # dh =int(height * self.jitter)
        # pleft = random.randint(-dw, dw)
        # pright = random.randint(-dw, dw)
        # ptop = random.randint(-dh, dh)
        # pbot = random.randint(-dh, dh)
        # swidth = width - pleft - pright
        # sheight = height - ptop - pbot
        # Note: These p-values can be negative, meaning the crop extends beyond original image.
        # This is not a standard "crop" for Albumentations.
        # If pleft is negative, it means we add padding to the left.
        # If pleft is positive, we crop from the left.

        # To handle this "random crop" (which is more like a random zoom/pan), we can:
        # 1. Calculate the crop coordinates (x_min, y_min, x_max, y_max)
        # 2. Use A.Crop with these coordinates.
        # 3. Handle padding if crop coordinates go beyond image boundaries.
        #    A.Pad can be used before or after.

        # Let's define the core pipeline with consistent random state for all frames.
        # We will use `AdditionalTargets` for video frames to ensure they are processed consistently.

        transform_pipeline = A.Compose([
            # Step 1: Custom Random Crop + Resize
            # This is tricky because the original 'random_crop' can effectively add borders.
            # Albumentations' RandomSizedCrop might be the closest, but it crops within bounds.
            # Let's approximate the original random crop behavior.
            # The original code effectively calculates a target top-left (pleft, ptop) and size (swidth, sheight)
            # then crops, then resizes.
            # If pleft/ptop are negative, it means effectively padding is applied.
            # Let's generate the crop parameters once.
            A.Lambda(
                name='CustomRandomCrop',
                always_apply=True,
                p=1.0,
                # This will calculate the crop parameters based on the original logic
                # and store them in the transform_state, so subsequent calls can use them.
                # This is an approximation as it's hard to exactly replicate the original's
                # handling of negative crop values (padding implicitly).
                # We'll focus on the target size and region.
                # The original `random_crop` has `dx`, `dy`, `sx`, `sy` for bbox adjustment.
                # Albumentations will handle bbox adjustment automatically if we pass the
                # correct bbox format and apply the transforms.
                # So we just need to ensure the cropping and resizing are correct.
                # A.RandomResizedCrop is probably the best direct replacement for a random crop followed by resize.
                # The `scale` parameter of RandomResizedCrop affects the zoom level, which is related to jitter.
                # We want to maintain a certain aspect ratio for the final image.
                # Original image is resized to self.img_size x self.img_size after cropping.
                # The `jitter` suggests the crop region could be smaller/larger than input.
                # Let's use RandomResizedCrop as a first attempt.
                # The 'scale' parameter controls the size of the cropped area relative to the original image.
                # scale=(1.0 - self.jitter, 1.0 + self.jitter)
                # No, the jitter works on width and height separately.
                # This makes RandomResizedCrop less direct for the exact jitter.

                # Let's manually apply the crop coordinates to each frame for consistency.
                # We need to compute pleft, ptop, swidth, sheight once.
                # `dw` and `dh` are maximum offsets.
                # `pleft`, `ptop` are the starting coordinates for cropping.
                # `swidth`, `sheight` are the dimensions of the cropped region.
                # This is more like A.Crop with dynamic coordinates and then A.Resize.
                # We need to get original width/height from the input.
                # The `additional_targets` will help here.

                # First, ensure all transforms are applied to the entire video clip.
                # Albumentations is typically image-centric. For video, you apply the same transform
                # parameters to each frame. A common way is to define an A.Compose and apply it
                # repeatedly with a fixed seed, or iterate through frames and apply the same transform instance.
            ),

            # A.HorizontalFlip(p=0.5), # Random flip
            # A.HueSaturationValue(
            #     hue_shift_limit=self.hue * 255, # A maps to -limit to +limit
            #     sat_shift_limit=np.log(self.saturation) * 100, # A uses percentage, original used multiplicative factor
            #     val_shift_limit=np.log(self.exposure) * 100, # A uses percentage, original used multiplicative factor
            #     p=1.0
            # ),
            # A.ColorJitter for exposure/saturation/hue as a single transform
            # The original code's `rand_scale` for saturation and exposure is not a direct range
            # but rather a multiplicative factor that can be 1/s or s.
            # Albumentations' `ColorJitter` has `brightness`, `contrast`, `saturation`, `hue` limits.
            # The `saturation_limit` and `hue_shift_limit` are in terms of shifts, not multiplicative.
            # We need to map the original `saturation` and `exposure` values to Albumentations' limits.
            # `hue` is a direct range.
            # For saturation and exposure, the original uses `dsat = rand_scale(saturation)` and `dexp = rand_scale(exposure)`.
            # This means the new value is `old_value * dsat` (or `dexp`).
            # Albumentations' `HueSaturationValue` uses `sat_shift_limit` and `val_shift_limit`.
            # These are typically additive or fractional shifts.
            # To simulate `x * dsat`, we would need `x + x * (dsat - 1)`.
            # If `dsat` can be `1/s` or `s`, then `dsat-1` can be `1/s - 1` or `s-1`.
            # This mapping is non-trivial for direct `HueSaturationValue`.
            # `ColorJitter` also might not directly match.

            # It's better to implement `random_distort_image` as a custom A.Lambda if exact behavior is needed.
            # For a close approximation, A.HueSaturationValue or A.ColorJitter can be used.
            # Let's try to map the parameters.
            # Hue: `dhue = random.uniform(-self.hue, self.hue)` -> `hue_shift_limit = self.hue * 255` (if PIL values are 0-255)
            # Saturation: `dsat` is multiplicative. Albumentations is additive shift.
            # Exposure: `dexp` is multiplicative. Albumentations is additive shift.

            # Given the complexity of directly mapping `random_distort_image` and `random_crop`
            # to single Albumentations transforms while maintaining the exact original behavior
            # (especially with bounding box handling), it's often more robust to
            # implement parts as `A.Lambda` functions that wrap the original logic or
            # more carefully craft a sequence of Albumentations transforms.

            # Let's rewrite the `__call__` function to integrate Albumentations properly for video data.
            # The core idea is to create *one* set of random parameters for the entire clip.
            # Then apply these parameters to each frame and its bounding boxes.
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.0, label_fields=['category_ids']))
        # We use 'pascal_voc' (xmin, ymin, xmax, ymax) as it's easier to map from the original.
        # min_area=1 ensures boxes that become too small are removed. Original had bw < 1 or bh < 1.
        # min_visibility=0.0 means even fully occluded boxes are kept unless area is 0.
        # `label_fields` is needed if you have class IDs. If `target` is just coordinates, omit.

        # The original target format is `[xmin, ymin, xmax, ymax]`.
        # Albumentations `pascal_voc` format expects `[x_min, y_min, x_max, y_max]`.
        # If your target numpy array contains class labels as well, like `[xmin, ymin, xmax, ymax, class_id]`,
        # then you'd define `bbox_params=A.BboxParams(..., label_fields=['class_ids'])` and pass `class_ids=[... ]`
        # alongside `bboxes`.

        # Let's define the Albumentations pipeline for a single frame, then iterate.
        # We need to ensure that transforms like crop and flip are *consistent* across frames.

        # --- Replicating the Original Logic with Albumentations ---

        # 1. Random Crop (and resize to img_size)
        # We need to calculate dx, dy, sx, sy for bbox adjustment as in the original code.
        # This is tricky with standard Albumentations transforms.
        # The original random crop applies a crop and then resizes.
        # The key is that `pleft`, `ptop` can be negative, meaning effective padding.
        # If `pleft` is -10, `swidth` is 200, it means the crop starts 10 pixels outside the left edge,
        # so we effectively need to pad by 10 pixels on the left before cropping.

        # Let's define a custom transform to mimic the random crop and get its parameters
        # or calculate the parameters outside and apply them with `A.Crop`.
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)

        # Calculate the actual crop region, considering negative p-values as padding
        x_min_crop_original = pleft
        y_min_crop_original = ptop
        x_max_crop_original = ow - pright
        y_max_crop_original = oh - pbot

        # We need to compute dx, dy, sx, sy relative to the original image dimensions for bbox transformation
        swidth_effective = x_max_crop_original - x_min_crop_original
        sheight_effective = y_max_crop_original - y_min_crop_original

        # The original code's `sx, sy` are `swidth/width`, `sheight/height` of the *cropped* region relative to original.
        # This implies a scaling factor for the content within the cropped region.
        # Then `dx, dy` are offsets.
        # Original: `dx = (float(pleft) / width)/sx`, `dy = (float(ptop) / height)/sy`
        # In terms of Albumentations, if we perform A.Crop followed by A.Resize,
        # Albumentations handles the bbox transformation automatically.
        # The issue is the "cropping outside" part.

        # A robust way to handle this custom crop:
        # Create a bounding box for the entire image: [0, 0, ow, oh]
        # Then apply the random crop using A.Crop or A.Lambda, and let Albumentations track the changes.

        # The `target` is expected to be a NumPy array of bounding boxes (N, 4).
        # Convert targets to Albumentations format if needed (e.g., add dummy labels).
        if target is not None and target.shape[0] > 0:
            # Assuming target is [xmin, ymin, xmax, ymax] normalized [0,1]
            # Convert to pixel coordinates for Albumentations (if it was normalized)
            # Then convert to 'pascal_voc' if it's not already.
            # Original code has target as already float and [0,1].
            # Then apply_bbox uses ow, oh to scale it back before applying deltas.
            # This means target is likely pixel coordinates that need to be normalized later by the user.
            # Or it's already normalized and apply_bbox denormalizes, then re-normalizes.
            # The apply_bbox code: `target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx))`
            # This suggests target is pixel coordinates. Let's assume `target` comes in as `[xmin, ymin, xmax, ymax]` in PIXELS.
            # If target is already normalized, skip `* ow` and `* oh`.
            # Given `target / ow * sx`, it means `target` is pixel coordinates.

            # Convert target to pixel coordinates (if not already) and add a dummy label for Albumentations.
            # Albumentations expects a list of tuples for bboxes: `[(xmin, ymin, xmax, ymax, class_id), ...]`
            bboxes = target.tolist() # Assuming target is (N, 4)
            category_ids = [0] * len(bboxes) # Dummy category ID if not provided
        else:
            bboxes = []
            category_ids = []

        # Create the Albumentations transform composition for one frame.
        # The order matters and should mimic the original.
        # Original: Random Crop -> Resize -> Flip -> Distort
        
        # Step 1: Random Crop and Resize
        # This is where the custom jitter logic is challenging.
        # We need to calculate `pleft`, `ptop`, `swidth`, `sheight` once for the clip.
        
        # Determine crop coordinates and final resize
        # The original jitter creates a 'sub-window' that can be smaller or larger than the original image,
        # and this sub-window is then resized to `img_size`.
        
        # Let's simulate the cropping and resizing with A.RandomResizedCrop.
        # The `scale` parameter of `RandomResizedCrop` controls the area of the crop.
        # (1 - jitter) for min_scale and (1) for max_scale might be a good start.
        # However, the original jitter can make the cropped area *larger* than the original image if `pleft` etc. are negative.
        # This is not directly supported by `RandomResizedCrop`'s `scale` range.

        # A more direct approach to replicate the original random_crop:
        # We need to compute `x_min_crop`, `y_min_crop`, `x_max_crop`, `y_max_crop` once.
        # Then we need to handle padding implicitly if `x_min_crop` or `y_min_crop` are negative
        # or `x_max_crop` > `width` or `y_max_crop` > `height`.

        # Let's generate the crop parameters first (pleft, ptop, swidth, sheight)
        # and then apply them.
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        pleft = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        
        # Original code used `swidth = width - pleft - pright`, `sheight = height - ptop - pbot`
        # `pright` and `pbot` are also random. This means the target width/height of the crop
        # also vary randomly.
        # This makes it a random crop of a *random size* that can also have padding.
        
        # Let's re-read the crop logic:
        # `swidth = width - pleft - pright`
        # `sheight = height - ptop - pbot`
        # `pleft`, `pright`, `ptop`, `pbot` can be negative or positive.
        # If `pleft` is negative, it means we are effectively extending the canvas to the left.
        # If `pleft` is positive, we are cropping from the left.
        
        # The most accurate way to do this in Albumentations is to:
        # 1. Apply A.Pad if any of pleft/ptop are negative or if swidth/sheight imply going beyond boundaries.
        # 2. Then apply A.Crop with `pleft`, `ptop`, `pleft + swidth`, `ptop + sheight`.
        # 3. Then apply A.Resize.

        # Let's calculate the padding needed for the 'random_crop'
        pad_left = max(0, -pleft)
        pad_top = max(0, -ptop)
        
        # The original crop rectangle starts at (pleft, ptop) and has size (swidth, sheight)
        # After padding, the image coordinates shift.
        # The new top-left for cropping would be (pleft + pad_left, ptop + pad_top)
        # And the new bottom-right for cropping would be (pleft + pad_left + swidth, ptop + pad_top + sheight)

        # However, the `dx`, `dy`, `sx`, `sy` are derived from the crop relative to the *original* image.
        # This suggests that `apply_bbox` expects the transform to be conceptualized as:
        # crop window (pleft, ptop, pleft+swidth, ptop+sheight) on the original image,
        # then resize this window to target size.

        # Albumentations' `A.Crop` and `A.Resize` will correctly handle bounding boxes if
        # applied sequentially. The challenge is the 'padding' aspect of `random_crop`.

        # Let's define the overall transform to be applied consistently.
        # We need to define `bbox_params` once for the `A.Compose`.

        # Custom logic for the random crop to calculate the crop parameters
        # and pass them to A.Crop.
        
        # We need to determine the final crop coordinates (x_min, y_min, x_max, y_max)
        # on the *padded* image (if padding is needed) or original image.
        # The original code's `pleft`, `ptop`, `swidth`, `sheight` define this region.
        
        # Let's directly implement the equivalent Albumentations pipeline for a single frame,
        # then apply it in a loop for the video.
        
        # Define the Albumentations transforms for one frame:
        # We'll use a `ReplayCompose` to ensure the same random parameters are used across frames.
        # Or, generate random parameters once and use `A.Lambda`.
        
        # For video augmentation consistency, we need to generate random states ONCE per video clip.
        # A good way to do this is to apply the transform to the first frame, record the parameters,
        # and then apply them to the rest of the frames using A.Lambda or A.ReplayCompose.
        
        # Let's define the main transform for a single image, then iterate.
        
        # The `random_distort_image` is best done as a custom `A.Lambda` for exact replication
        # because Albumentations' built-in color jitter might not perfectly match the specific HSV math.
        
        class CustomColorDistortion(A.ImageOnlyTransform):
            def __init__(self, hue_range, sat_range, exp_range, always_apply=False, p=1.0):
                super().__init__(always_apply, p)
                self.hue_range = hue_range
                self.sat_range = sat_range
                self.exp_range = exp_range

            def apply(self, img, **params):
                # img is numpy array in RGB
                image_pil = Image.fromarray(img)
                
                dhue = random.uniform(-self.hue_range, self.hue_range)
                dsat = self._rand_scale(self.sat_range)
                dexp = self._rand_scale(self.exp_range)

                image_pil = image_pil.convert('HSV')
                cs = list(image_pil.split())
                cs[1] = cs[1].point(lambda i: i * dsat)
                cs[2] = cs[2].point(lambda i: i * dexp)
                
                def change_hue(x):
                    x += dhue * 255 # Assuming PIL hue is 0-255
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
        # The most reliable way for video:
        # 1. Convert all PIL images to numpy arrays.
        # 2. Create an `A.Compose` for a single image.
        # 3. For transformations that affect spatial layout (crop, flip), apply them to the *first* frame
        #    and then use the `replay` mechanism to apply the same transformation to other frames.
        # 4. For color transformations, apply them independently or use a consistent random seed.

        # Let's simplify the random crop for Albumentations.
        # A.RandomResizedCrop is a good general purpose replacement for random crop + resize.
        # `scale` parameter corresponds to the jitter effect.
        # min_scale = (image_size - jitter * image_size) / image_size = 1 - jitter
        # max_scale = (image_size + jitter * image_size) / image_size = 1 + jitter
        # This is an approximation of the original jitter effect.

        # Initializing random state for transformations that need to be consistent across frames
        # For transforms that *do not* have `p` parameter (like Resize, Normalize), they are deterministic.
        # For random ones (Flip, Crop, Color), we need to handle consistency.

        # Use A.ReplayCompose for consistent spatial transformations
        # Or, manually generate random parameters and apply with Lambda.
        
        # Let's try to map the steps to Albumentations transforms:
        # 1. Random Crop: A.RandomResizedCrop or A.Crop and A.Resize.
        #    The `jitter` in the original implies a crop that can effectively zoom in/out.
        #    `RandomResizedCrop(height=img_size, width=img_size, scale=(1.0 - self.jitter, 1.0 + self.jitter), p=1.0)`
        #    This does a crop then resize. The `scale` param handles the "zoom".
        #    It's not exactly the same as the original's `pleft`, `ptop` for coordinate transformation for bboxes.

        # Let's use `A.Compose` and rely on its bbox handling.
        # For the custom crop, it's safer to determine the crop parameters outside and pass them.

        # Step 1: Compute custom crop parameters as in the original code.
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        pleft = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        
        pright = random.randint(-dw, dw) # Original also had random pright/pbot
        pbot = random.randint(-dh, dh)

        # The actual crop region relative to the original image
        x_min_crop_region = pleft
        y_min_crop_region = ptop
        x_max_crop_region = ow - pright
        y_max_crop_region = oh - pbot

        swidth_cropped = x_max_crop_region - x_min_crop_region
        sheight_cropped = y_max_crop_region - y_min_crop_region
        
        # The `random_crop` method then resizes this `cropped_clip` to `self.img_size`.
        # This is essentially: A.Crop + A.Resize.
        # And if `x_min_crop_region` or `y_min_crop_region` are negative, it implies padding.
        # Albumentations' `A.Crop` expects coordinates within the image.
        # We need to pad first if crop coordinates go beyond boundaries.
        
        pad_left_needed = max(0, -x_min_crop_region)
        pad_top_needed = max(0, -y_min_crop_region)
        pad_right_needed = max(0, x_max_crop_region - ow)
        pad_bottom_needed = max(0, y_max_crop_region - oh)

        # Updated crop coordinates on the *padded* image
        crop_x_min = x_min_crop_region + pad_left_needed
        crop_y_min = y_min_crop_region + pad_top_needed
        crop_x_max = x_max_crop_region + pad_left_needed
        crop_y_max = y_max_crop_region + pad_top_needed

        # Create a single transform that includes padding, cropping, resizing.
        # Use A.ReplayCompose to ensure consistent random states for flip and color distortions.
        # The spatial transforms (padding, crop, resize) are defined deterministically based on calculated params.

        # Let's define the sequence of transforms:
        # 1. Optional Pad (if crop region goes outside)
        # 2. Crop
        # 3. Resize
        # 4. Flip
        # 5. Custom Color Distortion
        # 6. Normalize
        # 7. ToTensorV2

        spatial_transform = A.Compose([
            A.Pad(
                # pads based on max needed values from random crop calculation
                pad_left=pad_left_needed,
                pad_top=pad_top_needed,
                pad_right=pad_right_needed,
                pad_bottom=pad_bottom_needed,
                p=1.0 # always apply if calculated
            ),
            A.Crop(
                x_min=crop_x_min,
                y_min=crop_y_min,
                x_max=crop_x_max,
                y_max=crop_y_max,
                p=1.0 # always apply with the computed crop box
            ),
            A.Resize(height=self.img_size, width=self.img_size, p=1.0),
            A.HorizontalFlip(p=0.5), # Random flip
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.0, label_fields=['category_ids']))

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
                max_pixel_value=255.0, # Images are 0-255 before normalization
                p=1.0
            ),
            ToTensorV2(p=1.0) # Converts to C, H, W and scales to 0-1 (before normalization)
        ])

        transformed_video_clip = []
        transformed_bboxes = []
        
        # Apply the same spatial transform to all frames
        # To achieve consistent spatial transforms (crop, flip) for all frames,
        # we can apply the spatial_transform to the first frame and then replay it.
        # This will save the random state (e.g., flip state) and apply it to subsequent calls.
        
        # First frame for replay
        replayed_spatial_transform = spatial_transform(
            image=video_clip_np[0],
            bboxes=bboxes,
            category_ids=category_ids # Pass category_ids if your bboxes include them
        )
        
        first_frame_spatial = replayed_spatial_transform['image']
        transformed_bboxes_first_frame = replayed_spatial_transform['bboxes']
        
        transformed_video_clip.append(first_frame_spatial)

        # Apply the same "replay" to the rest of the frames
        for i in range(1, len(video_clip_np)):
            replayed = A.ReplayCompose.replay(
                replayed_spatial_transform,
                image=video_clip_np[i],
                bboxes=[], # No bboxes for other frames during spatial replay if we only need to track for the first
                # Or, apply bbox to all frames during spatial transform for consistency, then filter.
                # Let's apply bbox transformation to first frame only as in original code.
            )
            transformed_video_clip.append(replayed['image'])

        # Now apply color distortion and final normalization/to_tensor to each frame independently
        # (original code applies color distortion independently per image in the loop)
        final_video_clip = []
        for img_np in transformed_video_clip:
            processed_frame = color_and_final_transform(image=img_np)['image']
            final_video_clip.append(processed_frame)

        # Stack the processed frames into a single tensor
        video_clip_out = torch.stack(final_video_clip, dim=1) # Stacks along dim 1, resulting in [C, T, H, W]

        # The bounding box logic in the original code filters out small boxes.
        # Albumentations' `min_area` in `BboxParams` handles this.
        # The `target` is modified only once based on the first frame's spatial transformation.
        # So we only need the bboxes from the first frame's spatial transform result.
        
        # Original code applied `apply_bbox` to `target` after crop and flip.
        # Albumentations handles this automatically if `bbox_params` are set.
        # The `transformed_bboxes_first_frame` already contains the adjusted bounding boxes.

        # Refine target based on original `apply_bbox` filtering logic (if min_area/min_visibility not sufficient)
        final_target = []
        if transformed_bboxes_first_frame:
            for bbox in transformed_bboxes_first_frame:
                # bbox is (x_min, y_min, x_max, y_max, class_id)
                xmin, ymin, xmax, ymax = bbox[0:4]
                
                # Check for width/height < 1.0 (after normalization by img_size implicitly done by Albumentations)
                # Albumentations outputs normalized coordinates (0-1) for 'pascal_voc' if input was,
                # or pixel if input was pixel. Assuming output is normalized after resize.
                # Original `apply_bbox` had `bw < 1. or bh < 1.` (in original pixel values after scale).
                # After resize to `img_size`, if `bw` or `bh` are less than 1 pixel, it's too small.
                # If Albumentations output is normalized (0-1), then it would be `width * img_size < 1` etc.
                
                # If using `min_area=1.0` and `min_visibility=0.0` with `BboxParams`,
                # Albumentations should already filter boxes whose area is less than 1 pixel.
                # Let's trust Albumentations' filtering for now.
                final_target.append(list(bbox[0:4])) # Extract only coordinates (xmin, ymin, xmax, ymax)

        final_target = torch.as_tensor(np.array(final_target)).float()

        return video_clip_out, final_target