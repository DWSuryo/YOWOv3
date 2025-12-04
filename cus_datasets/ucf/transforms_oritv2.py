import torch
import numpy as np
import random
import torchvision.transforms.functional as F
# Note: PIL is no longer needed

class UCF_transform():
    """
    Transforms for validation/testing.
    Accepts:
        clip (torch.Tensor): [C, F, H, W] uint8 tensor
        targets (np.array): [N, 84] absolute pixel coordinates
    Returns:
        clip (torch.Tensor): [C, F, H_out, W_out] float32, normalized
        targets (torch.Tensor): [N, 84] relative coordinates
    """
    def __init__(self, img_size):
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __call__(self, clip, targets):
        # clip is [C, F, H, W]
        C, F, H, W = clip.shape

        # 1. Convert boxes to relative
        targets = torch.as_tensor(targets).float()
        if targets.shape[0] > 0:
            targets[:, :4] /= torch.tensor([W, H, W, H])
            targets[:, :4] = targets[:, :4].clamp(0, 1) #_torchvision

        # 2. Resize clip
        # F.resize works on [C, ...], [..., H, W], and 5D clips
        clip = F.resize(clip, [self.img_size, self.img_size], antialias=True)
        
        # 3. Convert to float [0, 1]
        clip = clip.float() / 255.0

        # 4. Normalize
        clip = (clip - self.mean) / self.std

        return clip, targets


class Augmentation(object):
    """
    Transforms for training.
    Accepts:
        clip (torch.Tensor): [C, F, H, W] uint8 tensor
        targets (np.array): [N, 84] absolute pixel coordinates
    Returns:
        clip (torch.Tensor): [C, F, H_out, W_out] float32, normalized
        targets (torch.Tensor): [N_new, 84] relative coordinates (filtered)
    """
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def rand_scale(self, s):
        scale = random.uniform(1, s)
        return scale if random.randint(0, 1) else 1./scale

    def random_crop(self, clip, width, height):
        # Same logic as before
        dw = int(width * self.jitter)
        dh = int(height * self.jitter)
        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)
        swidth = width - pleft - pright
        sheight = height - ptop - pbot

        # Use F.crop on the tensor
        # F.crop takes (top, left, height, width)
        cropped_clip = F.crop(clip, ptop, pleft, sheight, swidth)
        
        # Calculate deltas (same as before)
        dx = (float(pleft) / width)
        dy = (float(ptop) / height)
        sx = float(swidth) / width
        sy = float(sheight) / height

        return cropped_clip, dx, dy, sx, sy

    def random_distort_image(self, clip):
        # clip is [C, F, H, W], uint8
        
        # torchvision.v2.ColorJitter can do this in one step
        # and works on 5D Tensors!
        # This is a much cleaner, faster implementation.
        # We apply jitter with 50% probability
        
        # if random.random() < 0.5:
        #     # We must convert to float for jitter ops
        #     clip = clip.float()
            
        #     # Get random factors (same as rand_scale)
        #     dsat = self.rand_scale(self.saturation)
        #     # Exposure in PIL's V channel is gamma.
        #     dexp = self.rand_scale(self.exposure) 
        #     dhue = random.uniform(-self.hue, self.hue)
            
        #     # Apply transforms one by one
        #     # F.adjust_... works on [..., C, H, W]
        #     clip = F.adjust_saturation(clip, dsat)
        #     clip = F.adjust_gamma(clip, dexp)
        #     clip = F.adjust_hue(clip, dhue) # range [-0.5, 0.5]
            
        #     # Convert back to uint8 for next step
        #     clip = clip.clamp(0, 255).to(torch.uint8)

        if random.random() < 0.5:
            dsat = self.rand_scale(self.saturation)
            dexp = self.rand_scale(self.exposure) 
            dhue = random.uniform(-self.hue, self.hue) # [-0.1, 0.1]
            
            # Permute from [C, F, H, W] -> [F, C, H, W]
            # This treats the frames as a "batch" of images
            clip = clip.permute(1, 0, 2, 3) 
            
            # Apply color ops to the "batch"
            clip = F.adjust_saturation(clip, dsat)
            clip = F.adjust_gamma(clip, dexp) 
            clip = F.adjust_hue(clip, dhue)   
            
            # Permute back to [C, F, H, W]
            clip = clip.permute(1, 0, 2, 3)
            
            clip = clip.clamp(0, 1) # Clamp after color ops

        return clip

    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
        # This function is pure numpy, no changes needed
        # It converts from absolute to relative *and* applies transform
        target[..., 0] = np.minimum(0.999, np.maximum(0, (target[..., 0] / ow - dx) / sx))
        target[..., 1] = np.minimum(0.999, np.maximum(0, (target[..., 1] / oh - dy) / sy))
        target[..., 2] = np.minimum(0.999, np.maximum(0, (target[..., 2] / ow - dx) / sx))
        target[..., 3] = np.minimum(0.999, np.maximum(0, (target[..., 3] / oh - dy) / sy))

        # Refine target (filter small boxes)
        refine_target = []
        for i in range(target.shape[0]):
            tgt = target[i]
            # Calculate width/height in *new* coordinate system
            bw = (tgt[2] - tgt[0]) * self.img_size
            bh = (tgt[3] - tgt[1]) * self.img_size
            if bw < 1. or bh < 1.:
                continue
            refine_target.append(tgt)

        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])
        return refine_target

    def __call__(self, clip, target):
        # clip is [C, F, H, W] uint8, target is [N, 84] np.array
        oh = clip.shape[2]
        ow = clip.shape[3]
        
        # 1. Random crop (torchvision.F)
        clip, dx, dy, sx, sy = self.random_crop(clip, ow, oh)

        # 2. Resize (torchvision.F)
        clip = F.resize(clip, [self.img_size, self.img_size], antialias=True)

        # 3. Random flip (torchvision.F)
        flip = random.randint(0, 1)
        if flip:
            clip = F.hflip(clip)

        # 4. Distort (torchvision.F)
        # This will convert clip to float and back to uint8
        # A bit inefficient, but directly replaces your logic
        # clip = self.random_distort_image(clip) 
        
        # --- A better way for Distort + Normalize ---
        # 5. Convert to float [0, 1]
        clip = clip.float() / 255.0

        # 6. Apply Color Jitter (as float)
        if random.random() < 0.5:
            dsat = self.rand_scale(self.saturation)
            dexp = self.rand_scale(self.exposure) 
            dhue = random.uniform(-self.hue, self.hue) # [-0.1, 0.1]
            
            # Permute from [C, F, H, W] -> [F, C, H, W]
            # This treats the frames as a "batch" of images
            clip = clip.permute(1, 0, 2, 3) 
            
            # Apply color ops to the "batch"
            clip = F.adjust_saturation(clip, dsat)
            clip = F.adjust_gamma(clip, dexp) 
            clip = F.adjust_hue(clip, dhue)   
            
            # Permute back to [C, F, H, W]
            clip = clip.permute(1, 0, 2, 3)
            
            clip = clip.clamp(0, 1) # Clamp after color ops
        # --- End Distort ---

        # 7. Process target (Numpy)
        if target is not None and target.shape[0] > 0:
            target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
            if flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
        else:
            target = np.array([])
            
        # 8. Normalize (Torch)
        clip = (clip - self.mean) / self.std
        
        # 9. Convert target to tensor
        target = torch.as_tensor(target).float()

        return clip, target