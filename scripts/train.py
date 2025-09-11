import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

from math import sqrt
from utils.gradflow_check import plot_grad_flow
from utils.EMA import EMA
import logging
from utils.build_config import build_config
from cus_datasets.ucf.load_data import UCF_dataset
from cus_datasets.collate_fn import collate_fn
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3 
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
import shutil
from utils.flops import get_info

import tqdm

def train_model(config):

    # Save config file
    #######################################################
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config_new.yaml')
    shutil.copyfile(source_file, destination_file)
    #######################################################
    
    # create dataloader, model, criterion
    ####################################################
    dataset = build_dataset(config, phase='train')
    # print(dataset)
    
    # load data
    dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=collate_fn
                                 , num_workers=config['num_workers'], pin_memory=True)
    num_steps = len(dataloader)
    print(f"steps: {num_steps}")

    # # 1. Get a single batch from the DataLoader
    # data_iter = iter(dataloader)
    # batch_clip, batch_bboxes, batch_labels = next(data_iter)

    # # Check the batch size and move data to CPU
    # batch_size = batch_clip.shape[0]
    # batch_clip = batch_clip.cpu()
    # batch_bboxes = [b.cpu() for b in batch_bboxes]
    # batch_labels = [l.cpu() for l in batch_labels]

    # print(f"Batch size: {batch_size}")
    # print(f"Clip shape: {batch_clip.shape}") # Should be [B, 3, T, H, W]

    # # 2. Visualize a single sample from the batch (e.g., the first sample)
    # sample_idx = 0

    # # Get the key frame (the last frame of the clip)
    # # The clip shape is [C, T, H, W], so we grab the last frame (T-1)
    # clip = batch_clip[sample_idx]
    # key_frame = clip[:, -1, :, :] 

    # # Convert the key frame tensor to a NumPy array and permute dimensions for plotting
    # # PyTorch: [C, H, W] -> NumPy: [H, W, C]
    # key_frame_np = key_frame.permute(1, 2, 0).numpy()

    # # Denormalize the image for display
    # # Albumentations transforms normalize, so we need to reverse it
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # key_frame_np = key_frame_np * std + mean
    # key_frame_np = np.clip(key_frame_np * 255, 0, 255).astype(np.uint8)

    # # Get the bounding boxes and labels for this sample
    # boxes = batch_bboxes[sample_idx]
    # labels = batch_labels[sample_idx]

    # # 3. Plot the image and draw the bounding boxes
    # plt.figure(figsize=(10, 10))
    # ax = plt.gca()
    # ax.imshow(key_frame_np)

    # # Draw each bounding box on the image
    # for box, label in zip(boxes, labels):
    #     x1, y1, x2, y2 = box.numpy()
        
    #     # Create a rectangle patch
    #     rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
    #                         linewidth=2, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
        
    #     # Add the label
    #     ax.text(x1, y1, f'Label: {label.item()}', 
    #             bbox=dict(facecolor='yellow', alpha=0.5),
    #             fontsize=8, color='black')

    # plt.title("Sample from DataLoader")
    # plt.axis('off')
    # plt.show()


    # build model
    model = build_yowov3(config)
    get_info(config, model)
    model.to("cuda")
    model.train()
    
    # build loss algorithm
    criterion = build_loss(model, config)
    #####################################################

    #optimizer  = optim.AdamW(params=model.parameters(), lr= config['lr'], weight_decay=config['weight_decay'])

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  
    optimizer.add_param_group({"params": g[2], "weight_decay": 0.0}) 
    
    warmup_lr  = LinearWarmup(config)

    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad          = config['acc_grad'] 
    max_epoch         = config['max_epoch'] 
    lr_decay          = config['lr_decay']
    save_folder       = config['save_folder']
    
    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    loss_acc = 0.0
    ema = EMA(model)

    while(cur_epoch <= max_epoch):
        cnt_pram_update = 0
        p_bar = enumerate(dataloader)
        print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
        p_bar = tqdm.tqdm(p_bar, total=num_steps)
        for iteration, (batch_clip, batch_bboxes, batch_labels) in p_bar: 

            batch_size   = batch_clip.shape[0]
            batch_clip   = batch_clip.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")

            # test = list(zip(batch_bboxes, batch_labels))
            # print(test[0])

            outputs = model(batch_clip)

            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                if nbox == 0:
                    continue
                nclass = labels.shape[1]
                target = torch.Tensor(nbox, 5 + nclass)
                target[:, 0] = i
                target[:, 1:5] = bboxes
                target[:, 5:] = labels
                targets.append(target)
                # try:
                # except IndexError:
                #     a = (bboxes, labels)
                #     print("IndexError. Here is the error result")
                #     print(a)
                #     raise IndexError
                
            targets = torch.cat(targets, dim=0)

            # loss function
            loss = criterion(outputs, targets) / acc_grad
            loss_acc += loss.item()
            loss.backward()
            #plot_grad_flow(model.named_parameters()) #model too large, can't see anything!
            #plt.show()

            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update = cnt_pram_update + 1
                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_pram_update)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

                # progress bar
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 1) % (f'{cur_epoch}/{max_epoch}', memory, loss_acc)
                                                    #    avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                p_bar.set_description(s)

                # print("epoch : {}, update : {}, loss = {}".format(cur_epoch,  cnt_pram_update, loss_acc), flush=True)
                with open(os.path.join(config['save_folder'], "logging_new.txt"), "w") as f:
                    f.write("epoch : {}, update : {}, loss = {}".format(cur_epoch,  cnt_pram_update, loss_acc))

                loss_acc = 0.0
                #if cnt_pram_update % 500 == 0:
                    #torch.save(model.state_dict(), r"/home/manh/Projects/My-YOWO/weights/model_checkpoint/epch_{}_update_".format(cur_epoch) + str(cnt_pram_update) + ".pth")

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= lr_decay
        
        #          model.state_dict()
        save_path_ema = os.path.join(save_folder, "ema_epoch_" + str(cur_epoch) + ".pth")
        torch.save(ema.ema.state_dict(), save_path_ema)

        save_path     = os.path.join(save_folder, "epoch_"     + str(cur_epoch) + ".pth")
        torch.save(model.state_dict(), save_path)

        print("Saved model at epoch : {}".format(cur_epoch), flush=True)

        #log_path = '/home/manh/Projects/YOLO2Stream/training.log'
        #map50, mean_ap = call_eval(save_path)
        #logging.basicConfig(filename=log_path, level=logging.INFO)
        #logging.info('mAP 0.5 : {}, mAP : {}'.format(map50, mean_ap))

        cur_epoch += 1

if __name__ == "__main__":
    config = build_config()
    train_model(config)   