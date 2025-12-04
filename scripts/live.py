
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
from torchvision.transforms import v2
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
import time

from cus_datasets.build_dataset import build_dataset
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from PIL import Image
from utils.flops import get_info

# --- HELPER FUNCTION FOR PROGRESS BAR ---
def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete',
                       decimals=1, length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
# ------------------------------------------

class live_transform():
    # """
    # Args:
    #     clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
    #     boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    # Return:
    #     clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
    #     boxes : not change
    # """
    """
    Transforms a NumPy frame (from cv2) into a normalized tensor
    for the model.
    """

    def __init__(self, img_size, convert_color=True):
        self.img_size = img_size

        # --- THIS IS THE NEW SWITCH ---
        self.convert_color = convert_color
        print(f"[Transform] Color conversion (BGR->RGB) set to: {self.convert_color}")
        # ------------------------------

        # The rest of the V2 pipeline
        self.transform_pipeline = v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        pass

    # def to_tensor(self, image):
    #     return FT.to_tensor(image)
    
    def normalize(self, clip, mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, frame):
        # W, H = img.size
        # img = img.resize([self.img_size, self.img_size])
        # img = self.to_tensor(img)
        # img = self.normalize(img)

        # return img
        # # 1. Resize using OpenCV
        # img = cv2.resize(frame, (self.img_size, self.img_size))
        # # 2. Convert BGR (OpenCV default) to RGB
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- USE THE SWITCH HERE ---
        if self.convert_color:
            # If True, convert from BGR to RGB for the model
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # If False, just pass the raw BGR frame
            img = frame
        # ---------------------------

        # 3. Convert to Tensor: HWC -> CHW
        # Transpose from (Height, Width, Channel) to (Channel, Height, Width)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))

        # 4. Scale from [0, 255] to [0.0, 1.0]
        # FT.to_tensor used to do this automatically, now we do it manually.
        # img_tensor = img_tensor.float() / 255.0
        # Move to CUDA *BEFORE* processing
        img_tensor = img_tensor.to("cuda")

        # 5. Normalize
        # img_tensor = self.normalize(img_tensor)
        # Apply the V2 pipeline on the GPU
        img_tensor = self.transform_pipeline(img_tensor)
        return img_tensor

def detect(config):

    # --- ADD THIS SWITCH ---
    # True = Correct colors for model (model will be accurate)
    # False = Incorrect colors for model (model will fail)
    CONVERT_COLOR_FOR_MODEL = True
    # -----------------------
    
    # --- CONFIGURATION SWITCHES ---
    # Set this to True to run a timed benchmark.
    # Set to False to run in "Free Mode" until you press 'q'.
    enable_time_limit = False

    # Set the duration (in seconds) ONLY if time limit is enabled.
    time_limit_seconds = 30.0
    # --------------------------------

    model = build_yowov3(config) 
    get_info(config, model)
    model.to("cuda")
    model.eval()
    mapping = config['idx2name']

    # Use CAP_DSHOW for better Windows compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    frame_list = []
    
    transform = live_transform(config['img_size'], convert_color=CONVERT_COLOR_FOR_MODEL)

    # --- STATS SETUP ---
    # For live display
    loop_fps = 0.0
    latency_ms = 0.0  # Will only update when inference runs
    
    # For final report
    total_proc_time = 0.0
    total_inferences = 0
    
    loop_start_time = time.time()
    prev_loop_time = loop_start_time
    exit_reason = "User quit." # Default exit reason
    
    print("--- Starting Live Session ---")
    if enable_time_limit:
        print(f"Mode: Time Limit ({time_limit_seconds}s)")
        print("NOTE: Terminal will show progress, CV2 window shows live stats.")
        print_progress_bar(0, time_limit_seconds, prefix='Time:')
    else:
        print("Mode: Free (Press 'q' in the CV2 window to quit)")
    # --------------------------------

    while True:
        # --- OVERALL LOOP FPS (START) ---
        # This measures the speed of the entire while loop
        curr_time = time.time()
        loop_time = curr_time - prev_loop_time
        prev_loop_time = curr_time
        if loop_time > 0:
            loop_fps = 1.0 / loop_time
        # --- (END) ---
        
        # --- TIME LIMIT CHECK & PROGRESS BAR ---
        if enable_time_limit:
            elapsed_time = curr_time - loop_start_time
            if elapsed_time > time_limit_seconds:
                exit_reason = "Time limit reached."
                break # Exit the while loop
            
            # Update terminal progress bar
            print_progress_bar(elapsed_time, time_limit_seconds, prefix='Time:')
        # -----------------------------------------

        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame, skipping...")
            continue

        # Pass the raw cv2 frame
        frame_list.append(transform(frame))
        
        if (len(frame_list) > 16):
            frame_list.pop(0)

        # We must resize the original frame for display *every* loop
        origin_image = cv2.resize(frame, (config['img_size'], config['img_size']))

        # --- INFERENCE BLOCK ---
        # This only runs when the buffer is full
        if (len(frame_list) == 16):
            
            # 1. Measure Latency
            start_proc_time = time.time()

            clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous()
            clip = clip.unsqueeze(0).to("cuda")

            with torch.no_grad():
                outputs = model(clip)
            
            outputs = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)[0]

            end_proc_time = time.time()
            proc_time = end_proc_time - start_proc_time
            
            # 2. Update Live Latency
            latency_ms = proc_time * 1000
            
            # 3. Accumulate for Final Report
            total_proc_time += proc_time
            total_inferences += 1
            
            # 4. Draw boxes
            draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)
        
        # --- DRAW LIVE STATS ---
        # Display last known Inference Latency
        # This value will only "update" every 16 frames
        latency_text = f"Latency: {latency_ms:.2f} ms"
        cv2.putText(origin_image, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display Loop FPS (will be high and variable)
        fps_text = f"Loop FPS: {loop_fps:.2f}"
        cv2.putText(origin_image, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # -------------------------

        cv2.imshow('img', origin_image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # --- (END OF LOOP) ---
    
    # --- FINAL BENCHMARK REPORT ---
    if enable_time_limit:
        print_progress_bar(time_limit_seconds, time_limit_seconds, prefix='Time:') # Fill bar
    
    print(f"\n--- Session Finished ({exit_reason}) ---")
    
    # This report runs for ALL modes and ALL exit reasons
    if total_inferences > 0:
        # Calculate averages based on accumulated stats
        avg_latency_s = total_proc_time / total_inferences
        avg_latency_ms = avg_latency_s * 1000
        
        # "Average FPS" is the inverse of the average *processing* time
        avg_inference_fps = total_inferences / total_proc_time
        
        print(f"Total session time: {time.time() - loop_start_time:.2f} seconds")
        print(f"Total inferences: {total_inferences}")
        print(f"Total processing-only time: {total_proc_time:.2f} seconds")
        print("---------------------------------------")
        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Average FPS (Inference): {avg_inference_fps:.2f}")
    else:
        print("No inferences were completed.")
    # -----------------------------------
    
    cap.release()
    cv2.destroyAllWindows()
         

if __name__ == "__main__":
    config = build_config()
    detect(config)