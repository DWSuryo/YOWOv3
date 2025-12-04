
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

from cus_datasets.build_dataset import build_dataset
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3 
from utils.build_config import build_config
from utils.flops import get_info

# --- HELPER FUNCTION FOR PROGRESS BAR ---
def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete',
                       decimals=1, length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # \r is a "carriage return" - it moves the cursor to the start of the line
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
# ------------------------------------------

def detect(config):

    # --- CONFIGURATION SWITCHES ---
    # Set this to True to run a timed benchmark.
    # Set to False to run in "Free Mode" on the entire dataset.
    enable_time_limit = True

    # Set the duration (in seconds) ONLY if time limit is enabled.
    time_limit_seconds = 30.0

    # Set this to True to see the (fast) video display.
    # Set to False for a "pure" benchmark (no display I/O).
    enable_display = True
    # --------------------------------

    #########################################################################
    dataset = build_dataset(config, phase='test')
    model = build_yowov3(config) 
    get_info(config, model)
    ##########################################################################
    mapping = config['idx2name']
    model.to("cuda")
    model.eval()

    # --- BENCHMARKING SETUP ---
    total_proc_time = 0.0
    total_inferences = 0
    loop_start_time = time.time()
    exit_reason = "Dataset finished." # Default exit reason
    
    print("--- Starting Benchmark ---")
    if enable_time_limit:
        print(f"Mode: Time Limit ({time_limit_seconds}s)")
        if enable_display:
            print("Display: ON (Benchmark will be slightly slower)")
        else:
            print("Display: OFF (Pure benchmark)")
            print_progress_bar(0, time_limit_seconds, prefix='Time:')
    else:
        print("Mode: Free (Full Dataset)")
        if enable_display:
            print("Display: ON (Press 'q' to quit)")
        else:
            print("Display: OFF (Processing full dataset...)")
    # --------------------------------

    for idx in range(dataset.__len__()):
        
        # --- TIME LIMIT CHECK & PROGRESS BAR ---
        if enable_time_limit:
            elapsed_time = time.time() - loop_start_time
            if elapsed_time > time_limit_seconds:
                exit_reason = "Time limit reached."
                break # Exit the for-loop
            
            # Update progress bar (only update if display is off,
            # to avoid mixing with display metrics)
            if not enable_display:
                # We update the progress bar here
                print_progress_bar(elapsed_time, time_limit_seconds, prefix='Time:')
        # -----------------------------------------

        origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)
        clip = clip.unsqueeze(0).to("cuda")

        # --- Time the inference block ---
        start_time = time.time()

        with torch.no_grad():
            outputs = model(clip)
        
        outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]

        end_time = time.time()
        # -------------------------------

        proc_time = end_time - start_time

        # --- BENCHMARKING ACCUMULATION ---
        total_proc_time += proc_time
        total_inferences += 1
        # -----------------------------------

        # --- VISUALIZATION BLOCK ---
        if enable_display:
            latency_ms = proc_time * 1000
            inference_fps = 1.0 / proc_time if proc_time > 0 else 0.0
            
            origin_image = cv2.resize(origin_image, (config['img_size'], config['img_size']))
            draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)

            # Draw text
            latency_text = f"Latency: {latency_ms:.2f} ms"
            cv2.putText(origin_image, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            fps_text = f"Inference FPS: {inference_fps:.2f}"
            cv2.putText(origin_image, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('img', origin_image)
            k = cv2.waitKey(1)
            
            if k == ord('q'):
                exit_reason = "User quit."
                break
        # ---------------------------

    # --- BENCHMARKING RESULTS (AFTER LOOP) ---
    if not enable_display and enable_time_limit:
        print_progress_bar(time_limit_seconds, time_limit_seconds, prefix='Time:') # Fill bar
    
    print(f"\n--- Benchmark Finished ({exit_reason}) ---")
    
    # This report runs for ALL modes and ALL exit reasons
    if total_inferences > 0:
        avg_latency_s = total_proc_time / total_inferences
        avg_latency_ms = avg_latency_s * 1000
        avg_inference_fps = total_inferences / total_proc_time
        
        print(f"Total time in loop: {time.time() - loop_start_time:.2f} seconds")
        print(f"Total inferences: {total_inferences}")
        print(f"Total processing-only time: {total_proc_time:.2f} seconds")
        print("---------------------------------------")
        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Average FPS: {avg_inference_fps:.2f}")
    else:
        print("No inferences were completed.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = build_config()
    detect(config)