import torch
import os
from torch.utils.flop_counter import FlopCounterMode
from utils.build_config import build_config
from model.TSN.YOWOv3 import build_yowov3

def get_clean_name(name):
    """Maps code variable names to readable report names."""
    name_map = {
        "net2D": "Backbone 2D",
        "net3D": "Backbone 3D",
        "decoupled_head": "Decoupled Head",
        "fusion": "Fusion Module",
        "detection_head": "Detection Head (DFL)"
    }
    return name_map.get(name, name)

def analyze_and_save_yowo(config, filename="yowo_spec.txt"):
    # 1. Build Model & Setup Device
    print("[INFO] Building YOWOv3 Model...")
    model = build_yowov3(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Crucial for static analysis

    # 2. Create Dummy Input (Standard Batch Size = 1 for benchmarking)
    # Shape: [Batch, Channels, Clip_Length, Height, Width]
    dummy_input = torch.randn(
        1, 3, config['clip_length'], config['img_size'], config['img_size']
    ).to(device)

    # 3. Prepare Logging
    lines = []
    def log(text):
        print(text)
        lines.append(text)

    log(f"===================================================================")
    log(f"MODEL SPECIFICATION REPORT: YOWOv3")
    log(f"===================================================================")
    log(f"Input Shape: {dummy_input.shape}")
    log(f"Device: {device.upper()}")
    log(f"Mode: {config.get('mode', 'unknown')}")
    log(f"-------------------------------------------------------------------")

    # 4. Run FLOP Analysis (The Heavy Lifting)
    flop_counts = {}
    try:
        # display=False prevents it from printing to console immediately
        with FlopCounterMode(display=False) as mode:
            # Run the forward pass
            model(dummy_input)
            
            # Get the raw dictionary of {module_instance: flops}
            flop_counts = mode.get_flop_counts()
            
    except Exception as e:
        log(f"[ERROR] FLOP counting failed: {e}")

    # 5. Aggregation & Formatting
    log(f"{'BLOCK / MODULE':<25} | {'PARAMS (M)':<12} | {'FLOPS (G)':<12} | {'RAW NAME'}")
    log("-" * 75)

    total_params = 0
    total_flops_accumulated = 0.0
    
    # Loop through the high-level blocks (net2D, net3D, fusion, etc.)
    for name, child in model.named_children():
        
        # A. Count Parameters for this block
        num_params = sum(p.numel() for p in child.parameters())
        params_m = num_params / 1e6
        total_params += num_params
        
        # B. Count FLOPs for this block (Recursive Sum)
        # This fixes the "Zero FLOPs" issue. 
        # We look at every single layer INSIDE this block.
        child_flops = 0
        for m in child.modules():
            # If this specific layer (e.g., Conv2d) did work, add it.
            if m in flop_counts:
                child_flops += sum(flop_counts[m].values())
        
        flops_g = child_flops / 1e9
        total_flops_accumulated += flops_g
        
        # C. Formatting
        clean_name = get_clean_name(name)
        log(f"{clean_name:<25} | {params_m:<12.3f} | {flops_g:<12.3f} | {name}")

    log("-" * 75)
    log(f"{'TOTAL':<25} | {total_params/1e6:<12.3f} M   | {total_flops_accumulated:<12.3f} G   |")
    log(f"===================================================================")
    
    # 6. Save to File
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[SUCCESS] Report saved to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"[ERROR] Could not save file: {e}")

if __name__ == "__main__":
    config = build_config()
    analyze_and_save_yowo(config)