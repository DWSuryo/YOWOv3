from scripts import train, ava_eval, ucf_eval, detect, live
import argparse
from utils.build_config import build_config
from datetime import datetime

if __name__ == "__main__":
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))
    parser = argparse.ArgumentParser(description="YOWOv3")

    parser.add_argument('-m', '--mode', type=str, help='train/eval/live/detect/onnx')
    parser.add_argument('-cf', '--config', type=str, help='path to config file')

    args = parser.parse_args()

    config = build_config(args.config)

    if args.mode == 'train':
        train.train_model(config=config)

    elif args.mode == 'eval':
        if config['dataset'] == 'ucf' or config['dataset'] == 'jhmdb' or config['dataset'] == 'ucfcrime':
            ucf_eval.eval(config=config)
        elif config['dataset'] == 'ava':
            ava_eval.eval(config=config)

    elif args.mode == 'detect':
        detect.detect(config=config)

    elif args.mode == 'live':
        live.detect(config=config)
    
    elif args.mode == 'onnx':
        onnx.export2onnx(config=config)

    time_end = datetime.now()
    print("Finished at Date and Time:", time_end.strftime("%Y-%m-%d %H:%M:%S"))
    time_duration = time_end - time_start
    # Format the duration as Days HH:MM:SS
    days = time_duration.days
    seconds = time_duration.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_duration = f"{days} Days {hours:02}:{minutes:02}:{seconds:02}"
    print(f"Code execution time: {formatted_duration}")