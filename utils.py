import os
import requests
from tqdm import tqdm
import config
from playsound import playsound
import threading
import logging
import csv
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, filename='detection_log.csv', format='%(message)s')

def log_detection(verdict, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('detection_log.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, verdict, f"{confidence:.4f}"])

def play_alert():
    def _play():
        try:
            # You would need a real alert.mp3 in the folder
            if os.path.exists("alert.mp3"):
                playsound("alert.mp3")
        except:
            pass
    threading.Thread(target=_play).start()

def download_weights():
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    
    # In a real scenario, we provide URLs for MesoNet or Xception weights
    # Since specific FaceForensics weights are huge/private, 
    # we'll assume the user might have them or we download general pretrained ones.
    pass

def check_gpu():
    import torch
    if torch.cuda.is_available():
        return f"GPU Detected: {torch.cuda.get_device_name(0)}"
    return "Running on CPU (No GPU found)"
