import torch

# General Settings
THRESHOLD = 0.85  # Higher threshold for extreme accuracy
BATCH_SIZE = 1
FRAME_WINDOW = 30 # Longer window for smoother, more stable verdicts
WEBCAM_ID = 0
IMG_SIZE = 224

# Evidence Capture
SAVE_EVIDENCE = True
EVIDENCE_DIR = "recordings"
SCREENSHOT_DIR = "recordings/screenshots"

# Biological Pulse
PULSE_THRESHOLD = 0.3 # Minimum confidence to be considered 'Biological'


# Model Paths (will be downloaded)
MODEL_DIR = "weights"
XCEPTION_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/xception-b5690688.pth" # Placeholder or general xception
# In practice, for FaceForensics++, weights are specific. 
# I will use a logic in utils.py to download if missing.

# Liveness Thresholds
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 3

# UI Settings
DYNAMIC_BOX_COLOR_REAL = (0, 255, 0)
DYNAMIC_BOX_COLOR_FAKE = (0, 0, 255)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
