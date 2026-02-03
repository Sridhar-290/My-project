import subprocess
import sys
import os

def install_deps():
    print("Installing dependencies...")
    packages = [
        "opencv-python",
        "torch",
        "torchvision",
        "facenet-pytorch",
        "mtcnn",
        "streamlit",
        "playsound",
        "numpy",
        "pillow",
        "timm",
        "mediapipe",
        "pandas",
        "requests",
        "tqdm"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def create_dirs():
    if not os.path.exists("weights"):
        os.makedirs("weights")
    if not os.path.exists("recordings"):
        os.makedirs("recordings")

if __name__ == "__main__":
    create_dirs()
    try:
        install_deps()
        print("\nSetup Complete!")
        print("Run the application with: streamlit run main.py")
    except Exception as e:
        print(f"Error during setup: {e}")
