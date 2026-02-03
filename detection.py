import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
from torchvision import transforms
import config

class FaceDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.detector = MTCNN(keep_all=True, device=device)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def extract_faces(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img_full = Image.fromarray(rgb_frame)
        
        # 1. Quality Check: Detect if the input is too blurry (common in screen-recordings)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_low_quality = lap_var < 100 # Threshold for high-quality detection
        
        # Detect faces returning bounding boxes
        boxes, _ = self.detector.detect(pil_img_full)
        
        extracted_faces = []
        bboxes = []
        quality_scores = []
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Basic boundary check
                x1, y1 = max(0, x1), max(0, y1)
                
                face_img = rgb_frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                    
                # Convert to PIL for transforms
                pil_img = Image.fromarray(face_img)
                tensor_img = self.transform(pil_img).to(self.device).unsqueeze(0)
                
                extracted_faces.append(tensor_img)
                bboxes.append((x1, y1, w, h))
                # Add 0.1 bonus to fake probability if quality is suspiciously low
                quality_scores.append(0.15 if is_low_quality else 0.0)
            
        return extracted_faces, bboxes, quality_scores

    def draw_results(self, frame, bboxes, probabilities, threshold=0.5):
        for (x, y, w, h), prob in zip(bboxes, probabilities):
            is_fake = prob > threshold
            label = "ARTIFICIAL" if is_fake else "REAL HUMAN"
            # Show confidence in the decision
            confidence = prob if is_fake else (1.0 - prob)
            
            # Vibrant Colors (BGR)
            # Real = Bright Green, Fake = Bright Red
            color = (0, 255, 0) if not is_fake else (0, 0, 255)
            
            # Draw glow-like effect for high confidence
            thickness = 2
            if confidence > 0.9:
                thickness = 3
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, 1) # Outer glow
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Background for text
            text = f"{label} {confidence*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-th-15), (x+tw+10, y), color, -1)
            
            cv2.putText(frame, text, (x+5, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
