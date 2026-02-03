import cv2
import mediapipe as mp
import numpy as np
import time

class FakeFaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_count = 0
        self.blink_state = False # False = open, True = closed
        self.frames_processed = 0
        self.start_time = time.time()
        self.jitter_history = []
        
    def calculate_ear(self, eye_landmarks):
        # Ear calculation for eye liveness (Eye Aspect Ratio)
        # 1-5, 2-4 vertical, 0-3 horizontal
        p1 = eye_landmarks[1]
        p5 = eye_landmarks[5]
        p2 = eye_landmarks[2]
        p4 = eye_landmarks[4]
        p0 = eye_landmarks[0]
        p3 = eye_landmarks[3]
        
        dist_v1 = np.linalg.norm(np.array(p1) - np.array(p5))
        dist_v2 = np.linalg.norm(np.array(p2) - np.array(p4))
        dist_h = np.linalg.norm(np.array(p0) - np.array(p3))
        
        ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
        return ear

    def analyze_frame(self, frame):
        self.frames_processed += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        verdict = "Processing..."
        confidence = 0.0
        indicators = {
            "liveness": 0.0,
            "texture": 0.0,
            "stability": 0.0,
            "blink_detected": self.blink_count > 0
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            ih, iw, _ = frame.shape
            
            # 1. Liveness: Check for Blinking
            # Right eye indices: [33, 160, 158, 133, 153, 144]
            # Left eye indices: [362, 385, 387, 263, 373, 380]
            re = [33, 160, 158, 133, 153, 144]
            le = [362, 385, 387, 263, 373, 380]
            
            re_pts = [(face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih) for i in re]
            le_pts = [(face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih) for i in le]
            
            ear_r = self.calculate_ear(re_pts)
            ear_l = self.calculate_ear(le_pts)
            avg_ear = (ear_r + ear_l) / 2.0
            
            if avg_ear < 0.2: # Eye closed threshold
                if not self.blink_state:
                    self.blink_state = True
            else:
                if self.blink_state:
                    self.blink_count += 1
                    self.blink_state = False
            
            indicators["liveness"] = min(1.0, (self.blink_count / max(1, (time.time() - self.start_time) / 10.0)))

            # 2. Texture: Check for blur (common in deepfakes)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            indicators["texture"] = min(1.0, laplacian_var / 500.0) # Arbitrary threshold

            # 3. Stability: Landmark Jitter
            # Deepfakes often have jitter at face edges
            nose_tip = face_landmarks.landmark[1]
            self.jitter_history.append((nose_tip.x, nose_tip.y))
            if len(self.jitter_history) > 30:
                self.jitter_history.pop(0)
                std_dev = np.std(self.jitter_history, axis=0)
                jitter_score = np.sum(std_dev)
                indicators["stability"] = 1.0 - min(1.0, jitter_score * 10.0)
            else:
                indicators["stability"] = 0.5

            # Calculate Final Confidence
            # Fake Detection Logic: 
            # High stability + Low texture (blur) + No blinks = High probability of Fake
            # We want Confidence to represent "Realness"
            realness_score = (indicators["liveness"] * 0.4 + 
                             indicators["texture"] * 0.3 + 
                             indicators["stability"] * 0.3)
            
            confidence = realness_score
            if confidence > 0.7:
                verdict = "REAL"
            elif confidence > 0.4:
                verdict = "SUSPICIOUS"
            else:
                verdict = "FAKE"
        else:
            verdict = "NO FACE DETECTED"
            
        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 2),
            "indicators": indicators,
            "blink_count": self.blink_count
        }
