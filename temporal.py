import numpy as np
import config
from collections import deque
import cv2
import time

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, 'solutions')
except ImportError:
    HAS_MEDIAPIPE = False

class TemporalAnalyzer:
    def __init__(self):
        self.window = deque(maxlen=config.FRAME_WINDOW)
        self.blink_frames = 0
        self.is_blinked = False
        
        # Motion detection fallback
        self.prev_face_crop = None
        self.motion_detected = False
        
        if HAS_MEDIAPIPE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
            except Exception:
                self.face_mesh = None
        else:
            self.face_mesh = None

    def calculate_ear(self, landmarks, img_w, img_h):
        if not landmarks: return 0.3
        
        def get_ear(indices):
            pts = []
            for i in indices:
                l = landmarks.landmark[i]
                pts.append(np.array([l.x * img_w, l.y * img_h]))
            
            v1_v5 = np.linalg.norm(pts[1] - pts[5])
            v2_v4 = np.linalg.norm(pts[2] - pts[4])
            h0_h3 = np.linalg.norm(pts[0] - pts[3])
            return (v1_v5 + v2_v4) / (2.0 * h0_h3 + 1e-6)

        lar = get_ear([362, 385, 387, 263, 373, 380])
        rar = get_ear([33, 160, 158, 133, 153, 144])
        return (lar + rar) / 2.0

    def process(self, frame, current_prob):
        # 1. Update smoothing window
        self.window.append(current_prob)
        smoothed_prob = sum(self.window) / len(self.window)
        
        # 2. Liveness Check
        h, w, _ = frame.shape
        if h < 10 or w < 10: return smoothed_prob, self.is_blinked
        
        # Fallback Motion Detection (Standardized size to prevent crashes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (200, 200)) # Fixed size for comparison
        
        if self.prev_face_crop is not None:
            # Simple diff for motion
            diff = cv2.absdiff(gray_resized, self.prev_face_crop)
            if np.mean(diff) > 2.0: 
                self.motion_detected = True
        self.prev_face_crop = gray_resized

        # Try Mediapipe Blink Detection
        self.last_landmarks = None
        if self.face_mesh:
            try:
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    self.last_landmarks = results.multi_face_landmarks[0]
                    ear = self.calculate_ear(self.last_landmarks, w, h)
                    
                    if ear < config.EAR_THRESHOLD:
                        self.blink_frames += 1
                    else:
                        if self.blink_frames >= config.EAR_CONSEC_FRAMES:
                            self.is_blinked = True 
                        self.blink_frames = 0
            except Exception:
                pass 
        else:
            self.is_blinked = self.motion_detected

        return smoothed_prob, self.is_blinked, self.last_landmarks
class ChallengeManager:
    def __init__(self):
        self.challenges = ["TURN LEFT", "TURN RIGHT", "LOOK UP"]
        self.current_idx = 0
        self.verified = False
        self.start_time = time.time()
        self.last_status = "PENDING"

    def get_pose(self, landmarks, img_w, img_h):
        if not landmarks: return 0, 0
        
        # Nose tip, Chin, Left eye, Right eye, Left mouth, Right mouth
        # Indices in FaceMesh: 1, 152, 33, 263, 61, 291
        nose_tip = landmarks.landmark[1]
        
        # Simple Yaw estimation: relative position of nose tip between eyes
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        
        # Calculate yaw: normalize nose position between eyes (-1.0 to 1.0)
        # Center should be roughly 0.0
        eye_center = (left_eye.x + right_eye.x) / 2
        eye_dist = right_eye.x - left_eye.x
        yaw = (nose_tip.x - eye_center) / (eye_dist + 1e-6)
        
        # Simple Pitch: nose relative to eyes vertical
        avg_eye_y = (left_eye.y + right_eye.y) / 2
        pitch = (nose_tip.y - avg_eye_y) / (eye_dist + 1e-6)
        
        return yaw, pitch

    def update(self, landmarks, img_w, img_h):
        if self.verified: return "VERIFIED", (0, 255, 0)
        
        yaw, pitch = self.get_pose(landmarks, img_w, img_h)
        target = self.challenges[self.current_idx]
        
        status = f"CHALLENGE: {target}"
        color = (255, 255, 255)
        
        # Detect Completion
        success = False
        if target == "TURN LEFT" and yaw < -0.4: success = True
        elif target == "TURN RIGHT" and yaw > 0.4: success = True
        elif target == "LOOK UP" and pitch < -0.2: success = True
        
        if success:
            self.current_idx += 1
            if self.current_idx >= len(self.challenges):
                self.verified = True
                return "100% AUTHENTIC", (0, 255, 0)
            else:
                self.start_time = time.time() # Reset timer for next part
                
        # Timeout check (optional)
        if time.time() - self.start_time > 10 and not self.verified:
            return "CHALLENGE TIMEOUT", (0, 0, 255)
            
        return status, color
