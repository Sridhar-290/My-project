import cv2
import torch
import numpy as np
import os
import time
from datetime import datetime
from models import load_models
from detection import FaceDetector
from temporal import TemporalAnalyzer, ChallengeManager
from engine.rppg import LiveRPPG
import config

try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    HAS_MESH = True
except Exception:
    HAS_MESH = False

class ShadowAnalyzer:
    @staticmethod
    def analyze_shadows(face_roi):
        if face_roi is None or face_roi.size == 0: return 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        # Calculate mean brightness difference
        diff = abs(np.mean(left_half) - np.mean(right_half))
        # Deepfakes often have unnaturally balanced lighting or extreme mismatches
        # Return a 'naturalness' score - extremely high diff is suspicious
        return min(1.0, diff / 50.0)

class EvidenceCapturer:
    def __init__(self):
        self.last_capture = 0
        if not os.path.exists(config.SCREENSHOT_DIR):
            os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    def capture(self, frame, verdict):
        now = time.time()
        if now - self.last_capture > 5: # Limit to one capture every 5 seconds
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.SCREENSHOT_DIR}/{verdict}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f">>> EVIDENCE CAPTURED: {filename}")
            self.last_capture = now

def run_live_detection():
    print("--------------------------------------------------")
    print("GUARDIAN AI PRO | BIOMETRIC FRAUD DETECTOR")
    print("--------------------------------------------------")
    print("Initializing Advanced Subsystems...")
    
    device = config.DEVICE
    try:
        model = load_models(device)
        detector = FaceDetector(device)
        analyzer = TemporalAnalyzer()
        rppg = LiveRPPG()
        capturer = EvidenceCapturer()
        challenge = ChallengeManager() # Active Liveness Challenge
        
        if HAS_MESH:
            face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
            print(">>> 3D Facial Mesh Active")
        
        print(f"Models loaded successfully on: {device}")
    except Exception as e:
        print(f"Error loading subsystems: {e}")
        return

    cap = None
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened(): break
        cap.release()
        cap = None

    if cap is None:
        print("CRITICAL ERROR: No webcam detected.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n>>> LIVE FEED ACTIVE - BIOMETRIC MONITORING")
    print(">>> PRESS 'Q' TO STOP AND EXIT\n")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
            
        display_frame = frame.copy()
        faces, bboxes, q_scores = detector.extract_faces(frame)
        
        if faces:
            areas = [w * h for (x, y, w, h) in bboxes]
            primary_idx = areas.index(max(areas))
            
            for i, (face_tensor, (x, y, w, h)) in enumerate(zip(faces, bboxes)):
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0: continue

                with torch.no_grad():
                    texture_prob = model(face_tensor).item() + q_scores[i]
                
                # 1. Biological Pulse Detection (Only Primary)
                bpm, pulse_conf = (0, 0)
                challenge_status, challenge_color = "WAITING...", (255, 255, 255)
                
                if i == primary_idx:
                    vitals = rppg.update(face_roi)
                    if vitals:
                        bpm = vitals['bpm']
                        pulse_conf = vitals['conf']
                    else:
                        bpm, pulse_conf = 0, 0
                        
                    smoothed_prob, blinked = analyzer.process(face_roi, texture_prob)
                    
                    # 2. Update Active Challenge Logic
                    if HAS_MESH:
                        mesh_results = face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        if mesh_results.multi_face_landmarks:
                            landmarks = mesh_results.multi_face_landmarks[0]
                            challenge_status, challenge_color = challenge.update(landmarks, w, h)
                            
                            # Mesh Drawing
                            mp_drawing.draw_landmarks(
                                image=display_frame[y:y+h, x:x+w],
                                landmark_list=landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=drawing_spec)
                else:
                    smoothed_prob = texture_prob
                    blinked = False
                
                # 3. Final Verdict Integration
                is_ai = smoothed_prob > config.THRESHOLD
                is_spoof = not blinked and not analyzer.motion_detected
                
                if is_ai:
                    verdict, color = "AI GENERATED", (0, 0, 255)
                elif is_spoof and not challenge.verified:
                    verdict, color = "STATIC SPOOF", (0, 0, 255)
                elif challenge.verified:
                    verdict, color = "100% AUTHENTIC", (0, 255, 0)
                elif not blinked:
                    verdict, color = "REAL (WAITING LIVENESS)", (255, 120, 0)
                else:
                    verdict, color = "REAL HUMAN (VERIFIED)", (0, 255, 0)

                # 4. Automatic Evidence Capture
                if "AI" in verdict or "SPOOF" in verdict:
                    capturer.capture(frame, verdict.split()[0])

                # 5. Drawing Overlays
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # PROFESSIONAL HUD
                if i == primary_idx:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (380, 155), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    
                    cv2.putText(display_frame, f"STATUS: {verdict}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display_frame, f"ACTIVE VERIFICATION:", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(display_frame, f"> {challenge_status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, challenge_color, 2)
                    
                    cv2.putText(display_frame, f"PULSE: {bpm} BPM (Conf: {pulse_conf*100:.0f}%)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(display_frame, f"TEXTURE SCORE: {smoothed_prob*100:.1f}%", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    liveness_icon = "DETECTED" if blinked else "WAITING..."
                    cv2.putText(display_frame, f"PASIVE LIVENESS (BLINK): {liveness_icon}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('GuardianAI PRO | Advanced Biometric Security', display_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()
