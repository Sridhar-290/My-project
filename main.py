import streamlit as st
import cv2
import torch
import numpy as np
from models import load_models
from detection import FaceDetector
from temporal import TemporalAnalyzer, ChallengeManager
from engine.rppg import LiveRPPG
import utils
import config
import time
import pandas as pd
from PIL import Image

import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GuardianAI | Deepfake Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .verdict-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .verdict-real {
        background: rgba(34, 197, 94, 0.2);
        border: 2px solid #22c55e;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
    }
    .verdict-fake {
        background: rgba(239, 68, 68, 0.2);
        border: 2px solid #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    .challenge-text {
        font-size: 24px;
        font-weight: bold;
        color: #fbbf24;
        text-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR & CONFIG ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.title("GuardianAI Core")
    st.divider()
    
    device_status = utils.check_gpu()
    st.info(f"🚀 {device_status}")
    
    st.subheader("Security Settings")
    sensitivity = st.slider("Detection Sensitivity", 0.0, 1.0, config.THRESHOLD)
    enable_rppg = st.toggle("Enable Biological Pulse (rPPG)", True)
    enable_challenges = st.toggle("Interactive Liveness Challenges", True)
    
    st.divider()
    st.subheader("System Logs")
    if st.button("Clear Detection Logs"):
        if os.path.exists("detection_log.csv"):
            os.remove("detection_log.csv")
            st.success("Logs cleared")

# --- INITIALIZATION ---
@st.cache_resource
def init_pipeline():
    device = config.DEVICE
    model = load_models(device)
    detector = FaceDetector(device)
    analyzer = TemporalAnalyzer()
    rppg = LiveRPPG()
    challenge = ChallengeManager()
    return model, detector, analyzer, rppg, challenge

model, detector, analyzer, rppg_engine, challenge_manager = init_pipeline()

# --- MAIN UI ---
st.title("🛡️ GuardianAI | Advanced Deepfake Fraud Detector")
st.caption("Live forensic analysis and biological liveness verification system.")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Active Forensic Scan")
    video_placeholder = st.empty()
    challenge_placeholder = st.empty()

with col2:
    st.subheader("Neural Intelligence Hub")
    verdict_placeholder = st.empty()
    
    m1, m2 = st.columns(2)
    with m1:
        pulse_metric = st.empty()
    with m2:
        blink_metric = st.empty()
        
    st.divider()
    st.subheader("Confidence Timeline")
    confidence_chart = st.empty()
    
    st.subheader("Challenge Status")
    challenge_status = st.empty()

# --- PROCESSING LOOP ---
history = []
run = st.checkbox("ACTIVATE SYSTEM MONITORING", value=False)

if run:
    cap = cv2.VideoCapture(config.WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera access failed.")
                break
                
            # 1. Detection & Classification
            # NOTE: detector.extract_faces returns (faces, bboxes, quality_scores)
            faces, bboxes, q_scores = detector.extract_faces(frame)
            
            ensemble_prob = 0.5 # Neutral if no face
            current_face_roi = None
            
            if faces:
                # Process the primary (first) face detected
                face_tensor = faces[0]
                with torch.no_grad():
                    ensemble_prob = model(face_tensor).item()
                
                # Apply quality penalty/bonus
                ensemble_prob = min(1.0, ensemble_prob + q_scores[0])
                
                # Extract ROI for rPPG
                x, y, w, h = bboxes[0]
                current_face_roi = frame[y:y+h, x:x+w]
            
            # 2. Temporal Analysis (Blink & Smoothing)
            smoothed_prob, blinked, landmarks = analyzer.process(frame, ensemble_prob)
            
            # 3. Biological Pulse Analysis (rPPG)
            pulse_data = {"bpm": 0, "conf": 0}
            if enable_rppg and current_face_roi is not None:
                updated_pulse = rppg_engine.update(current_face_roi)
                if updated_pulse:
                    pulse_data = updated_pulse
            
            # 4. Interactive Liveness Challenges
            challenge_msg = "STABLE"
            challenge_color = (255, 255, 255)
            if enable_challenges and landmarks:
                challenge_msg, challenge_color = challenge_manager.update(
                    landmarks, frame.shape[1], frame.shape[0]
                )
            
            # 5. Composite Decision Fusion
            # We trust rPPG and Challenges more for liveness than just the CNN
            final_fake_score = smoothed_prob
            
            # Penalty for lack of liveness (after initial warmup)
            if not blinked and len(history) > 100: 
                final_fake_score = max(final_fake_score, 0.6)
                
            if enable_rppg and pulse_data and pulse_data.get("conf", 0) > 0.4:
                # If we detect a biological pulse, reduce fake probability significantly
                final_fake_score *= 0.5
            
            # Challenge Authentication
            if challenge_manager.verified:
                final_fake_score *= 0.2 # Strongly suggests REAL
            
            # Map to verdict
            verdict = "ARTIFICIAL" if final_fake_score > sensitivity else "REAL HUMAN"
            confidence = final_fake_score if verdict == "ARTIFICIAL" else (1 - final_fake_score)
            
            # 6. UI Updates
            # Draw results on frame
            frame_display = detector.draw_results(frame, bboxes, [final_fake_score] * len(bboxes), threshold=sensitivity)
            video_placeholder.image(frame_display, channels="BGR", use_container_width=True)
            
            # Challenge UI
            challenge_placeholder.markdown(f"<div class='challenge-text' style='text-align: center;'>{challenge_msg}</div>", unsafe_allow_html=True)
            
            # Verdict Box
            is_artif = verdict == "ARTIFICIAL"
            v_class = "verdict-fake" if is_artif else "verdict-real"
            v_color = "#ef4444" if is_artif else "#22c55e"
            verdict_placeholder.markdown(f"""
                <div class='verdict-box {v_class}'>
                    <h1 style='color: {v_color}; margin: 0;'>{verdict}</h1>
                    <p style='color: white; opacity: 0.8;'>Confidence: {confidence*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            pulse_metric.metric("Heart Rate (BPM)", f"{pulse_data.get('bpm', 0)}", f"{pulse_data.get('conf', 0)*100:.0f}% Conf")
            blink_icon = "✅" if blinked else "❌"
            blink_metric.metric("Blink Detected", blink_icon)
            
            # Chart
            history.append(final_fake_score)
            if len(history) > 100: history.pop(0)
            confidence_chart.line_chart(pd.DataFrame(history, columns=["Fraud Score"]), height=150)
            
            challenge_status.write(f"Verification Stage: {challenge_manager.current_idx + 1}/{len(challenge_manager.challenges)}")
            
            # Alerts & Logging
            if verdict == "FAKE" and confidence > 0.9:
                utils.play_alert()
                utils.log_detection(verdict, final_fake_score)
            
            # Performance Throttle
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"System Error: {e}")
    finally:
        cap.release()
else:
    st.info("System Standby. Activate Monitoring to begin forensic analysis.")
