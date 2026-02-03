#!/usr/bin/env python3
"""
Main CLI entry point for the offline fake face detector.

Usage:
    python fake_face_detector.py --input <image_or_video>

This script orchestrates rPPG (video), GAN fingerprint (image/frame), CNN-based proxy detector
using a pretrained EfficientNet_B0 (as a feature extractor/proxy), and lightweight forensic checks.
It fuses scores into a calibrated fraud probability and prints verdict, probability, risk, and explanation.
"""
import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

from rppg import estimate_rppg
from gan_fingerprint import detect_gan_fingerprint
from cnn_detector import cnn_detect
from forensic_checks import run_forensic_checks


def fuse_scores(scores, weights):
    # scores: dict of name->value (0..1), None values treated as neutral (0.5)
    vals = []
    ws = []
    for k, w in weights.items():
        v = scores.get(k, None)
        if v is None:
            v = 0.5
        vals.append(v)
        ws.append(w)
    vals = np.array(vals, dtype=float)
    ws = np.array(ws, dtype=float)
    if ws.sum() == 0:
        fused = float(vals.mean())
    else:
        fused = float((vals * ws).sum() / ws.sum())
    # simple calibration (sigmoid-ish) to push extreme values
    calibrated = 1.0 / (1.0 + np.exp(-4 * (fused - 0.5)))
    return calibrated


def map_verdict(prob):
    if prob >= 0.65:
        return "Fake"
    if prob >= 0.45:
        return "Suspicious"
    return "Real"


def map_risk(prob):
    if prob >= 0.75:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


def analyze_path(path: str, sample_frames: int = 8):
    p = Path(path)
    if not p.exists():
        print(f"Input not found: {path}")
        sys.exit(2)

    ext = p.suffix.lower()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    scores_collect = {"cnn": [], "gan": [], "forensic": []}
    rppg_score = None

    if is_video:
        print("Processing video (rPPG + frame sampling)...")
        # rPPG heartbeat liveness detection (video only)
        rppg_score, heartbeat_bpm = estimate_rppg(str(p), max_seconds=30)
        print(f"  rPPG heartbeat score: {rppg_score:.3f} bpm_estimate={heartbeat_bpm}")

        # sample frames across the video for per-frame analyses
        cap = cv2.VideoCapture(str(p))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # fallback: sample first N frames
            indices = list(range(sample_frames))
        else:
            indices = np.linspace(0, max(0, total - 1), min(sample_frames, total)).astype(int)

        frames = {}
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            frames[i] = frame
        cap.release()

        for i, frame in frames.items():
            cnn_s = cnn_detect(frame)
            gan_s = detect_gan_fingerprint(frame)
            f_s = run_forensic_checks(frame)
            scores_collect["cnn"].append(cnn_s)
            scores_collect["gan"].append(gan_s)
            scores_collect["forensic"].append(f_s)

    else:
        print("Processing image (no rPPG)...")
        img = cv2.imread(str(p))
        if img is None:
            print("Failed to read image.")
            sys.exit(2)
        scores_collect["cnn"].append(cnn_detect(img))
        scores_collect["gan"].append(detect_gan_fingerprint(img))
        scores_collect["forensic"].append(run_forensic_checks(img))

    # aggregate per-module scores (mean)
    agg = {}
    for k, arr in scores_collect.items():
        if len(arr) == 0:
            agg[k] = None
        else:
            # if an element is tuple (score, info) handle it
            vals = []
            for a in arr:
                if isinstance(a, tuple):
                    vals.append(a[0])
                else:
                    vals.append(float(a))
            agg[k] = float(np.mean(vals))

    # Include rppg as separate score
    agg["rppg"] = rppg_score

    # Weights (banking-grade): rPPG stronger for video when present
    weights = {"rppg": 0.4 if agg.get("rppg") is not None else 0.0,
               "cnn": 0.3,
               "gan": 0.2,
               "forensic": 0.1}

    fused = fuse_scores(agg, weights)

    verdict = map_verdict(fused)
    risk = map_risk(fused)

    # Build short explanation
    reasons = []
    if agg.get("rppg") is None:
        reasons.append("rPPG not available (single image)")
    else:
        reasons.append(f"rPPG heartbeat_score={agg['rppg']:.2f}")
    reasons.append(f"CNN_score={agg.get('cnn', 0.5):.2f}")
    reasons.append(f"GAN_score={agg.get('gan', 0.5):.2f}")
    reasons.append(f"forensic_score={agg.get('forensic', 0.5):.2f}")

    return {
        "verdict": verdict,
        "probability": float(fused),
        "risk": risk,
        "explanation": "; ".join(reasons)
    }


def main():
    parser = argparse.ArgumentParser(description="Offline fake face detector (CLI)")
    parser.add_argument("--input", required=True, help="Path to image or video file")
    args = parser.parse_args()

    out = analyze_path(args.input)

    print("\n=== Fake Face Detector Result ===")
    print(f"Verdict: {out['verdict']}")
    print(f"Fake probability: {out['probability']:.3f}")
    print(f"Fraud risk level: {out['risk']}")
    print(f"Signals: {out['explanation']}")


if __name__ == '__main__':
    main()
