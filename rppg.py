"""
rPPG estimation module (remote photoplethysmography)

Provides a lightweight, offline rPPG estimator that extracts the mean green-channel
signal from the detected face ROI and finds a dominant frequency in the cardiac band (0.7-4.0 Hz).

Returns a normalized score in [0,1] where higher means stronger evidence of a biological heartbeat.
"""
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import math


def bandpass(signal, fs, low=0.7, high=4.0, order=3):
    ny = 0.5 * fs
    lown = low / ny
    highn = high / ny
    b, a = butter(order, [lown, highn], btype='band')
    return filtfilt(b, a, signal)


def detect_face_gray(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))
    if len(faces) == 0:
        return None
    # return largest face
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces[0]
    return (x,y,w,h)


def estimate_rppg(video_path, max_seconds=30):
    """Estimate rPPG heartbeat score from a video file.

    Returns (score, bpm_estimate) where score in [0,1] indicates confidence of biological pulse.
    If not sufficient frames or failure, returns (None, None).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(min(total_frames, max_seconds * fps))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    greens = []
    times = []
    read = 0
    while read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        face = detect_face_gray(frame, face_cascade)
        if face is not None:
            x,y,w,h = face
            # take central region of face to avoid background
            cx = int(x + w*0.2)
            cy = int(y + h*0.2)
            cw = int(w*0.6)
            ch = int(h*0.6)
            roi = frame[cy:cy+ch, cx:cx+cw]
            if roi.size == 0:
                continue
            # mean green channel
            g = float(np.mean(roi[:,:,1]))
            greens.append(g)
            times.append(t)
        read += 1

    cap.release()

    if len(greens) < 30:
        return None, None

    greens = np.array(greens)
    times = np.array(times)
    # resample to uniform fs using linear interpolation
    duration = times[-1] - times[0]
    if duration <= 1.0:
        return None, None
    target_fs = 30.0
    t_uniform = np.arange(times[0], times[-1], 1.0/target_fs)
    g_uniform = np.interp(t_uniform, times, greens)

    # detrend
    g_d = g_uniform - np.polyval(np.polyfit(np.arange(len(g_uniform)), g_uniform, 1), np.arange(len(g_uniform)))

    # bandpass filter to cardiac band
    try:
        g_f = bandpass(g_d, fs=target_fs, low=0.7, high=4.0, order=3)
    except Exception:
        g_f = g_d

    # compute FFT and find peak
    n = len(g_f)
    freqs = np.fft.rfftfreq(n, d=1.0/target_fs)
    spec = np.abs(np.fft.rfft(g_f))
    # restrict to cardiac band
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return None, None
    freqs_band = freqs[mask]
    spec_band = spec[mask]
    peak_idx = np.argmax(spec_band)
    peak_freq = freqs_band[peak_idx]
    bpm = peak_freq * 60.0

    # SNR-like measure: ratio of peak energy to median
    peak_power = spec_band[peak_idx]
    median_power = np.median(spec_band) + 1e-8
    snr = float((peak_power / median_power - 1.0) / 10.0)
    # normalize snr roughly into [0,1]
    score = max(0.0, min(1.0, snr))

    return score, round(bpm, 1)
