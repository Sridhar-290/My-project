import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time

class LiveRPPG:
    def __init__(self, window_size=250, fs=30):
        self.window_size = window_size
        self.fs = fs
        self.buffer = deque(maxlen=window_size)
        self.times = deque(maxlen=window_size)
        self.bpm = 0
        self.hrv = 0 # Heart Rate Variability
        self.resp_rate = 0 # Respiratory Rate
        self.confidence = 0.0
        self.last_update = time.time()

    def _bandpass_filter(self, data, low=0.75, high=4.0):
        ny = 0.5 * self.fs
        b, a = butter(3, [low/ny, high/ny], btype='band')
        return filtfilt(b, a, data)

    def update(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return None
            
        h, w, _ = face_roi.shape
        # Target forehead and cheeks
        roi = face_roi[int(h*0.1):int(h*0.35), int(w*0.3):int(w*0.7)]
        if roi.size == 0: return None
        
        green_val = np.mean(roi[:, :, 1])
        self.buffer.append(green_val)
        self.times.append(time.time())

        if len(self.buffer) < self.fs * 4: # Need 4 seconds for stabilization
            return {"bpm": 0, "hrv": 0, "respiration": 0, "conf": 0.0}

        if time.time() - self.last_update < 0.5:
            return {"bpm": self.bpm, "hrv": self.hrv, "respiration": self.resp_rate, "conf": self.confidence}

        try:
            data = np.array(self.buffer)
            data = data - np.mean(data)
            
            # 1. BPM Calculation (Cardiac Band)
            filtered_cardiac = self._bandpass_filter(data, 0.75, 4.0)
            fft = np.abs(np.fft.rfft(filtered_cardiac))
            freqs = np.fft.rfftfreq(len(filtered_cardiac), 1.0/self.fs)
            idx = np.where((freqs >= 0.75) & (freqs <= 4.0))[0]
            if len(idx) > 0:
                peak_idx = idx[np.argmax(fft[idx])]
                self.bpm = freqs[peak_idx] * 60
                self.confidence = min(1.0, (fft[peak_idx] / (np.median(fft[idx]) + 1e-6)) / 12.0)

            # 2. HRV Calculation (Time between peaks)
            peaks, _ = find_peaks(filtered_cardiac, distance=self.fs//3)
            if len(peaks) > 2:
                intervals = np.diff(peaks) / self.fs
                self.hrv = np.std(intervals) * 1000 # Standard deviation in ms

            # 3. Respiration Rate (Lower Frequency Band)
            # Breathing is typically 0.1Hz to 0.4Hz (6-24 breaths per min)
            filtered_resp = self._bandpass_filter(data, 0.1, 0.5)
            resp_fft = np.abs(np.fft.rfft(filtered_resp))
            resp_freqs = np.fft.rfftfreq(len(filtered_resp), 1.0/self.fs)
            resp_idx = np.where((resp_freqs >= 0.1) & (resp_freqs <= 0.5))[0]
            if len(resp_idx) > 0:
                self.resp_rate = resp_freqs[resp_idx[np.argmax(resp_fft[resp_idx])]] * 60

            self.last_update = time.time()
        except Exception:
            pass

        return {
            "bpm": round(self.bpm, 1), 
            "hrv": round(self.hrv, 1), 
            "respiration": round(self.resp_rate, 1), 
            "conf": round(self.confidence, 2)
        }
