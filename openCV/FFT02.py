
"""
Heart-rate estimation from pre-recorded facial video using rPPG
Modified 2025-07-07: fixed TypeError in quality_weighted_fusion
"""

import cv2
import numpy as np
import sys
import csv
import os
from collections import deque, defaultdict
from datetime import datetime

# ────────────────────────────────────────────────────────────────
# Optional dependencies
# ────────────────────────────────────────────────────────────────
try:
    from scipy.fftpack import fft, fftfreq
    from scipy.signal import find_peaks, butter, filtfilt, medfilt
except ImportError:
    print("Error: SciPy library not found. Install it with:")
    print("   pip install scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

# ────────────────────────────────────────────────────────────────
# Face detection utilities
# ────────────────────────────────────────────────────────────────
def detect_face(img):
    """Detect faces and return the largest face rectangle (x, y, w, h)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = (
        "/Users/suziteng/Documents/GitHub/"
        "Computational-estimates-of-heart-rate-in-pre-recorded-videos/XML/"
        "haarcascade_frontalface_default.xml"
    )
    face_detect = cv2.CascadeClassifier(cascade_path)
    if face_detect.empty():
        print(f"[ERROR] Could not load cascade: {cascade_path}")
        return None

    faces = face_detect.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) > 0:
        return max(faces, key=lambda r: r[2] * r[3])
    return None


def get_face_regions(face_rect):
    """Return ROIs (x, y, w, h) for forehead, cheeks and nose."""
    x, y, w, h = face_rect
    return {
        "forehead": (x + int(w * 0.2), y + int(h * 0.1), int(w * 0.6), int(h * 0.15)),
        "left_cheek": (x + int(w * 0.1), y + int(h * 0.4), int(w * 0.3), int(h * 0.2)),
        "right_cheek": (x + int(w * 0.6), y + int(h * 0.4), int(w * 0.3), int(h * 0.2)),
        "nose": (x + int(w * 0.35), y + int(h * 0.3), int(w * 0.3), int(h * 0.2)),
    }


def calculate_rgb_average(roi):
    """Return mean (B, G, R) values of an ROI."""
    return np.mean(roi, axis=(0, 1))

# ────────────────────────────────────────────────────────────────
# Signal processing
# ────────────────────────────────────────────────────────────────
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def preprocess_signal(signal, fps):
    """Detrend, band-pass, median-filter, and window a raw signal."""
    detrended = signal - np.mean(signal)
    filtered = butter_bandpass_filter(detrended, 0.75, 4.0, fps)  # 45–240 BPM
    denoised = medfilt(filtered, kernel_size=3)
    window = np.hanning(len(denoised))
    return denoised * window


def compute_heart_rate(signal, fps, min_hr=48, max_hr=180):
    """Return heart-rate (BPM) computed via FFT of a pre-processed signal."""
    n = len(signal)
    if n < 10:
        return None

    processed = preprocess_signal(signal, fps)
    fft_vals = fft(processed)
    freqs = fftfreq(n, d=1.0 / fps)

    # Take positive frequencies only
    half = n // 2
    freqs = freqs[:half]
    mags = np.abs(fft_vals[:half])

    bpm_freqs = freqs * 60.0
    mask = (bpm_freqs > min_hr) & (bpm_freqs < max_hr)
    if not np.any(mask):
        return None

    peaks, _ = find_peaks(mags[mask], height=0)
    if len(peaks) == 0:
        return None

    peak_idx = peaks[np.argmax(mags[mask][peaks])]
    return bpm_freqs[mask][peak_idx]


def calculate_snr(signal):
    """Crude SNR estimation (dB) of a signal."""
    signal = np.asarray(signal)
    fft_vals = np.abs(fft(signal - np.mean(signal)))
    ps = np.max(fft_vals)
    pn = np.median(fft_vals)
    return 10 * np.log10(ps / (pn + 1e-6))

# ────────────────────────────────────────────────────────────────
# ★★★  FIXED FUNCTION  ★★★
# ────────────────────────────────────────────────────────────────
def quality_weighted_fusion(region_signals):
    """
    Fuse multiple region signals into one, weighting by each signal's SNR.

    region_signals: dict(name -> 1-D iterable of samples)
    returns: 1-D NumPy array (fused signal)
    """
    signals = []
    weights = []

    for name, sig in region_signals.items():
        sig_arr = np.asarray(sig, dtype=np.float32)  # ← ensure NumPy array
        signals.append(sig_arr)
        weights.append(calculate_snr(sig_arr))

    weights = np.array(weights, dtype=np.float32) + 1e-6
    weights /= np.sum(weights)

    fused = np.zeros_like(signals[0])
    for sig_arr, w in zip(signals, weights):
        fused += sig_arr * w
    return fused

# ────────────────────────────────────────────────────────────────
# Plotting & CSV helpers
# ────────────────────────────────────────────────────────────────
def create_hr_plot(heart_rates, time_points):
    """Return an image of a HR-vs-time matplotlib plot."""
    if not PLOTTING_ENABLED or len(heart_rates) < 2:
        return np.zeros((300, 500, 3), dtype=np.uint8)

    plt.figure(figsize=(5, 3))
    plt.plot(time_points, heart_rates, "b-", linewidth=2)
    plt.plot(time_points, heart_rates, "ro", markersize=4)
    plt.title("Heart Rate Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.grid(True)

    plt.text(
        time_points[-1],
        min(heart_rates),
        f"Min: {min(heart_rates):.1f}",
        fontsize=9,
        va="top",
    )
    plt.text(
        time_points[-1],
        max(heart_rates),
        f"Max: {max(heart_rates):.1f}",
        fontsize=9,
        va="bottom",
    )
    plt.tight_layout()
    plt.savefig("temp_plot.png", dpi=100)
    plt.close()
    img = cv2.imread("temp_plot.png")
    return img if img is not None else np.zeros((300, 500, 3), dtype=np.uint8)


def save_hr_data(time_pts, hr_vals, fps):
    """Whole face; Write HR data to a timestamped CSV; return filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"heart_rate_data_{ts}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Heart Rate (BPM)", "Frame Rate (FPS)"])
        for t, hr in zip(time_pts, hr_vals):
            writer.writerow([f"{t:.2f}", f"{hr:.1f}", f"{fps:.2f}"])
    print(f"[INFO] Saved: {fname}")
    return fname

# ────────────────────────────────────────────────────────────────
# Main application
# ────────────────────────────────────────────────────────────────
def main():
    video_path = (
        "/Users/suziteng/Documents/GitHub/"
        "Computational-estimates-of-heart-rate-in-pre-recorded-videos/face.mp4"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"Video frame rate: {fps:.2f} FPS")

    wait_time = int(1000 / fps)
    window_size = int(fps * 10)  # 10-second sliding buffer

    region_buffers = {
        "forehead": deque(maxlen=window_size),
        "left_cheek": deque(maxlen=window_size),
        "right_cheek": deque(maxlen=window_size),
        "nose": deque(maxlen=window_size),
    }

    heart_rates, time_points = [], []
    current_hr = None
    frame_count = 0
    start_tick = cv2.getTickCount()

    cv2.namedWindow("Heart Rate Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Heart Rate Estimation", 1200, 700)
    hr_plot_img = np.zeros((300, 500, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        face_rect = detect_face(frame)

        if face_rect is not None:
            face_rois = get_face_regions(face_rect)
            for name, (x, y, w, h) in face_rois.items():
                roi = frame[y : y + h, x : x + w]
                b_avg, g_avg, r_avg = calculate_rgb_average(roi)
                region_buffers[name].append(g_avg)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    name,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

            # HR update every 0.5 s
            if frame_count % max(1, int(fps / 2)) == 0:
                if all(len(buf) > fps * 2 for buf in region_buffers.values()):
                    signals_to_fuse = {
                        n: np.array(buf, dtype=np.float32)
                        for n, buf in region_buffers.items()
                    }
                    fused_signal = quality_weighted_fusion(signals_to_fuse)
                    current_hr = compute_heart_rate(fused_signal, fps)

                    if current_hr is not None:
                        elapsed = (
                            cv2.getTickCount() - start_tick
                        ) / cv2.getTickFrequency()
                        heart_rates.append(current_hr)
                        time_points.append(elapsed)
                        print(f"[{elapsed:.1f}s] HR: {current_hr:.1f} BPM")

                        if len(heart_rates) > 1 and len(heart_rates) % 2 == 0:
                            hr_plot_img = create_hr_plot(heart_rates, time_points)

        # On-screen overlays
        if current_hr is not None:
            cv2.putText(
                frame,
                f"Heart Rate: {current_hr:.1f} BPM",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
            cv2.putText(
                frame,
                f"Time: {elapsed:.1f}s",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Combine frame + plot
        if hr_plot_img is not None and hr_plot_img.size > 0:
            h = frame.shape[0]
            w = int(hr_plot_img.shape[1] * (h / hr_plot_img.shape[0]))
            plot_resized = cv2.resize(hr_plot_img, (w, h))
            combined = np.hstack((frame, plot_resized))
        else:
            combined = frame

        cv2.imshow("Heart Rate Estimation", combined)
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and heart_rates:
            save_hr_data(time_points, heart_rates, fps)

    # Save upon exit
    if heart_rates:
        save_hr_data(time_points, heart_rates, fps)

    cap.release()
    if os.path.exists("temp_plot.png"):
        os.remove("temp_plot.png")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
