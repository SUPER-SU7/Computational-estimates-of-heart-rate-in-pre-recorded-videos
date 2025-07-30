import cv2
import numpy as np
import sys
import csv
import os
from collections import deque
from datetime import datetime

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

# ───────────────────────────────
# ROI Selection (Global State)
# ───────────────────────────────
roi_box = None
selecting = False
start_point = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global roi_box, selecting, start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        frame_copy = param.copy()
        cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        end_point = (x, y)
        x0, y0 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x1, y1 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        roi_box = (x0, y0, x1 - x0, y1 - y0)
        print(f"[INFO] Selected ROI: {roi_box}")

# ───────────────────────────────
# Color averaging
# ───────────────────────────────
def calculate_rgb_average(roi):
    return np.mean(roi, axis=(0, 1))

# ───────────────────────────────
# Signal processing
# ───────────────────────────────
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fps):
    detrended = signal - np.mean(signal)
    filtered = butter_bandpass_filter(detrended, 0.75, 4.0, fps)
    denoised = medfilt(filtered, kernel_size=3)
    window = np.hanning(len(denoised))
    return denoised * window

def compute_heart_rate(signal, fps, min_hr=48, max_hr=180):
    n = len(signal)
    if n < 10:
        return None
    processed = preprocess_signal(signal, fps)
    fft_vals = fft(processed)
    freqs = fftfreq(n, d=1.0 / fps)
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

# ───────────────────────────────
# Plotting & CSV helpers
# ───────────────────────────────
def create_hr_plot(heart_rates, time_points):
    if not PLOTTING_ENABLED or len(heart_rates) < 2:
        return np.zeros((300, 500, 3), dtype=np.uint8)
    plt.figure(figsize=(5, 3))
    plt.plot(time_points, heart_rates, "b-", linewidth=2)
    plt.plot(time_points, heart_rates, "ro", markersize=4)
    plt.title("Heart Rate Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.text(time_points[-1], min(heart_rates), f"Min: {min(heart_rates):.1f}", fontsize=9, va="top")
    plt.text(time_points[-1], max(heart_rates), f"Max: {max(heart_rates):.1f}", fontsize=9, va="bottom")
    plt.tight_layout()
    plt.savefig("temp_plot.png", dpi=100)
    plt.close()
    img = cv2.imread("temp_plot.png")
    return img if img is not None else np.zeros((300, 500, 3), dtype=np.uint8)

def save_hr_data(time_pts, hr_vals, fps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"heart_rate_data_{ts}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Heart Rate (BPM)", "Frame Rate (FPS)"])
        for t, hr in zip(time_pts, hr_vals):
            writer.writerow([f"{t:.2f}", f"{hr:.1f}", f"{fps:.2f}"])
    print(f"[INFO] Saved: {fname}")
    return fname

# ───────────────────────────────
# Main application
# ───────────────────────────────
def main():
    global roi_box

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
    window_size = int(fps * 10)
    green_buffer = deque(maxlen=window_size)

    heart_rates, time_points = [], []
    current_hr = None
    frame_count = 0
    start_tick = cv2.getTickCount()

    # ───────── Select ROI ─────────
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        return

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback, first_frame.copy())
    print("[INFO] Please select a face ROI. Press any key when done.")
    while True:
        temp_frame = first_frame.copy()
        if roi_box:
            x, y, w, h = roi_box
            cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_frame)
        if cv2.waitKey(1) & 0xFF != 255:
            break
    cv2.destroyWindow("Select ROI")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # ───────── Main Loop ─────────
    cv2.namedWindow("Heart Rate Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Heart Rate Estimation", 1200, 700)
    hr_plot_img = np.zeros((300, 500, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Use ROI if selected
        if roi_box:
            x, y, w, h = roi_box
            roi = frame[y:y+h, x:x+w]
        else:
            roi = frame

        b_avg, g_avg, r_avg = calculate_rgb_average(roi)
        green_buffer.append(g_avg)

        print(f"[Frame {frame_count}] Green Channel Avg: {g_avg:.2f}")

        if frame_count % max(1, int(fps / 2)) == 0:
            if len(green_buffer) > fps * 2:
                signal = np.array(green_buffer, dtype=np.float32)
                current_hr = compute_heart_rate(signal, fps)
                if current_hr is not None:
                    elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
                    heart_rates.append(current_hr)
                    time_points.append(elapsed)
                    print(f"[{elapsed:.1f}s] HR: {current_hr:.1f} BPM")
                    if len(heart_rates) > 1 and len(heart_rates) % 2 == 0:
                        hr_plot_img = create_hr_plot(heart_rates, time_points)

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

    if heart_rates:
        save_hr_data(time_points, heart_rates, fps)

    cap.release()
    if os.path.exists("temp_plot.png"):
        os.remove("temp_plot.png")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
