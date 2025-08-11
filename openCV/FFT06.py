import cv2
import numpy as np
import sys
import csv
import os
from collections import deque
from datetime import datetime

# 检查可选依赖
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

# ----------------------------
# 颜色平均值计算
# ----------------------------
def calculate_hsv_average(roi):
    """计算HSV各通道的平均值"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_avg = np.mean(hsv[:, :, 0])  # 色调
    s_avg = np.mean(hsv[:, :, 1])  # 饱和度
    v_avg = np.mean(hsv[:, :, 2])  # 明度
    return h_avg, s_avg, v_avg

def calculate_grayscale_average(roi):
    """计算灰度平均值"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_pca_signal(roi):
    """计算PCA主成分信号"""
    reshaped = roi.reshape(-1, 3).astype(np.float32)
    reshaped -= np.mean(reshaped, axis=0)
    cov = np.cov(reshaped, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pca_component = eigvecs[:, np.argmax(eigvals)]
    projected = reshaped @ pca_component
    return np.mean(projected)

# ----------------------------
# 信号处理
# ----------------------------
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fps):
    """信号预处理"""
    detrended = signal - np.mean(signal)
    filtered = butter_bandpass_filter(detrended, 0.75, 4.0, fps)  # 心率范围对应频率
    denoised = medfilt(filtered, kernel_size=3)  # 中值滤波去噪
    window = np.hanning(len(denoised))  # 加窗减少频谱泄漏
    return denoised * window

def compute_heart_rate(signal, fps, min_hr=48, max_hr=180):
    """计算心率"""
    n = len(signal)
    if n < 10:
        return None
    
    processed = preprocess_signal(signal, fps)
    fft_vals = fft(processed)
    freqs = fftfreq(n, d=1.0 / fps)
    
    half = n // 2
    freqs = freqs[:half]
    mags = np.abs(fft_vals[:half])
    bpm_freqs = freqs * 60.0  # 转换为BPM
    
    # 只在合理的心率范围内寻找峰值
    mask = (bpm_freqs > min_hr) & (bpm_freqs < max_hr)
    if not np.any(mask):
        return None
    
    peaks, _ = find_peaks(mags[mask], height=0)
    if len(peaks) == 0:
        return None
    
    peak_idx = peaks[np.argmax(mags[mask][peaks])]
    return bpm_freqs[mask][peak_idx]

# ----------------------------
# 绘图和CSV辅助函数
# ----------------------------
def create_hr_plot(heart_rates, time_points):
    """创建心率曲线图"""
    if not PLOTTING_ENABLED or len(heart_rates) < 2:
        return np.zeros((300, 500, 3), dtype=np.uint8)
    
    plt.figure(figsize=(5, 3))
    plt.plot(time_points, heart_rates, "b-", linewidth=2)
    plt.plot(time_points, heart_rates, "ro", markersize=4)
    plt.title("Heart Rate Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.text(time_points[-1], min(heart_rates), f"Min: {min(heart_rates):.1f}", 
             fontsize=9, va="top")
    plt.text(time_points[-1], max(heart_rates), f"Max: {max(heart_rates):.1f}", 
             fontsize=9, va="bottom")
    plt.tight_layout()
    plt.savefig("temp_plot.png", dpi=100)
    plt.close()
    
    img = cv2.imread("temp_plot.png")
    return img if img is not None else np.zeros((300, 500, 3), dtype=np.uint8)

def save_hr_data(time_pts, hr_vals, fps):
    """保存心率数据到CSV"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"heart_rate_data_{ts}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Heart Rate (BPM)", "Frame Rate (FPS)"])
        for t, hr in zip(time_pts, hr_vals):
            writer.writerow([f"{t:.2f}", f"{hr:.1f}", f"{fps:.2f}"])
    print(f"[INFO] Saved: {fname}")
    return fname

# ----------------------------
# 主应用程序
# ----------------------------
def main():
    # 修改为你的视频路径
    video_path = "face.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"Video frame rate: {fps:.2f} FPS")

    wait_time = int(1000 / fps)
    window_size = int(fps * 10)  # 10秒的滑动窗口

    # 信号缓冲区
    hsv_h_buffer = deque(maxlen=window_size)  # H通道
    hsv_s_buffer = deque(maxlen=window_size)  # S通道
    hsv_v_buffer = deque(maxlen=window_size)  # V通道
    gray_buffer = deque(maxlen=window_size)    # 灰度
    pca_buffer = deque(maxlen=window_size)     # PCA

    heart_rates, time_points = [], []
    current_hr = None
    frame_count = 0
    start_tick = cv2.getTickCount()

    signal_mode = "hsv_v"  # 默认使用HSV的V通道
    # 可选模式: "hsv_h", "hsv_s", "hsv_v", "gray", "pca"

    cv2.namedWindow("Heart Rate Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Heart Rate Estimation", 1200, 700)
    hr_plot_img = np.zeros((300, 500, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        roi = frame  # 这里可以添加ROI选择逻辑

        # 计算所有通道的信号
        h_avg, s_avg, v_avg = calculate_hsv_average(roi)
        hsv_h_buffer.append(h_avg)
        hsv_s_buffer.append(s_avg)
        hsv_v_buffer.append(v_avg)
        
        gray_avg = calculate_grayscale_average(roi)
        gray_buffer.append(gray_avg)
        
        pca_avg = calculate_pca_signal(roi)
        pca_buffer.append(pca_avg)

        # 定期计算心率（每秒计算约2次）
        if frame_count % max(1, int(fps / 2)) == 0:
            if len(hsv_v_buffer) > fps * 2:  # 确保有足够的数据
                # 根据当前模式选择信号源
                if signal_mode == "hsv_h":
                    signal = np.array(hsv_h_buffer, dtype=np.float32)
                elif signal_mode == "hsv_s":
                    signal = np.array(hsv_s_buffer, dtype=np.float32)
                elif signal_mode == "hsv_v":
                    signal = np.array(hsv_v_buffer, dtype=np.float32)
                elif signal_mode == "gray":
                    signal = np.array(gray_buffer, dtype=np.float32)
                elif signal_mode == "pca":
                    signal = np.array(pca_buffer, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported signal mode: {signal_mode}")

                current_hr = compute_heart_rate(signal, fps)
                if current_hr is not None:
                    elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
                    heart_rates.append(current_hr)
                    time_points.append(elapsed)
                    print(f"[{elapsed:.1f}s] HR ({signal_mode}): {current_hr:.1f} BPM")

                    # 定期更新心率图
                    if len(heart_rates) > 1 and len(heart_rates) % 2 == 0:
                        hr_plot_img = create_hr_plot(heart_rates, time_points)

        # 在视频帧上显示信息
        if current_hr is not None:
            # 显示当前使用的具体通道
            mode_display = {
                "hsv_h": "HSV (Hue)",
                "hsv_s": "HSV (Saturation)",
                "hsv_v": "HSV (Value)",
                "gray": "Grayscale",
                "pca": "PCA"
            }.get(signal_mode, signal_mode)
            
            cv2.putText(frame, f"Heart Rate: {current_hr:.1f} BPM", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Mode: {mode_display}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        # 组合视频帧和心率图
        if hr_plot_img is not None and hr_plot_img.size > 0:
            h = frame.shape[0]
            w = int(hr_plot_img.shape[1] * (h / hr_plot_img.shape[0]))
            plot_resized = cv2.resize(hr_plot_img, (w, h))
            combined = np.hstack((frame, plot_resized))
        else:
            combined = frame

        cv2.imshow("Heart Rate Estimation", combined)
        
        # 键盘控制
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord("q"):  # 退出
            break
        elif key == ord("s") and heart_rates:  # 保存数据
            save_hr_data(time_points, heart_rates, fps)
        elif key == ord("h"):  # 切换到HSV的H通道
            signal_mode = "hsv_h"
            print("Switched to HSV (Hue) channel")
        elif key == ord("s"):  # 切换到HSV的S通道
            signal_mode = "hsv_s"
            print("Switched to HSV (Saturation) channel")
        elif key == ord("v"):  # 切换到HSV的V通道
            signal_mode = "hsv_v"
            print("Switched to HSV (Value) channel")
        elif key == ord("g"):  # 切换到灰度模式
            signal_mode = "gray"
            print("Switched to Grayscale mode")
        elif key == ord("p"):  # 切换到PCA模式
            signal_mode = "pca"
            print("Switched to PCA mode")

    # 清理资源
    if heart_rates:
        save_hr_data(time_points, heart_rates, fps)

    cap.release()
    if os.path.exists("temp_plot.png"):
        os.remove("temp_plot.png")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()