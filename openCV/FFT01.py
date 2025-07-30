#有折线图的生成，也可以有实时心率，输出结果为.csv文件

import cv2
import numpy as np
import sys
import csv
import os
from collections import deque
from datetime import datetime

# Try to import necessary libraries
try:
    from scipy.fftpack import fft, fftfreq 
    from scipy.signal import find_peaks
except ImportError:
    print("Error: SciPy library not found. Please install with:")
    print("pip install scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

# Implement face detection function
def detect_face(img):
    """Detect faces and return the largest face region"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = '/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/XML/haarcascade_frontalface_default.xml'
    
    try:
        face_detect = cv2.CascadeClassifier(cascade_path)
    except:
        print(f"Error: Unable to load cascade classifier {cascade_path}")
        return None
    
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        return max(faces, key=lambda rect: rect[2] * rect[3])
    return None

# Locate forehead region
def get_forehead_region(face_rect):
    """Extract forehead region from face rectangle"""
    x, y, w, h = face_rect
    forehead_height = int(h * 0.25)
    forehead_width = int(w * 0.6)
    forehead_x = x + int((w - forehead_width) / 2)
    forehead_y = y + int(h * 0.1)
    
    return (forehead_x, forehead_y, forehead_width, forehead_height)

def calculate_rgb_average(roi):
    """Calculate average RGB values of ROI region"""
    return np.mean(roi, axis=(0, 1))

# Core heart rate calculation algorithm
def compute_heart_rate(signal, fps, min_hr=48, max_hr=180):
    """Calculate heart rate (BPM) using FFT"""
    n = len(signal)
    if n < 10:
        return None
    
    detrended = signal - np.mean(signal)
    window = np.hanning(n)
    windowed = detrended * window
    
    fft_result = fft(windowed)
    freqs = fftfreq(n, d=1.0/fps)
    
    half_n = n // 2
    freqs = freqs[:half_n]
    magnitudes = np.abs(fft_result[:half_n])
    
    bpm_freqs = freqs * 60.0
    valid_mask = (bpm_freqs > min_hr) & (bpm_freqs < max_hr)
    valid_freqs = bpm_freqs[valid_mask]
    valid_magnitudes = magnitudes[valid_mask]
    
    if len(valid_freqs) == 0:
        return None
    
    peaks, _ = find_peaks(valid_magnitudes, height=0)
    if len(peaks) == 0:
        return None
    
    max_idx = peaks[np.argmax(valid_magnitudes[peaks])]
    return valid_freqs[max_idx]

def create_hr_plot(heart_rates, time_points):
    if not PLOTTING_ENABLED:
        print("Plotting disabled because matplotlib is not installed.")
        return np.zeros((300, 500, 3), dtype=np.uint8)

# Create heart rate plot image
def create_hr_plot(heart_rates, time_points):
    """Create a heart rate plot image for display in OpenCV window"""
    if not PLOTTING_ENABLED:
        return np.zeros((300, 500, 3), dtype=np.uint8)

    if len(heart_rates) < 2:
        return np.zeros((300, 500, 3), dtype=np.uint8)
    
    plt.figure(figsize=(5, 3))
    plt.plot(time_points, heart_rates, 'b-', linewidth=2)
    plt.plot(time_points, heart_rates, 'ro', markersize=4)
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('BPM')
    plt.grid(True)
    
    min_hr = min(heart_rates)
    max_hr = max(heart_rates)
    plt.text(time_points[-1], min_hr, f'Min: {min_hr:.1f}', fontsize=9, verticalalignment='top')
    plt.text(time_points[-1], max_hr, f'Max: {max_hr:.1f}', fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('temp_plot.png', dpi=100)
    plt.close()
    
    plot_img = cv2.imread('temp_plot.png')
    if plot_img is None:
        return np.zeros((300, 500, 3), dtype=np.uint8)
    return plot_img

# Save heart rate data to CSV
def save_hr_data(time_points, heart_rates, fps):
    """Save heart rate data to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'heart_rate_data_{timestamp}.csv'
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time (s)', 'Heart Rate (BPM)', 'Frame Rate (FPS)'])
        for t, hr in zip(time_points, heart_rates):
            writer.writerow([f"{t:.2f}", f"{hr:.1f}", f"{fps:.2f}"])
    
    print(f"Heart rate data saved to {filename}")
    return filename

# Main function
def main():
    video_path = '/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/face.mp4'
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video frame rate: {fps:.2f} FPS")
    
    wait_time = int(1000 / fps)

    window_size = int(fps * 10)
    green_values = deque(maxlen=window_size)
    
    heart_rates = []
    time_points = []
    plot_update_counter = 0
    current_hr = None
    
    cv2.namedWindow('Heart Rate Estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Heart Rate Estimation', 1200, 700)
    
    hr_plot_img = np.zeros((300, 500, 3), dtype=np.uint8)
    
    frame_count = 0
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        face_rect = detect_face(frame)
        
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            fx, fy, fw, fh = get_forehead_region(face_rect)
            forehead_roi = frame[fy:fy+fh, fx:fx+fw]
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            
            _, g_avg, _ = calculate_rgb_average(forehead_roi)
            green_values.append(g_avg)
            
            if frame_count % max(1, int(fps / 2)) == 0 and len(green_values) > max(10, fps * 2):
                current_hr = compute_heart_rate(list(green_values), fps)
                
                if current_hr is not None:
                    current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                    heart_rates.append(current_hr)
                    time_points.append(current_time)
                    
                    print(f"Time: {current_time:.1f}s, Heart Rate: {current_hr:.1f} BPM")
                    
                    plot_update_counter += 1
                    if plot_update_counter % 2 == 0 and len(heart_rates) > 1:
                        hr_plot_img = create_hr_plot(heart_rates, time_points)
        
        if current_hr is not None:
            hr_text = f"Heart Rate: {current_hr:.1f} BPM"
            cv2.putText(frame, hr_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            time_text = f"Time: {elapsed_time:.1f}s"
            cv2.putText(frame, time_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if hr_plot_img is not None and hr_plot_img.size > 0:
            plot_height = frame.shape[0]
            plot_width = int(hr_plot_img.shape[1] * (plot_height / hr_plot_img.shape[0]))
            resized_plot = cv2.resize(hr_plot_img, (plot_width, plot_height))
            combined_frame = np.hstack((frame, resized_plot))
        else:
            combined_frame = frame
        
        cv2.imshow('Heart Rate Estimation', combined_frame)
        
        key = cv2.waitKey(wait_time)
        if key == ord('q'):
            break
        elif key == ord('s') and len(heart_rates) > 0:
            save_hr_data(time_points, heart_rates, fps)
    
    if len(heart_rates) > 0:
        save_hr_data(time_points, heart_rates, fps)
    
    cap.release()
    if os.path.exists('temp_plot.png'):
        os.remove('temp_plot.png')
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
