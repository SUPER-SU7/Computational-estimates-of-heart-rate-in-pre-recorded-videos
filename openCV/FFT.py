#单一的抓取实时的心率，可以在视频中实时显示出来，但是没有折线图


import cv2
import numpy as np
import sys
from collections import deque

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

# Main function
def main():
    video_path = '/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/face1.mp4'
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video frame rate: {fps:.2f} FPS")
    
    wait_time = int(1000 / fps)  # Control playback speed based on frame rate

    window_size = int(fps * 10)
    green_values = deque(maxlen=window_size)
    heart_rate_history = deque(maxlen=100)
    
    if PLOTTING_ENABLED:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    frame_count = 0
    current_hr = None
    
    # Video processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        face_rect = detect_face(frame)
        
        # Face detection
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            fx, fy, fw, fh = get_forehead_region(face_rect)
            forehead_roi = frame[fy:fy+fh, fx:fx+fw]
            
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            
            # Signal acquisition
            _, g_avg, _ = calculate_rgb_average(forehead_roi)
            green_values.append(g_avg)
            
            if frame_count % max(1, int(fps / 2)) == 0 and len(green_values) > max(10, fps * 2):
                current_hr = compute_heart_rate(list(green_values), fps)
                
                if current_hr is not None:
                    heart_rate_history.append(current_hr)
                    avg_hr = np.mean(list(heart_rate_history)[-10:]) if heart_rate_history else current_hr
                    print(f"Current heart rate: {avg_hr:.1f} BPM")
                    
                    if PLOTTING_ENABLED:
                        update_plots(ax1, ax2, list(green_values), list(heart_rate_history), fps)
        
        if current_hr is not None:
            hr_text = f"Heart Rate: {current_hr:.1f} BPM"
            cv2.putText(frame, hr_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Heart Rate Estimation', frame)
        
        key = cv2.waitKey(wait_time)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if PLOTTING_ENABLED:
        plt.ioff()
        plt.show()

# Data visualization
def update_plots(ax1, ax2, signal, heart_rates, fps):
    """Update real-time plots"""
    ax1.clear()
    ax2.clear()
    
    ax1.set_title('Green Channel Signal')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Intensity')
    ax1.plot(signal, 'g-')
    
    if heart_rates:
        ax2.set_title('Heart Rate Estimation')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('BPM')
        
        time_axis = np.arange(len(heart_rates)) * 0.1
        ax2.plot(time_axis, heart_rates, 'b-')
        ax2.plot(time_axis, heart_rates, 'ro')
        
        last_hr = heart_rates[-1]
        ax2.text(time_axis[-1], last_hr, f'{last_hr:.1f}', fontsize=9)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    main()