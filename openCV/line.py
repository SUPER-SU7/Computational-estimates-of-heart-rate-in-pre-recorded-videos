import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Read CSV file (replace with your actual file path)
file_path = "/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/7.19/video1:FFT01:heart_rate_data_20250704_103631 copy.csv"
df = pd.read_csv(file_path)

# Data preprocessing
# Ensure time series continuity (handle possible missing values with linear interpolation)
df['Time (s)'] = df['Time (s)'].interpolate(method='linear')
df['Heart Rate (BPM)'] = df['Heart Rate (BPM)'].interpolate(method='linear')

# Smooth heart rate data using Savitzky-Golay filter
window_size = min(9, len(df))  # Ensure window size doesn't exceed data length
if window_size % 2 == 0:  # Window size must be odd
    window_size = max(3, window_size - 1)
    
smoothed_hr = savgol_filter(df['Heart Rate (BPM)'], window_size, 3)

# Create figure
plt.figure(figsize=(12, 6))

# Plot raw heart rate data (light color with transparency)
plt.plot(df['Time (s)'], df['Heart Rate (BPM)'], 
         'o-', color='skyblue', alpha=0.5, 
         label='Raw Heart Rate', markersize=4)

# Plot smoothed heart rate data (main trend line)
plt.plot(df['Time (s)'], smoothed_hr, 
         '-', color='dodgerblue', linewidth=2.5, 
         label='Smoothed Heart Rate')

# Mark anomalies (below 60 or above 100 BPM)
threshold_low = 60
threshold_high = 100
anomalies = df[(df['Heart Rate (BPM)'] < threshold_low) | 
            (df['Heart Rate (BPM)'] > threshold_high)]

plt.scatter(anomalies['Time (s)'], anomalies['Heart Rate (BPM)'], 
            color='red', s=70, zorder=5, 
            label=f'Anomalies (<{threshold_low} or >{threshold_high} BPM)')

# Add average heart rate reference line
avg_hr = df['Heart Rate (BPM)'].mean()
plt.axhline(y=avg_hr, color='green', linestyle='--', 
            label=f'Average: {avg_hr:.1f} BPM')

# Calculate and add heart rate variability (HRV)
hr_diff = np.diff(df['Heart Rate (BPM)'])
hrv = np.sqrt(np.mean(hr_diff**2))  # Simplified HRV calculation

# Add chart elements
plt.title('Heart Rate Over Time', fontsize=15, pad=20)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Heart Rate (BPM)', fontsize=12)
plt.grid(alpha=0.2, linestyle='--')
plt.legend(loc='upper right', framealpha=0.9)

# Add information text box
text_str = f"""Data Points: {len(df)}
Avg. Heart Rate: {avg_hr:.1f} BPM
Heart Rate Variability: {hrv:.2f} (RMSSD)
Frame Rate: {df['Frame Rate (FPS)'].iloc[0]} FPS"""
plt.figtext(0.75, 0.15, text_str, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
            fontsize=10)

# Optimize layout and display chart
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Save the chart
output_path = file_path.replace('.csv', '_heart_rate_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Heart rate plot saved to: {output_path}")

plt.show()