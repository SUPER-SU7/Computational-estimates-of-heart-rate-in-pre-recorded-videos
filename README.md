# Computational-estimates-of-heart-rate-in-pre-recorded-videos

# Heart Rate Estimation from Pre-recorded Videos

## Main Files:

`FFT.py` - Head tracking implementation  
`FFT01.py` - Heart rate detection via forehead tracking  
`FFT02.py` - Heart rate detection using forehead, cheeks, and nose features  
`FFT03.py` - Full-face heart rate detection (no specific feature tracking)  
`FFT04.py` -  can monitor heart rate by capturing faces through custom rectangles
Each script generates a `.csv` file containing timestamp and heart rate data when executed.
`FFT05.py` - 

`line.py` - Visualization tool that processes the generated `.csv` files and creates plots

## Video Processing Sequence:
-- The order of authorship for the videos follows the order in the database（video 1，2，3.....）





主要文件：
FFT.py 是一个头部跟踪文件
FFT01.py是通过额头检测心率的文件
FFT02.py是通过额头，脸颊，鼻子检测心率的文件
FFT03.py是通过全脸检测心率的文件，不涉及到抓取特征
每次执行他们各自都可以生成一个.csv文件，用于存储数据（时间，心率）
line.py 是一个执行文件，他可以通过抓取固定的.csv文件，并且将她们可视化
FFT04.py可以通过自定义矩形抓取面部从而完成心率监测
FFT05.py
g → 使用绿色通道进行心率估计
y → 使用灰度图像平均值
p → 使用 PCA 主成分作为信号
s → 保存当前心率数据为 CSV
q → 退出程序

在 main() 中添加一个配置参数 signal_mode，支持 "green", "gray", "pca"。
signal_mode = "green"  # Options: "green", "gray", "pca"