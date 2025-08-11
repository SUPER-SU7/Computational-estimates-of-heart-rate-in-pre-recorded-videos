import cv2 as cv

# 读取图片
img = cv.imread('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/openCV/face.jpg')  # 请确认 face.jpg 文件存在，且路径正确

# 判断是否成功读取图片
if img is None:
    print("无法读取图像，请检查文件是否存在或路径是否正确。")
else:
    # 显示图片
    cv.imshow('read_img', img)
    # 等待按键
    cv.waitKey(0)
    # 释放资源
    cv.destroyAllWindows()





    