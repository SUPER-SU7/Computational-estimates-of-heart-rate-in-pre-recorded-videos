import cv2 as cv
def fac_detect_demo(img):
 gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载特征数据
 face_detect=cv.CascadeClassifier('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/XML/haarcascade_frontalface_default.xml')
 faces=face_detect.detectMultiScale(gray)
 for x,y,w,h in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
 cv.imshow('result',img)

#读取视频
cap=cv.VideoCapture('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/face.mp4')
while True:#循环
    flag,frame=cap.read()
    if not flag:
        break
    fac_detect_demo(frame)
    if ord('q') ==cv.waitKey(10000000):
      break

cv.destroyAllWindows()
cap.release()