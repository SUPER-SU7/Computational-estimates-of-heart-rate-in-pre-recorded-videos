import cv2 as cv
def fac_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载特征数据
    face_detect=cv.CascadeClassifier('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/XML/haarcascade_frontalface_default.xml')
    faces=face_detect.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0.205,0),thickness=2)
    cv.imshow('result',img)
img=cv.imread('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/openCV/face.jpg')
fac_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows()