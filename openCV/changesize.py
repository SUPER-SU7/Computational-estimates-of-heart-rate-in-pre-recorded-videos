import cv2 as cv
img= cv.imread('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/openCV/face.jpg')
resize_img=cv.resize(img,dsize=(100,400))
cv.imshow('resize_img',resize_img)

while True:
    if ord('q')==cv.waitKey(0):
           break

cv.destroyAllWindows()