import cv2 as cv
img=cv.imread('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/openCV/face.jpg')
#将图片灰度转化
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('BGR_img',gray_img)
#保存图片
cv.imwrite('gray_face.jpg',gray_img)

cv.waitKey(0)
cv.destroyAllWindows()