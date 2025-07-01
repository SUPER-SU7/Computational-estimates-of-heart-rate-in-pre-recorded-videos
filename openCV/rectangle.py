import cv2 as cv
img=cv.imread('/Users/suziteng/Documents/GitHub/Computational-estimates-of-heart-rate-in-pre-recorded-videos/openCV/face.jpg')
#左上角的坐标（x，y）矩形的宽度和高度（w，h）
x,y,w,h=100,100,100,100
cv.rectangle(img,(x,y,x+w,y+h),color=(0,255,0),thickness=2)
x,y,r=200,200,100
cv.circle(img,center=(x,y),radius=r,color=(0,0,255),thickness=2)
#显示图片
cv.imshow('rectangle_img',img)
cv.waitKey(0)
cv.destroyAllWindows()