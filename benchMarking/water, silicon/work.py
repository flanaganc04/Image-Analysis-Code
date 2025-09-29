import cv2 as cv
import os
import sys

imagePath = 'OT-2 Image analysis/benchMarking/water, silicon/h20.3.2.BMP'

if os.path.exists(imagePath):
    image = cv.imread(imagePath)
else: 
    print('Path to image does not exist, check path, check file, check image')
    sys.exit()

print(image.shape)
print(image.size)

imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 20, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.imwrite('OT-2 Image analysis/benchMarking/water, silicon/h20.3.2.thresh.BMP', thresh)

for contour in contours:
    if contour.size > 200:
        print(contour.shape)
        print(contour.size)
        x,y,w,h = cv.boundingRect(contour)
        print(x,y,w,h)
        drawn = cv.drawContours(image, [contour], 0, (255,0,0), 2) #, isClosed = False)

cv.imwrite('OT-2 Image analysis/benchMarking/water, silicon/h20.3.2.Drawing.BMP', drawn)