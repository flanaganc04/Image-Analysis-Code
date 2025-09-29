
import cv2 as cv
import numpy as np

imagePath  = 'again/Sample1/frames/Sample1Frame2.png'
newPath = 'again/testImage.png'
image = cv.imread(imagePath)
image = image[700:750, 1600:2300]
cv.imwrite(newPath, image)
image = cv.imread(newPath)

imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 250, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if contour.size > 22 and contour.size < 80:
        print(contour.size)
        cv.drawContours(image, [contour], 0, (0,0,255), 2)
        cv.imwrite(newPath, image)
        referenceContour = contour
        np.save('again/referenceContour.npy',referenceContour)
