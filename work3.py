import cv2 as cv
import numpy as np

referenceContour = np.load('again/referenceContour.npy')

newImage = 'again/Sample7/frames/Sample7Frame8.png'
cropPath = 'again/Sample7/frames/Sample7Frame8Crop.png'
image = cv.imread(newImage)
image = image[700:750, 1600:2300]
cv.imwrite(cropPath, image)
image = cv.imread(cropPath)

imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 250, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
        similarity = cv.matchShapes(referenceContour, contour, cv.CONTOURS_MATCH_I2, 0.0)

        if similarity < 13:
            cv.drawContours(image, [contour], 0, (0,0,255), 2)
            cv.imwrite(cropPath, image)
            print(contour.size)
            print(similarity)