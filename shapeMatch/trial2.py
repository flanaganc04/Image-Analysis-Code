import cv2 as cv
from PreprocessMod import Preprocess
import numpy as np

referenceContour = np.load('shapeMatch/template_contour.npy')

imagePath = 'shapeMatch/DJI_20240530160604_0067_D.JPG'
binaryPath = 'shapeMatch/0067_Clean.JPG'

preprop = Preprocess(imagePath)
preprop.process_image(binaryPath)
binary = cv.imread(binaryPath)
print(binary.shape)

image = cv.imread(imagePath)

imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 225, 255, 0)

contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

matched_contours = []

for contour in contours:
    similarity = cv.matchShapes(referenceContour, contour, cv.CONTOURS_MATCH_I2, 0.0)
    
    if 12 < similarity < 13 and contour.size > 30:
        matched_contours.append(contour)
        print("Size: " + contour.size.__str__())
        print('Shape ' + contour.shape.__str__())
        print('Similarity: ' + similarity.__str__())
        cv.drawContours(binary, [contour], 0, (0,0,255), 3)
        
    

cv.imwrite(binaryPath,binary)


