import os
from moviepy.editor import VideoFileClip
import cv2 as cv
import math
from PreprocessMod import Preprocess
import sys
import csv
import shutil
import numpy as np

referenceContour = np.load('again/referenceContour.npy')
bigData = []
fieldnames = ['Polymer','Molecular Weight', 'Solvent', 'Sample', 'Frame', 'Concentration', 'Drop Volume', 'Contact Angle L', 'Contact Angle R', 'Contact Angle M'] 


print(f'Processing Frame')
path = f'graphics/Nice Drop.JPG'
cropPath = f'graphics/Nice Drop Crop.JPG'
binaryPath = f'graphics/Nice Drop Bin.JPG'
drawnPath = f'graphics/Nice Drop Drawn.JPG'
binaryCropPath = f'graphics/Nice Drop Crop Bin.JPG'

frame = cv.imread(path)
print(frame.shape)
frame = frame[900:1000, 1375:1560]
cv.imwrite(cropPath,frame)
frame = cv.imread(cropPath)

binary = cv.imread(binaryPath)
binary = binary[900:1000, 1375:1560]
cv.imwrite(binaryCropPath, binary)

imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 250, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

possibleDrops = []

for contour in contours:
    similarity = cv.matchShapes(referenceContour, contour, cv.CONTOURS_MATCH_I2, 0)
    # print("similarity: " + similarity.__str__())
    # print('size: '+contour.size.__str__())

    

    if similarity < 20 and contour.size > 200:

        print(similarity)
        possibleDrops.append(contour)
        
        print(contour.size)

print('Number of possible drops' + len(possibleDrops).__str__())
if len(possibleDrops) == 1:
        loneContour = possibleDrops[0]
        print(loneContour.size)
        

        M = cv.moments(loneContour)
        cx = int(M['m10'] / M['m00'])
        rightContour = loneContour[loneContour[:, :, 0] >= cx]
        leftContour = loneContour[loneContour[:, :, 0] <= cx]
        cv.drawContours(binary, [contour], 0, (255,0,0), 2)
        vx, vy, x, y  = cv.fitLine(leftContour, cv.DIST_L2, 0, 0.01, 0.01)

        x_single = x[0]
        y_single = y[0]
        vx_single = vx[0]
        vy_single = vy[0]

        va, vb, a, b  = cv.fitLine(rightContour, cv.DIST_L2, 0, 0.01, 0.01)

        a_single = a[0]
        b_single = b[0]
        va_single = va[0]
        vb_single = vb[0]

        slope_left = -vy_single/vx_single
        slope_right = vb_single/va_single

        radians_left = math.atan(slope_left)
        radians_right = math.atan(slope_right)

        degrees_left = (radians_left*180)/math.pi
        degrees_right = (radians_right*180)/math.pi
        mean = (degrees_right + degrees_left)/2


        # draw the line of best fit
        cv.line(binary, (int(x_single), int(y_single)), (int(x_single+40*vx_single),int(y_single + 40*vy_single)), (0, 0, 255), 2)

        cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-100*vx_single),int(y_single+-100*vy_single)), (0, 0, 255), 2)

        cv.line(binary, (int(a_single), int(b_single)), (int(a_single+100*va_single),int(b_single+100*vb_single)), (0, 0, 255), 2)

        cv.line(binary, (int(a_single), int(b_single)), (int(a_single+-40*va_single),int(b_single+-40*vb_single)), (0, 0, 255), 2)

      

cv.imwrite(drawnPath, binary)
    
       

    

