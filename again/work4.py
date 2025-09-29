import os
from moviepy.editor import VideoFileClip
import cv2 as cv
import math
from PreprocessMod import Preprocess
import sys
import csv
import shutil
import numpy as np

referenceContour = np.load('OT-2 Image analysis/again/referenceContour.npy')
bigData = []
fieldnames = ['Polymer','Molecular Weight', 'Solvent', 'Surface Roughness','Concentration', 'Drop Volume', 'Contact Angle L', 'Contact Angle R', 'Contact Angle M'] 

for j in range(12):
    print(f"Processing Sample {j+1}")
    for n in range(60):
        print(f'Processing Frame {n+1}')
        path = f'OT-2 Image analysis/again/Sample{j+1}/frames/Sample{j+1}Frame{n+1}.png'
        cropPath = f'OT-2 Image analysis/again/Sample{j+1}/crops/Sample{j+1}Frame{n+1}.png'
        binaryPath = f'OT-2 Image analysis/again/sample{j+1}/binary/Sample{j+1}Frame{n+1}Binary.png'
        drawnPath = f'OT-2 Image analysis/again/sample{j+1}/drawn/Sample{j+1}Frame{n+1}drawn.png'

        frame = cv.imread(path)
        print(frame.shape)
        frame = frame[700:750, 1600:2300]
        cv.imwrite(cropPath,frame)
        frame = cv.imread(cropPath)

        binary = cv.imread(binaryPath)
        binary = binary[700:750, 1600:2300]

        imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 250, 255, 0)
        contours, mat = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            data = {}
            data['Polymer'] = 'PVA'
            data['Molecular Weight'] = '100,000'
            data['Solvent'] = 'Toluene'
            data['Surface Roughness'] = 'R'
            # data['Sample'] = f'{j+1}'
            # data['Frame'] = f'{n+1}'

            if j in [0, 1, 2, 3]:
                data["Concentration"] = "0.25"
            elif j in [4, 5, 6, 7]:
                data["Concentration"] = "0.15"
            elif j in [8, 9, 10, 11]:
                data["Concentration"] = "0.05"

            if j in [0, 4, 8]:
                data['Drop Volume'] = "7"
            elif j in [1, 5, 9]:
                data['Drop Volume'] = "5"
            elif j in [2, 6, 10]:
                data['Drop Volume'] = "3"
            elif j in [3, 7, 11]:
                data['Drop Volume'] = "1"

            

            similarity = cv.matchShapes(referenceContour, contour, cv.CONTOURS_MATCH_I2, 0)
            # print("similarity: " + similarity.__str__())
            # print('size: '+contour.size.__str__())

            possibleDrops = []

            if similarity < 20 and contour.size > 50:

                print(similarity)
                possibleDrops.append(contour)
                print('Number of possible drops' + len(possibleDrops).__str__())
            else:
                data['Contact Angle L'] = f'N/A'
                data['Contact Angle R'] = f'N/A'
                data['Contact Angle M'] = f'N/A'

            if len(possibleDrops) == 1:
                loneContour = possibleDrops[0]
                print(loneContour.size)
                cv.drawContours(binary, [loneContour], 0, (255,0,0), 2)

                M = cv.moments(loneContour)
                cx = int(M['m10'] / M['m00'])
                rightContour = loneContour[loneContour[:, :, 0] >= cx]
                leftContour = loneContour[loneContour[:, :, 0] <= cx]

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
                cv.line(binary, (int(x_single), int(y_single)), (int(x_single+100*vx_single),int(y_single + 100*vy_single)), (0, 0, 255), 2)

                cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-100*vx_single),int(y_single+-100*vy_single)), (0, 0, 255), 2)

                cv.line(binary, (int(a_single), int(b_single)), (int(a_single+100*va_single),int(b_single+100*vb_single)), (0, 0, 255), 2)

                cv.line(binary, (int(a_single), int(b_single)), (int(a_single+-100*va_single),int(b_single+-100*vb_single)), (0, 0, 255), 2)

                data['Contact Angle L'] = f'{degrees_left}'
                data['Contact Angle R'] = f'{degrees_right}'
                data['Contact Angle M'] = f'{mean}'

                cv.imwrite(drawnPath, binary)
                if mean > 0:
                    bigData.append(data)
            
                
            


file_name = 'OT-2 Image analysis/again/PVA100000Toluene1.csv'
with open(file_name, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(bigData)
