import cv2 as cv
import math
from shapeMatch.PreprocessMod import Preprocess
    
path = "chloronap, silicon\\chloronap.silicon3.JPG"
img = cv.imread(path)
print(img.shape)

preprop = Preprocess(path)
preprop.process_image("chloronap, silicon\\binary.png")
binary = cv.imread("chloronap, silicon\\binary.png")
binary = binary[975:1075, 925:1125]

crop = img[975:1075, 925:1125]

imgray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 250, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(contours)

# print(img.shape)


for contour in contours:
    if contour.size > 214:
        print(contour.shape)
        print(contour.size)
        # cv.drawContours(binary, [contour], 0, (0, 0, 255), 2)
        contour1 = contour[38:46]
        contour2= contour[109:122]
        cv.drawContours(binary, [contour1], 0, (0, 0, 255), 2)
        cv.drawContours(binary, [contour2], 0, (0, 0, 255), 2)
       
        vx, vy, x, y  = cv.fitLine(contour1, cv.DIST_L2, 0, 0.01, 0.01)

        x_single = x[0]
        y_single = y[0]
        vx_single = vx[0]
        vy_single = vy[0]

        va, vb, a, b  = cv.fitLine(contour2, cv.DIST_L2, 0, 0.01, 0.01)

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
        cv.line(binary, (int(x_single), int(y_single)), (int(x_single+50*vx_single),int(y_single + 50*vy_single)), (0, 215, 255), 1)

        cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 1)

        cv.line(binary, (int(a_single), int(b_single)), (int(a_single+200*va_single),int(b_single+200*vb_single)), (0, 215, 255), 1)

        cv.line(binary, (int(a_single), int(b_single)), (int(a_single+-50*va_single),int(b_single+-50*vb_single)), (0, 215, 255), 1)

        
cv.imwrite("chloronap, silicon\\angles3.png", binary)
print(f"Slope of the left line is: {slope_left}")
print(f"Slope of the right line is: {slope_right}")
print(f'Contact angle left is: {degrees_left}') 
print(f"Contact angle right is: {degrees_right}")
print(f'Contact angle mean: {mean}')

 








    







