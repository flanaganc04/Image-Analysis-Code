import cv2 as cv
import math
from shapeMatch.PreprocessMod import Preprocess
    
path = 'Square Test\\square.png'
img = cv.imread(path)

prepop = Preprocess(path)
prepop.process_image('Square Test\\squareBinary.png')
binary = cv.imread('Square Test\\squareBinary.png')

cv.imwrite('Square Test\\squareBinaryCrop.png', binary)
binary = cv.imread('Square Test\\squareBinaryCrop.png',)

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour1 = contours[0]
contour2 = contours[1]
contour3 = contours[2]
M = cv.moments(contour2)
cx = int(M['m10'] / M['m00'])
right_contour = contour2[contour2[:, :, 0] > cx]

vx, vy, x, y = cv.fitLine(right_contour, cv.DIST_L2, 0, 0.01, 0.01)

x_single = x[0]
y_single = y[0]
vx_single = vx[0]
vy_single = vy[0]

slope = vy_single/vx_single
radians = math.atan(slope)
degrees = (radians*180)/math.pi
print(f"Slope of the line is: {slope}")
print(f"Contact angle is: {degrees}")

drawn_right = cv.drawContours(binary, [right_contour], 0, (0, 0, 255), 2)
cv.line(binary, (int(x_single), int(y_single)), (int(x_single+300*vx_single),int(y_single+300*vy_single)), (0, 215, 255), 2)

cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 2)

cv.imwrite("Square Test\\SquareAngle.png",drawn_right)








