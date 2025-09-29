import cv2 as cv
import math
from shapeMatch.PreprocessMod import Preprocess
    
path = 'Triangle Test\\triangle.jpg'
img = cv.imread(path)

prepop = Preprocess(path)
prepop.process_image('Triangle Test\\triangleBinary.jpg')
binary = cv.imread('Triangle Test\\triangleBinary.jpg')

cv.imwrite("Triangle Test\\triagleBinaryCrop.jpg", binary)
binary = cv.imread("Triangle Test\\triagleBinaryCrop.jpg")

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(contours)
contour1 = contours[5]

M = cv.moments(contour1)
cx = int(M['m10'] / M['m00'])
left_contour = contour1[contour1[:, :, 0] <= cx]
right_contour = contour1[contour1[:, :, 0] > cx]

vx, vy, x, y = cv.fitLine(contour1, cv.DIST_L2, 0, 0.01, 0.01)

x_single = x[0]
y_single = y[0]
vx_single = vx[0]
vy_single = vy[0]

slope = vy_single/vx_single
radians = math.atan(slope)
degrees = (radians*180)/math.pi
print(f"Slope of the line is: {slope}")
print(f"Contact angle is: {degrees}")

drawn_left = cv.drawContours(binary, [contour1], 0, (0, 0, 255), 2)
cv.line(binary, (int(x_single), int(y_single)), (int(x_single+300*vx_single),int(y_single+300*vy_single)), (0, 215, 255), 2)

cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 2)

cv.imwrite("Triangle Test\\triangleAngle.png",drawn_left)









    







