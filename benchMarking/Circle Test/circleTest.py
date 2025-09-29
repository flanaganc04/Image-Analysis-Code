from shapeMatch.PreprocessMod import Preprocess
import cv2 as cv
import math

path = 'Circle Test\circle.png'
invert = 'Circle Test\circleInvertedBinaryCrop.png'
finalPath = 'Circle Test\circleAngle.png'


img = cv.imread(path)
invert = cv.imread(invert)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour = contours[1]
contour = contour[0:290, 0:1]

cv.drawContours(img,[contour], 0, (0,0,255), 2)
img = img[170:215, 0:255]
cv.imwrite("new.png",img)

path = 'new.png'
img = cv.imread(path)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour = contours[0]
print(contour.shape)
M = cv.moments(contour)
cx = int(M['m10'] / M['m00'])
left_contour = contour[contour[:, :, 0] <= cx]
print(left_contour.shape)
left_contour = left_contour[30:38]
J= cv.moments(left_contour)
cx = int(J['m10']/J['m00'])
newLeft_contour = left_contour[left_contour[:,0] <= cx]
vx, vy, x, y = cv.fitLine(left_contour, cv.DIST_L2, 0, 0.01, 0.01)
x_single = x[0]
y_single = y[0]
vx_single = vx[0]
vy_single = vy[0]

cv.line(img, (int(x_single), int(y_single)), (int(x_single+300*vx_single),int(y_single+300*vy_single)), (0, 215, 255), 2)
cv.line(img, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 2)

cv.drawContours(img,[newLeft_contour], 0, (0,255,0), 2)
print(left_contour.shape)

slope = vy_single/vx_single
radians = math.atan(slope)
degrees = (radians*180)/math.pi
print(f"Slope of the line is: {slope}")
print(f"Contact angle is: {180+ degrees}")

cv.imwrite(finalPath,img)


