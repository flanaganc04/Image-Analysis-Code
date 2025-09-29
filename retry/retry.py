import cv2 as cv
from shapeMatch.PreprocessMod import Preprocess

# create a binary image to draw on
preprop = Preprocess("retry\\Sample1Frame1.png")
preprop.process_image("retry\\Sample1Frame1Binary.png")
binary = cv.imread("retry\\Sample1Frame1Binary.png")
binary = binary[1110:1127, 1900:2350]


# detect the contour of the drop
newimg = cv.imread("retry\\Sample1Frame1.png")
newimg = newimg[1110:1127, 1900:2350]
imgray = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour = contours[0]

# find the liquid, vapor, substrate interface
M = cv.moments(contour)
cx = int(M['m10'] / M['m00'])
right_contour = contour[contour[:, :, 0] >= cx]
M2 = cv.moments(right_contour)
cx = int(M2['m10'] / M2['m00'])
newRight_contour = right_contour[right_contour[:, 0] >= cx]
M3 = cv.moments(newRight_contour)
cx = int(M3['m10'] / M3['m00'])
new2Right_contour = newRight_contour[newRight_contour[:, 0] >= cx]

# draw the spot the computer is recognizing as the interface
cv.drawContours(binary, [new2Right_contour] , 0 , (0,0,255), 2)
vx,vy,x,y = cv.fitLine(new2Right_contour, cv.DIST_L2, 0, 0.01, 0.01)

x_single = x[0]
y_single = y[0]
vx_single = vx[0]
vy_single = vy[0]

# draw the line of best fit
cv.line(binary, (int(x_single), int(y_single)), (int(x_single+300*vx_single),int(y_single+300*vy_single)), (0, 215, 255), 2)

cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 2)

cv.imwrite('retry\\Sample1Frame1Binary.png', binary)
