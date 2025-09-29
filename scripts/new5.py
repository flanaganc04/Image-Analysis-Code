import cv2 as cv
import math
from shapeMatch.PreprocessMod import Preprocess
    
path = 'C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples\\Sample 1\\frames\\Sample1Frame1.png'
img = cv.imread(path)

prepop = Preprocess(path)
prepop.process_image('C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples\\Sample 1\\binarized_frames\\Binary.png')
binary = cv.imread('C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples\\Sample 1\\binarized_frames\\Binary.png')
binary = binary[1110:1127, 1900:2350] #whole drop
cv.imwrite('C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples\\Sample 1\\binarized_frames\\BinaryCrop.png', binary)
binary = cv.imread('C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples\\Sample 1\\binarized_frames\\BinaryCrop.png')

crop = img[1110:1127, 1900:2350] #whole drop
newPath = 'C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples\\Sample 1\\Sample1Frame1Cleaned.png'
cv.imwrite(newPath, crop)

newimg = cv.imread(newPath)
imgray = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour = contours[0]
M = cv.moments(contour)
cx = int(M['m10'] / M['m00'])
# left_contour = contour[contour[:, :, 0] <= cx]
right_contour = contour[contour[:, :, 0] > cx]
J= cv.moments(right_contour)
cx = int(J['m10']/J['m00'])
new_right_contour = right_contour[right_contour[:,0] > cx]
H= cv.moments(new_right_contour)
cx = int(H['m10']/H['m00'])
newer_right_contour = new_right_contour[new_right_contour[:,0] > cx]
Z= cv.moments(newer_right_contour)
cx = int(H['m10']/H['m00'])
newerer_right_contour = newer_right_contour[newer_right_contour[:,0] > cx]


for contour in contours:
    vx, vy, x, y = cv.fitLine(newerer_right_contour, cv.DIST_L2, 0, 0.01, 0.01)
    # a, b, va, vb = cv.fitLine(left_contour, cv.DIST_L2, 0, 0.01, 0.01)

    x_single = x[0]
    y_single = y[0]
    vx_single = vx[0]
    vy_single = vy[0]

    
    slope = vy_single/vx_single
    radians = math.atan(slope)
    degrees = (radians*180)/math.pi
    print(f"Slope of the line is: {slope}")
    print(f"Contact angle is: {degrees}")
    

    cv.line(binary, (int(x_single), int(y_single)), (int(x_single+300*vx_single),int(y_single+300*vy_single)), (0, 215, 255), 2)

    cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 2)
    # cv.line(binary, (int(a_single), int(b_single)), (int(a_single+va_single),int(b_single+vb_single)), (255, 0, 0), 2)
    # drawn_left = cv.drawContours(binary, [left_contour], 0, (0, 215, 255), 2)
    drawn_right = cv.drawContours(binary, [newerer_right_contour], 0, (0, 0, 255), 2)

    cv.imwrite("Contours.png",drawn_right)
    







