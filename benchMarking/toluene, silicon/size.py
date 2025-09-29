import cv2 as cv

img = cv.imread('OT-2 Image analysis/benchMarking/toluene, silicon/Toluene.Silicon2.JPG')
print(img.shape)
img = img[1900:2000, 825:1200]

cv.imwrite("OT-2 Image analysis/benchMarking/toluene, silicon/crop.png", img)
