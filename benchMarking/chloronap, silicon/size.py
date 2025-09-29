import cv2 as cv

img = cv.imread("chloronap, silicon\\chloronap.silicon1.JPG")
print(img.shape)
img = img[975:1075, 925:1125]

cv.imwrite('chloronap, silicon\\crop.png',img)
