import cv2
import numpy as np

path = 'PVA100000Toluene/Sample1/frames/Sample1Frame2.png'

img = cv2.imread(path)
print(img.shape) 
# crop = img[1110:1127, 2200:2350] #right side of drop
# crop = img[1110:1127,1900:2200] #left side of drop
#whole drop
img = img[700:750, 1600:2300]

cv2.imwrite("scripts/crop.png", img)
cv2.waitKey(0)



