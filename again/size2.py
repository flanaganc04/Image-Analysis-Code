import cv2
imagePath = "PVA100000Toluene/Sample1/frames/Sample1Frame1.png"
img = cv2.imread(imagePath)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 250, 250, 0)
contours, mat = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if contour.size < 100 and contour.size > 40:
        print(contour.size)
        print(contour.shape)
        cv2.drawContours(img, [contour], 0, (0,0,255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()    

