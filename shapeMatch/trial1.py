import cv2 as cv
from Processing import Binarize, ContourProcessor, Draw, Calculator
import numpy as np

image = 'OT-2 Image analysis/shapeMatch/0067.JPG'
preprop = Binarize(image)
preprop.binarize_image('OT-2 Image analysis/shapeMatch/0067Binary.JPG')
binary = cv.imread('OT-2 Image analysis/shapeMatch/0067Binary.JPG')
print(f'Binary Image Shape:{binary.shape}')
binary = binary[875:950, 1350:1575]
cv.imwrite('OT-2 Image analysis/shapeMatch/0067Binary.JPG', binary)


img = cv.imread(image)
img = img[875:950, 1350:1575]
print(f'Crop Shape: {img.shape}')
image = 'OT-2 Image analysis/shapeMatch/0067Crop.JPG'
cv.imwrite(image, img)

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 230, 255, 0)
contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
        if 500 > contour.size > 120:
                print(f'Contour Size:{contour.size}')
                print(f'Contour Shape:{contour.shape}')


                processor = ContourProcessor(contour)
                filteredContour = processor.filter_by_y_level(0, 70)

                processor = ContourProcessor(filteredContour)
                filteredContour = processor.filter_by_x_level(0,225)

                drawer = Draw(binary)
                drawer.contour(filteredContour, 255, 0, 0)

                processor = ContourProcessor(filteredContour)
                left_contour, right_contour = processor.split_horizontal()

                processorL = ContourProcessor(left_contour)
                processorR = ContourProcessor(right_contour)
                left_contour, _ = processorL.split_horizontal()
                # _, right_contour = processorR.split_horizontal()

                drawer.contour(left_contour, 255, 220, 0)
                drawer.contour(right_contour, 255, 220, 0)

                calcL = Calculator(left_contour)
                calcR = Calculator(right_contour)

                angleL, slopeL, x, y, vx, vy = calcL.fit_line_and_calculate_angle()
                angleR, slopeR, a, b, va, vb = calcR.fit_line_and_calculate_angle()

                print(f'Angle Left:{angleL}')
                print(f'Angle Right:{angleR}')
                
                drawer.line_of_best_fit(x,y,vx,vy, 100)
                drawer.line_of_best_fit(a,b,va,vb, 100)

cv.imwrite('OT-2 Image analysis/shapeMatch/0067Drawn.png', binary)
# np.save('shapeMatch/template_contour.npy', referenceContour)



