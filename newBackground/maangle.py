from Processing import Binarize, ContourProcessor, Calculator
import cv2 as cv

number = 3
path = f'OT-2 Image analysis/newBackground/{number}.JPG'

binarizer = Binarize(path)
binary_image = binarizer.process_image(f'OT-2 Image analysis/newBackground/{number}Bin.JPG')
print(f'Image Shape:{binary_image.shape}')
binary_image = binary_image[1950:2000,1050:1250]
cv.imwrite(f'OT-2 Image analysis/newBackground/{number}bincrop.JPG', binary_image)

binary_crop = cv.imread(f'OT-2 Image analysis/newBackground/{number}Bincrop.JPG')
print(f'Crop Shape:{binary_crop.shape}')

image = cv.imread(path)
print(f'Image Shape:{image.shape}')
image = image[1950:2000,1050:1250]
cv.imwrite(f'OT-2 Image analysis/newBackground/{number}crop.JPG', image)

crop = cv.imread(f'OT-2 Image analysis/newBackground/{number}crop.JPG')
print(f'Crop Shape:{crop.shape}')

# Convert the image to grayscale
imgray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
ret, thresh = cv.threshold(imgray, binarizer.global_thresh_value, 255, 0)

# Find contours in the thresholded image
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Save the thresholded image for reference
cv.imwrite(f'OT-2 Image analysis/newBackground/{number}thresh.JPG', thresh)

for contour in contours:
    if 300 > contour.size > 100:
        processor = ContourProcessor(contour)
        filtered_contour = processor.filter_by_y_level(0, 20)
        cv.drawContours(binary_crop, [filtered_contour], -1, (255,0,0), 2)

        processor = ContourProcessor(filtered_contour)
        left_contour, right_contour = processor.split_horizontal()

        processorL = ContourProcessor(left_contour)
        processorR = ContourProcessor(right_contour)
        left_contour, _ = processorL.split_horizontal()
        _, right_contour = processorR.split_horizontal()
        cv.drawContours(binary_crop, [left_contour], -1, (255,240,0), 2)
        cv.drawContours(binary_crop, [right_contour], -1, (255,240,0), 2)

        print('Left')
        calculatorL = Calculator(left_contour)
        degreesL, slopeL, x_single, y_single, vx_single, vy_single = calculatorL.fit_line_and_calculate_angle()

        print('Right')
        calculatorR = Calculator(right_contour)
        degreesR, slopeR, a_single, b_single, va_single, vb_single = calculatorR.fit_line_and_calculate_angle()

        


cv.imwrite(f'OT-2 Image analysis/newBackground/{number}final.JPG', binary_crop)

