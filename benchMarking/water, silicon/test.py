import cv2 as cv
import numpy as np
import os
import sys
import math

def ensure_2d_contour(contour):
    """
    Ensure the contour is a 2D array of points.
    
    Parameters:
        contour (numpy.ndarray): Input contour.
        
    Returns:
        numpy.ndarray: Reshaped contour as a 2D array if necessary.
    """
    if contour.ndim == 3:
        return contour.reshape(-1, 2)
    elif contour.ndim == 2 and contour.shape[1] == 2:
        return contour
    else:
        print("Warning: Contour has an unexpected shape")
        return np.array([])

def filter_contour_by_y_level(contour, y_level_min, y_level_max):
    """
    Filters out points in a contour that are outside a specified y-level range.

    Parameters:
        contour (numpy.ndarray): Input contour, should be a 2D array of points.
        y_level_min (int): The minimum y-level threshold; points below this level will be excluded.
        y_level_max (int): The maximum y-level threshold; points above this level will be excluded.

    Returns:
        numpy.ndarray: The filtered contour with points only within the specified y-level range.
    """
    contour = ensure_2d_contour(contour)
    if contour.size == 0:
        return contour

    # Filter points based on the specified y-level range
    filtered_contour = contour[(contour[:, 1] >= y_level_min) & (contour[:, 1] <= y_level_max)]
    return filtered_contour

def split_contour_horizontal(contour):
    """
    Splits a contour into left and right parts based on the x-coordinate of its centroid.
    
    Parameters:
        contour (numpy.ndarray): Input contour.

    Returns:
        tuple: Two contours, left and right.
    """
    contour = ensure_2d_contour(contour)
    if contour.size == 0:
        return np.array([]), np.array([])  # Return empty contours if invalid

    M = cv.moments(contour)
    if M['m00'] == 0:
        return np.array([]), np.array([])  # Return empty contours if invalid

    cx = int(M['m10'] / M['m00'])  # Center x-coordinate

    contour_points = contour.reshape(-1, 2)

    left_contour = contour_points[contour_points[:, 0] <= cx]
    right_contour = contour_points[contour_points[:, 0] >= cx]

    return left_contour.reshape(-1, 1, 2), right_contour.reshape(-1, 1, 2)

# Sample value
sample = '7.1'

# Define the path to the image
image_path = f'OT-2 Image analysis/benchMarking/water, silicon/h20.{sample}.BMP'

# Check if the image path exists
if not os.path.exists(image_path):
    print('Path to image does not exist. Please check the path, file, and image.')
    sys.exit()

# Read the image from the specified path
image = cv.imread(image_path)

# Print the shape and size of the image for debugging purposes
print(f"Image shape: {image.shape}")
print(f"Image size: {image.size}")

# Convert the image to grayscale
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
ret, thresh = cv.threshold(imgray, 100, 255, 0)

# Find contours in the thresholded image
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Save the thresholded image for reference
cv.imwrite(f'OT-2 Image analysis/benchMarking/water, silicon/h20.{sample}.thresh.BMP', thresh)

# Process each contour found in the image
for contour in contours:
    contour = ensure_2d_contour(contour)
    if contour.size > 200:  # Process only if the contour has more than 200 points
        print(f"Original contour shape: {contour.shape}")
        # Filter the contour points based on the specified y-level range

        filtered_contour = filter_contour_by_y_level(contour, 100, 380)

        # If there are points left after filtering, draw the contour on the image
        if filtered_contour.size > 0:
            approx_contour = cv.approxPolyDP(filtered_contour, 0.5, False)
            cv.drawContours(image, [approx_contour], -1, (255, 0, 0), 2)
            
            # Split the contour horizontally multiple times
            left_contour, right_contour = split_contour_horizontal(approx_contour)
            left_contour, _ = split_contour_horizontal(left_contour)
            _, right_contour = split_contour_horizontal(right_contour)
            left_contour, _ = split_contour_horizontal(left_contour)
            _, right_contour = split_contour_horizontal(right_contour)
            left_contour, _ = split_contour_horizontal(left_contour)
            _, right_contour = split_contour_horizontal(right_contour)
            
            # Draw the split contours
            cv.drawContours(image, [left_contour], -1, (255, 240, 0), 2)
            cv.drawContours(image, [right_contour], -1, (255, 240, 0), 2)

            # Fit lines and calculate angles for the left and right parts
            if len(left_contour) > 0:
                vx, vy, x, y = cv.fitLine(left_contour, cv.DIST_L2, 0, 0.1, 0.1)
                x_single, y_single = x[0], y[0]
                vx_single, vy_single = vx[0], vy[0]
                slope_left = -vy_single / vx_single
                radians_left = math.atan(slope_left)
                degrees_left = (radians_left * 180) / math.pi
            else:
                degrees_left = 'N/A'

            if len(right_contour) > 0:
                va, vb, a, b = cv.fitLine(right_contour, cv.DIST_L2, 0, 0.1, 0.1)
                a_single, b_single = a[0], b[0]
                va_single, vb_single = va[0], vb[0]
                slope_right = vb_single / va_single
                radians_right = math.atan(slope_right)
                degrees_right = (radians_right * 180) / math.pi
            else:
                degrees_right = 'N/A'

            # Calculate the mean angle
            mean_angle = (degrees_left + degrees_right) / 2 if degrees_left != 'N/A' and degrees_right != 'N/A' else 'N/A'
            print(f'Left Angle: {degrees_left}')
            print(f'Right Angle: {degrees_right}')
            print(f'Mean Angle: {mean_angle}')

            # Draw lines for visualization
            cv.line(image, (int(x_single), int(y_single)), (int(x_single+100*vx_single),int(y_single + 100*vy_single)), (0, 0, 255), 2)

            cv.line(image, (int(x_single), int(y_single)), (int(x_single+-100*vx_single),int(y_single+-100*vy_single)), (0, 0, 255), 2)

            cv.line(image, (int(a_single), int(b_single)), (int(a_single+100*va_single),int(b_single+100*vb_single)), (0, 0, 255), 2)

            cv.line(image, (int(a_single), int(b_single)), (int(a_single+-100*va_single),int(b_single+-100*vb_single)), (0, 0, 255), 2)

# Save the image with the drawn contours for reference
cv.imwrite(f'OT-2 Image analysis/benchMarking/water, silicon/h20.{sample}.filtered.BMP', image)
