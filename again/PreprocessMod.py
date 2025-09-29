import cv2
import numpy as np

class Preprocess:
    def __init__(self, directory):
        self.global_thresh_value = 245
        self.adaptive_thresh_window_size = 25
        self.adaptive_thresh_C = 2
        self.morph_kernel_size = 5
        self.morph_iterations = 1
        self.directory = directory
    def read_image(self):
        self.image_path = f'{self.directory}'
        return cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
    def global_threshold(self, image):
        _, global_thresh_mask = cv2.threshold(image, self.global_thresh_value, 255, cv2.THRESH_BINARY)
        return global_thresh_mask
    def save_cleaned_image(self, cleaned_image, output_path):
        cv2.imwrite(output_path, cleaned_image)
    def process_image(self, output_path):
        image = self.read_image()
        global_mask = self.global_threshold(image)
        self.save_cleaned_image(global_mask, output_path)
        return image




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

    M = cv2.moments(contour)
    if M['m00'] == 0:
        return np.array([]), np.array([])  # Return empty contours if invalid

    cx = int(M['m10'] / M['m00'])  # Center x-coordinate

    contour_points = contour.reshape(-1, 2)

    left_contour = contour_points[contour_points[:, 0] <= cx]
    right_contour = contour_points[contour_points[:, 0] >= cx]

    return left_contour.reshape(-1, 1, 2), right_contour.reshape(-1, 1, 2)

