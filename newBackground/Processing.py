import cv2
import numpy as np
import math

class Binarize:
    def __init__(self, directory):
        """
        Initialize the Binarize class with parameters for image processing.
        
        Parameters:
            directory (str): The path to the directory where the image is located.
        """
        self.global_thresh_value = 185
        self.adaptive_thresh_window_size = 25
        self.adaptive_thresh_C = 2
        self.directory = directory
    
    def read_image(self):
        """
        Reads an image from the specified directory path.
        
        Returns:
            numpy.ndarray: The grayscale image read from the file.
        
        Raises:
            FileNotFoundError: If the image file does not exist at the specified path.
        """
        self.image_path = f'{self.directory}'
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        return image
    
    def global_threshold(self, image):
        """
        Applies a global threshold to the provided image.
        
        Parameters:
            image (numpy.ndarray): The input grayscale image.
        
        Returns:
            numpy.ndarray: The binary image after applying the global threshold.
        """
        _, global_thresh_mask = cv2.threshold(image, self.global_thresh_value, 255, cv2.THRESH_BINARY)
        return global_thresh_mask
    
   
    def save_cleaned_image(self, cleaned_image, output_path):
        """
        Saves the cleaned image to the specified output path.
        
        Parameters:
            cleaned_image (numpy.ndarray): The image to be saved.
            output_path (str): The path where the image will be saved.
        
        Raises:
            IOError: If the image cannot be saved at the specified path.
        """
        success = cv2.imwrite(output_path, cleaned_image)
        if not success:
            raise IOError(f"Failed to save image at {output_path}")
    
    def process_image(self, output_path):
        """
        Processes the image by applying global thresholding and morphological operations,
        and then saves the result.
        
        Parameters:
            output_path (str): The path where the processed image will be saved.
        
        Returns:
            numpy.ndarray: The original grayscale image.
        """
        image = self.read_image()
        global_mask = self.global_threshold(image)
        self.save_cleaned_image(global_mask, output_path)
        return image




class ContourProcessor:
    def __init__(self, contour):
        """
        Initialize the ContourProcessor with a contour.
        
        Parameters:
            contour (numpy.ndarray): Input contour.
        """
        self.contour = self.ensure_2d_contour(contour)
    
    def ensure_2d_contour(self, contour):
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

    def filter_by_y_level(self, y_level_min, y_level_max):
        """
        Filters out points in the contour that are outside a specified y-level range.

        Parameters:
            y_level_min (int): The minimum y-level threshold; points below this level will be excluded.
            y_level_max (int): The maximum y-level threshold; points above this level will be excluded.

        Returns:
            numpy.ndarray: The filtered contour with points only within the specified y-level range.
        """
        if self.contour.size == 0:
            print("Warning: Contour is empty after ensuring 2D format")
            return self.contour

        # Filter points based on the specified y-level range
        filtered_contour = self.contour[(self.contour[:, 1] >= y_level_min) & (self.contour[:, 1] <= y_level_max)]
        return filtered_contour

    def split_horizontal(self):
        """
        Splits the contour into left and right parts based on the x-coordinate of its centroid.
        
        Returns:
            tuple: Two contours, left and right.
        """
        if self.contour.size == 0:
            print("Warning: Contour is empty or invalid")
            return np.array([]), np.array([])  # Return empty contours if invalid

        M = cv2.moments(self.contour)
        if M['m00'] == 0:
            print("Warning: Contour moments are zero")
            return np.array([]), np.array([])  # Return empty contours if invalid

        cx = int(M['m10'] / M['m00'])  # Center x-coordinate

        self.contour = self.contour.reshape(-1, 2)

        left_contour = self.contour[self.contour[:, 0] <= cx]
        right_contour = self.contour[self.contour[:, 0] >= cx]

        return left_contour.reshape(-1, 1, 2), right_contour.reshape(-1, 1, 2)
    

class Calculator:
    def __init__(self, contour):
        """
        Initialize the Calculator with a contour.
        
        Parameters:
            contour (numpy.ndarray): Input contour.
        """
        self.contour = self.ensure_2d_contour(contour)
        self.image = None
    
    def ensure_2d_contour(self, contour):
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
        
    def fit_line_and_calculate_angle(self):
        """
        Fits a line to the contour and calculates the angle of the line relative to the x-axis.
        
        The function uses OpenCV's `cv2.fitLine` to fit a line to the contour points, computes the slope,
        and converts this slope to an angle in degrees. The method also handles negative slopes by
        correcting them if necessary.

        Returns:
            tuple: A tuple containing:
                - `degrees` (float): The angle of the fitted line in degrees. Returns 'N/A' if no contour is provided.
                - `slope` (float): The slope of the fitted line. Returns `None` if no contour is provided.
                - `x_single` (float): The x-coordinate of the starting point of the fitted line.
                - `y_single` (float): The y-coordinate of the starting point of the fitted line.
                - `vx_single` (float): The x-component of the direction vector of the fitted line.
                - `vy_single` (float): The y-component of the direction vector of the fitted line.
        """
        if len(self.contour) > 0:
            vx, vy, x, y = cv2.fitLine(self.contour, cv2.DIST_L2, 0, 0.1, 0.1)
            x_single, y_single = x[0], y[0]
            vx_single, vy_single = vx[0], vy[0]
            slope = vy_single / vx_single
            
            if slope < 0:
                print(f'Slope: {slope}')
                slope = slope * -1
                print(f'Corrected Slope: {slope}')
            else:
                print(f'Slope: {slope}')

            radians = math.atan(slope)
            degrees = (radians * 180) / math.pi
            print(f'Angle: {degrees}')

            return degrees, slope, x_single, y_single, vx_single, vy_single
        else:
            return 'N/A', None, None, None, None


   
        
    