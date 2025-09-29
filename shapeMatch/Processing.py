import cv2
import numpy as np
import math
import os
import shutil
        
class VideoProcessing:
    @staticmethod
    def create_sample_folders(folder, j):
        """
        Creates a set of sample folders under a specified parent folder.

        Parameters:
            folder (str): The path to the parent directory where the sample folders will be created.
            j (int): An index used to name the sample folder.

        Returns:
            tuple: A tuple containing paths to the created folders:
                - sampleFolder (str): Path to the sample folder.
                - frameFolder (str): Path to the frames folder.
                - drawnFolder (str): Path to the drawn folder.
                - binaryFolder (str): Path to the binary folder.
                - cropFolder (str): Path to the crops folder.

        Example usage:
            sample_folders = VideoProcessing.create_sample_folders('path/to/parent/folder', 0)
            print(sample_folders)
        """
        sampleFolder = os.path.join(folder, f'Sample{j+1}')
        
        if os.path.exists(sampleFolder):
            shutil.rmtree(sampleFolder)
        os.mkdir(sampleFolder)

        frameFolder = os.path.join(sampleFolder, 'frames')
        os.mkdir(frameFolder)

        drawnFolder = os.path.join(sampleFolder, 'drawn')
        os.mkdir(drawnFolder)

        binaryFolder = os.path.join(sampleFolder, 'binary')
        os.mkdir(binaryFolder)

        cropFolder = os.path.join(sampleFolder, 'crops')
        os.mkdir(cropFolder)

        return sampleFolder, frameFolder, drawnFolder, binaryFolder, cropFolder
    
    def detect_contours(self, image):
        """
        Detects contours in the given image by converting it to grayscale, applying a binary threshold,
        and finding contours.

        Parameters:
            image (numpy.ndarray): The original BGR image.

        Returns:
            tuple: A tuple containing:
                - thresholded_image (numpy.ndarray): The binary image after applying the threshold.
                - contours (list): The list of detected contours.
                - hierarchy (numpy.ndarray): The hierarchy of contours.

        Example usage:
            thresh, contours, hierarchy = VideoProcessing().detect_contours(image)
            print(f'Thresholded Image Shape: {thresh.shape}, Number of Contours: {len(contours)}')
        """
        # Convert the image to grayscale
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the grayscale image
        ret, thresh = cv2.threshold(imgray, 250, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return thresh, contours, hierarchy
    
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

        Example usage:
            image = Binarize('path/to/image.png').read_image()
            print(f'Image Shape: {image.shape}')
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

        Example usage:
            binarizer = Binarize('path/to/image.png')
            image = binarizer.read_image()
            binary_image = binarizer.global_threshold(image)
            print(f'Binary Image Shape: {binary_image.shape}')
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

        Example usage:
            binarizer.save_cleaned_image(binary_image, 'path/to/output.png')
        """
        success = cv2.imwrite(output_path, cleaned_image)
        if not success:
            raise IOError(f"Failed to save image at {output_path}")
    
    def binarize_image(self, output_path):
        """
        Processes the image by applying global thresholding and morphological operations,
        and then saves the result.
        
        Parameters:
            output_path (str): The path where the processed image will be saved.
        
        Returns:
            numpy.ndarray: The original grayscale image.

        Example usage:
            binarizer = Binarize('path/to/image.png')
            original_image = binarizer.binarize_image('path/to/output.png')
            print(f'Original Image Shape: {original_image.shape}')
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
        if contour is None:
            raise ValueError("Contour cannot be None")
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

    def filter_by_x_level(self, x_level_min, x_level_max):
        """
        Filters out points in the contour that are outside a specified x-level range.

        Parameters:
            x_level_min (int): The minimum x-level threshold; points left of this level will be excluded.
            x_level_max (int): The maximum x-level threshold; points right of this level will be excluded.

        Returns:
            numpy.ndarray: The filtered contour with points only within the specified x-level range.
        """
        if self.contour.size == 0:
            print("Warning: Contour is empty after ensuring 2D format")
            return self.contour

        # Filter points based on the specified x-level range
        filtered_contour = self.contour[(self.contour[:, 0] >= x_level_min) & (self.contour[:, 0] <= x_level_max)]
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

    
    def split_horizontal(self):
        """
        Splits the contour into left and right parts based on the x-coordinate of its centroid.
        
        Returns:
            tuple: Two contours, left and right.

        Example usage:
            left_contour, right_contour = processor.split_horizontal()
            print(f'Left Contour Shape: {left_contour.shape}, Right Contour Shape: {right_contour.shape}')
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

        Example usage:
            calculator = Calculator(contour)
            print(f'Contour Shape: {calculator.contour.shape}')
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
        """
        if len(self.contour) > 0:
            vx, vy, x, y = cv2.fitLine(self.contour, cv2.DIST_L2, 0, 0.1, 0.1)
            x_single, y_single = x[0], y[0]
            vx_single, vy_single = vx[0], vy[0]
            slope = vy_single / vx_single
            
            if slope < 0:
                slope = -slope

            radians = math.atan(slope)
            degrees = (radians * 180) / math.pi
            
            return float(degrees), slope, x_single, y_single, vx_single, vy_single
        else:
            # Return default values or handle the case appropriately
            return 90, 0.0, 0.0, 0.0, 0.0, 0.0



class Draw:
    def __init__(self, image):
        """
        Initialize the Draw class with an image.

        Parameters:
            image (numpy.ndarray): The image on which to draw. Should be a BGR image in numpy array format.
        """
        self.image = image
    
    def line_of_best_fit(self, x, y, vx, vy, length):
        """
        Draws two lines of best fit on the image based on the given starting point and direction vector.

        Parameters:
            x (float): The x-coordinate of the starting point of the line.
            y (float): The y-coordinate of the starting point of the line.
            vx (float): The x-component of the direction vector of the line.
            vy (float): The y-component of the direction vector of the line.
            length (float): The length of the line to be drawn in both directions from the starting point.

        Example usage:
            Draw(image).line_of_best_fit(x, y, vx, vy, length)
        """
        # Draw the line in the direction of (vx, vy)
        cv2.line(self.image, (int(x), int(y)), (int(x + length * vx), int(y + length * vy)), (0, 0, 255), 2)
        # Draw the line in the opposite direction of (vx, vy)
        cv2.line(self.image, (int(x), int(y)), (int(x - length * vx), int(y - length * vy)), (0, 0, 255), 2)

    def contour(self, contour, blue, green, red):
        """
        Draws a contour on the image with the specified color.

        Parameters:
            contour (numpy.ndarray): The contour to be drawn. Should be a 2D array with shape (-1, 1, 2).
            blue (int): The blue color component of the contour in BGR format (0-255).
            green (int): The green color component of the contour in BGR format (0-255).
            red (int): The red color component of the contour in BGR format (0-255).

        Example usage:
            Draw(image).contour(contour, 255, 0, 0)  # Draw contour in red
        """
        cv2.drawContours(self.image, [contour], -1, (blue, green, red), 2)

