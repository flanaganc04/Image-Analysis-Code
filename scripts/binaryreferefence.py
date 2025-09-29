import math
# import matplotlib.pyplot as plt
import cv2
import numpy as np
# import scipy.constants as sc
# from sklearn.cluster import KMeans
# from skimage import measure
# import random
# from collections import deque
# import os
# import math
# import PIL
# # from scipy.fft import fft2, ifft2, fftshift
# # from joblib import Parallel, delayed
# from PIL import Image
# from PIL import ImageEnhance
# # from PIL import ImageFilter
# # from skimage import io
# # from cupy_common import check_cupy_available
# from PIL import ImageDraw
# # gpu_accelerated = check_cupy_available()

class Preprocess:
    def __init__(self, directory):
        self.global_thresh_value = 125
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

path = 'test video\\Samples\\Sample 1\\frames\\Sample1Frame1.png'
newpath ='test video\\Samples\\Sample 1\\frames\\Sample1Frame1Binary.png'  
sample = 6
preprocess = Preprocess(path)
cleaned_image = preprocess.process_image(newpath)

contour, matlike = cv2.findContours(cleaned_image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = cv2.drawContours(cleaned_image, [contour], 0, (0,0,255), 5)

# [vx, vy, x, y] = cv2.fitLine(contours,cv2.DIST_L2, 0, 0.01, 0.01)

# cv2.line(contours, (x, y), (x + vx, y + vy), (0, 255, 0), 2)
# cv2.drawContours(contours, [contour], 0, (0, 0, 255), 2)

cv2.imshow(contours)
cv2.waitKey(0)
cv2.destroyAllWindows()