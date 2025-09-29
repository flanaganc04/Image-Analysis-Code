import cv2
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
    # def adaptive_threshold(self, image):
    #     adaptive_mask = cv2.adaptiveThreshold(
    #         image, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
    #         self.adaptive_thresh_window_size, self.adaptive_thresh_C)
    #     return adaptive_mask
    # def combine_masks(self, global_mask, adaptive_mask):
    #     return cv2.bitwise_or(global_mask, adaptive_mask)
    # def morphological_operations(self, combined_mask):
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
    #     return cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
    def save_cleaned_image(self, cleaned_image, output_path):
        cv2.imwrite(output_path, cleaned_image)
    def process_image(self, output_path):
        image = self.read_image()
        global_mask = self.global_threshold(image)
        # adaptive_mask = self.adaptive_threshold(image)
        # combined_mask = self.combine_masks(global_mask, adaptive_mask)
        # cleaned_image = self.morphological_operations(combined_mask)
        self.save_cleaned_image(global_mask, output_path)
        return image
# Example usage:
sample = 6
preprocess = Preprocess(f'./{sample}.png')
cleaned_image = preprocess.process_image(f'./cleaned_{sample}.jpg')









