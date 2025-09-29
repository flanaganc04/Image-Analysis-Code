import cv2

class Preprocess:
    def __init__(self, directory):
        self.global_thresh_value = 253
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



