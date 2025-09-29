
import os
from moviepy.editor import *
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

    def save_cleaned_image(self, cleaned_image, output_path):
        cv2.imwrite(output_path, cleaned_image)
    def process_image(self, output_path):
        image = self.read_image()
        global_mask = self.global_threshold(image)
        self.save_cleaned_image(global_mask, output_path)
        return image

os.chdir("C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis")



video_file = 'test video\\test_video.MP4'

video = VideoFileClip('test video\\test_video.MP4')
start = [18,49,80,110, 278, 311, 342, 373, 562, 592,622, 652]
stop = []
number = []


for n in range(12):
    number.append(n+1)
    stop.append(start[n] + 2)

folder = "C:\\Users\\flana\\OneDrive\\Documents\\Coding\\OT-2 Image analysis\\test video\\Samples"

for x in range(2):
    sample_clip = video.subclip(start[x],stop[x])
    sample = sample_clip.write_videofile(f"Sample {number[x]}.mp4", audio = False)

    new_folder = os.path.join(folder,f'Sample {number[x]}')
    os.mkdir(new_folder)

    frame_folder = os.path.join(new_folder, "frames")
    os.mkdir(frame_folder)

    binarized_frames = os.path.join(new_folder, "binarized_frames")
    os.mkdir(binarized_frames)

    new_file = os.path.join(new_folder,f'Sample {number[x]}.mp4')
    os.rename(f"Sample {number[x]}.mp4", new_file)

    frames = sample_clip.duration * sample_clip.fps

    for n in range(round(frames)):
        sample_clip.save_frame(f"Sample{number[x]}Frame{n+1}.png", t = n*0.01)

        frame = f"Sample{number[x]}Frame{n+1}.png"
        moved_frame = os.path.join(frame_folder,frame)
        os.rename(frame, moved_frame)

        binarize = Preprocess(moved_frame)

        binary = f"Sample{number[x]}Frame{n+1}Binary.png"
        binarize.process_image(binary)
        
        image = cv2.imread(binary)
        crop = image[1110:1127, 1875:2350]
        

        

        moved_binary = os.path.join(binarized_frames,binary)
        os.rename(binary,moved_binary)

        
    

