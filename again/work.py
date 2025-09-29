import os
from moviepy import VideoFileClip
import sys
import shutil
from PreprocessMod import Preprocess
print(sys.executable)

videoPath = 'again/PVA100000Toluene.MP4'
video = VideoFileClip(videoPath)
folder = 'again'

start = [34.5, 61.9, 87.3, 112.8, 201, 227, 253, 278.5, 366, 393.5, 418.5, 443.7]
stop = [35.5, 62.9, 88.3, 113.8, 203, 228, 254, 279.5, 367, 393.5, 419.5, 444.7]

for j in range(12):
    sampleFolder = os.path.join(folder, f'Sample{j+1}')
    if os.path.exists(sampleFolder):
        shutil.rmtree(sampleFolder)
    os.mkdir(sampleFolder)

    frameFolder = os.path.join(sampleFolder, 'frames')
    os.mkdir(frameFolder)
    drawnFolder = os.path.join(sampleFolder, "drawn")
    os.mkdir(drawnFolder)
    drawnFolder = os.path.join(sampleFolder, "binary")
    os.mkdir(drawnFolder)
    cropFolder = os.path.join(sampleFolder, "crops")
    os.mkdir(cropFolder)

    clip = video.subclip(start[j],stop[j])
    moveclip = os.path.join(sampleFolder, f'Sample{j+1}.mp4')
    clip.write_videofile(moveclip, audio=False)

    for n in range(60):
        print(f'Saving frame{n+1}')
        framePath = f'again/sample{j+1}/frames/Sample{j+1}Frame{n+1}.png'
        binaryPath = f'again/sample{j+1}/binary/Sample{j+1}Frame{n+1}Binary.png'
        clip.save_frame(framePath, t = n * 0.01)

        preprop = Preprocess(framePath)
        preprop.process_image(binaryPath)
