import os
from moviepy.editor import VideoFileClip
import cv2 as cv
import math
from shapeMatch.PreprocessMod import Preprocess
import sys

print(sys.executable)

video_file = 'test_video.MP4'
video = VideoFileClip(video_file)
os.mkdir('samplesTest2')
folder = "samplesTest2"

polymer = "polymer"
molecular_weight = "enter MW"
solvent = "enter solvent"

# there should be thirty six times for a full plate but can be configured for different amounts
# times are in seconds
# stop array is two seconds after giving you 120 frames to parse
number = [1,2,3,4,5,6,7,8,9]
start = [18,49]
stop = [20, 51]
bigData = []


for x in range(2):
    
    # write the polymer type, molecular weight, solvent, and sample number to the csv for each sample
    data = {'Polymer':f'{polymer}','Molecular Weight':f'{molecular_weight}', 'Solvent':f'{solvent}', 'Sample Number':f'{x+1}'}

    # make folders for the sample video, frames, and binarized and drawn on frames
    sampleFolder = os.path.join(folder,f'Sample{x+1}')
    os.mkdir(sampleFolder)
    frameFolder = os.path.join(sampleFolder,'frames')
    os.mkdir(frameFolder)
    drawnFolder = os.path.join(sampleFolder,"drawn")
    os.mkdir(drawnFolder)

    # make the 2 second subclip and save it as a file
    clip = video.subclip(start[x],stop[x])
    moveClip = os.path.join(sampleFolder,f"Sample{x+1}.mp4")
    clip.write_videofile(moveClip, audio = False)

    print(data)
    frames = 60

    for n in range(frames):
        # write the frame number to the csv
        data['Frame'] = f'{n + 1}'

        # save the frame as a file and move it to the appropriate folder
        frame = f'samplesTest2\\Sample{x + 1}\\frames\\Sample{x + 1}Frame{n+1}.png'
        clip.save_frame(frame, t = n * 0.02)
        

        # create a binary image to draw on
        preprop = Preprocess(frame)
        moveBinary = f'samplesTest2\\Sample{x + 1}\\drawn\\Sample{x + 1}Frame{n + 1}Binary.png'
        preprop.process_image(moveBinary)
        binary = cv.imread(moveBinary)
        binary = binary[1112:1127, 1900:2300]


        # detect the contour of the drop
        newimg = cv.imread(frame)
        newimg = newimg[1112:1127, 1900:2300]
        imgray = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, mat = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        # find the liquid, vapor, substrate interface
        M = cv.moments(contour)
        cx = int(M['m10'] / M['m00'])
        right_contour = contour[contour[:, :, 0] >= cx]
        M2 = cv.moments(right_contour)
        cx = int(M2['m10'] / M2['m00'])
        newRight_contour = right_contour[right_contour[:, 0] >= cx]
        
        # draw the spot the computer is recognizing as the interface
        cv.drawContours(binary, [newRight_contour] , 0 , (0,0,255), 2)
        vx,vy,x,y = cv.fitLine(newRight_contour, cv.DIST_L2, 0, 0.01, 0.01)

        x_single = x[0]
        y_single = y[0]
        vx_single = vx[0]
        vy_single = vy[0]

        # draw the line of best fit
        cv.line(binary, (int(x_single), int(y_single)), (int(x_single+300*vx_single),int(y_single+300*vy_single)), (0, 215, 255), 2)

        cv.line(binary, (int(x_single), int(y_single)), (int(x_single+-300*vx_single),int(y_single+-300*vy_single)), (0, 215, 255), 2)

        cv.imwrite(moveBinary,binary)


        
        
        
        

        
   

            
            


            
