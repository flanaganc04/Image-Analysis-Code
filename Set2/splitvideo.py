from moviepy.editor import VideoFileClip

longPath = 'OT-2 Image analysis/Set2/DJI_20240715151641_0134_D.MP4'
longVideo = VideoFileClip(longPath)

subclip = longVideo.subclip(0, 600)
subclip.write_videofile('OT-2 Image analysis/Set2/PVA360000TolueneR.MP4', audio = False)
