import cv2
import os
from moviepy.editor import VideoFileClip


#Takes video path and returns all frames
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        #Read in next frame
        ret, frame = cap.read()
        #Ret is false when no more things to read
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def convert_avi_to_mp4(avi_file_path, output_name):
    # Get the directory and filename without extension
    directory = os.path.dirname(avi_file_path)
    filename = os.path.splitext(os.path.basename(avi_file_path))[0]
    
    # Create the output path
    output_name = os.path.join(directory, f"{filename}.mp4")

    # Convert the video
    clip = VideoFileClip(avi_file_path)
    clip.write_videofile(output_name)

    # Delete the original AVI file
    os.remove(avi_file_path)