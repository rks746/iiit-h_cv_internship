To perform object detection and segmentation on a video, it must first be broken down into individual frames.   
A video file contains the following 
- Frames (compressed images)
- FPS (frames per second)
- Codec
- Resolution 

cv2.VideoCapture (video -> frames)
- opens video file
- decodes it frame-by-frame
- exposes an iterator like interface

Each frame is processed (detection and segmentation) independently by model.  
Python libraries like cv2, os, shutil save the processed frames.  

cv2.VideoWriter (frames -> video)
- frames should have the same resolution 
- be in correct order (frame names are zero padded)
- match the original FPS

This exact method doesn't scale well because frame by frame computation is expensive.  
Some optimizations that can be done are frame skipping, batch inference, GPU acceleration and tracking.   