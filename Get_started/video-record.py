import cv2
import os


# ffmpeg

filename = 'video.avi'     # .mp4
frames_per_seconds = 24.0
res = '480P'               # 1080p

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    '480P': (640, 480),
    '720P': (1280, 720),
    '1080P': (1920, 1080),
    '4K': (3840, 2160),
}

def get_dims(cap, res='720P'):
    width, height = STD_DIMENSIONS['720P']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: https://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


cap = cv2.VideoCapture(0)
dims = get_dims(cap, res=res)
video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2, frames_per_seconds, dims)

while True:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # can not be written

    out.write(frame)
    cv2.imshow('frame', frame)    # imgshow
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()