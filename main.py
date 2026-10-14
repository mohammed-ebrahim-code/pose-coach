import numpy as np
import cv2 as cv
import mediapipe as mp
import time
from pathlib import Path

model_path_c = Path(__file__).parent / 'models/pose_landmarker_full.task'
model_path = str(model_path_c.resolve())
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
coordinates = (50, 100) 
font_scale = 1
font = cv.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)
thickness = 2


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)
last_frame_timestamp_ms = 0
connections = [(11, 13), (13, 15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (24,26), (25,27), (26,28),(11,12)]

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    start_time = time.monotonic()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.flip(frame, 1)

        # Our operations on the frame come here
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int((time.monotonic() - start_time) * 1000)
        try:
            fps = int(1000 / (frame_timestamp_ms - last_frame_timestamp_ms))
        except:
            fps = 0
        last_frame_timestamp_ms = frame_timestamp_ms
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)


        text_to_display = f"FPS :{str(fps)}"
        cv.putText(frame, text_to_display, coordinates, font, font_scale, color, thickness, cv.LINE_AA)

        #print(len(pose_landmarker_result.pose_landmarks))
        if pose_landmarker_result.pose_landmarks:
            landmarks = pose_landmarker_result.pose_landmarks[0]
            height, width = frame.shape[:2]
            for points in connections:
                start = landmarks[points[0]]
                end = landmarks[points[1]]
                if start.visibility < 0.5:
                    continue
                if end.visibility < 0.5:
                    continue
                x1 = (int(start.x * width))
                y1 = (int(start.y * height))
                x2 = (int(end.x * width))
                y2 = (int(end.y * height))
                cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            for lm in landmarks:
                if lm.visibility < 0.5:
                    continue
                pixel_x = (int(lm.x * width))
                pixel_y = (int(lm.y * height))
                
                cv.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)

        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break


        
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

