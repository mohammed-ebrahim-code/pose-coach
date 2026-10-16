import numpy as np
import cv2 as cv
import mediapipe as mp
import time
from pathlib import Path

def calculate_angle(a,b,c):
    ba = a - b
    bc = c - b

    top = np.dot(ba , bc)
    ba_len = np.linalg.norm(ba)
    bc_len = np.linalg.norm(bc)
    val = np.clip(top / (ba_len * bc_len), -1 , 1)
    angle_radians = np.arccos(val)
    angle_dgree = np.degrees(angle_radians)
    return angle_dgree

def to_pixel(landmark , width , height):
    pixel_x = (int(landmark.x * width))
    pixel_y = (int(landmark.y * height))
    return (pixel_x , pixel_y)


model_path_c = Path(__file__).parent / 'models/pose_landmarker_full.task'
model_path = str(model_path_c.resolve())
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

coordinates = (50, 100) 
coordinates_1 = (0,0) 

font_scale = 1
font = cv.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)
thickness = 2


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

last_frame_timestamp_ms = 0

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

connections = [(11, 13), (13, 15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (24,26), (25,27), (26,28),(11,12)]

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    start_time = time.monotonic()

    while True:
        right_arm_angle = None
        left_arm_angle = None

        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        height, width = frame.shape[:2]

        # Our operations on the frame come here
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int((time.monotonic() - start_time) * 1000)
        try:
            fps = int(1000 / (frame_timestamp_ms - last_frame_timestamp_ms))
        except ZeroDivisionError:
            continue
        last_frame_timestamp_ms = frame_timestamp_ms
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        #print(len(pose_landmarker_result.pose_landmarks))
        if pose_landmarker_result.pose_landmarks:
            landmarks = pose_landmarker_result.pose_landmarks[0]
            for points in connections:
                start = landmarks[points[0]]
                end = landmarks[points[1]]
                if start.visibility < 0.5:
                    continue
                if end.visibility < 0.5:
                    continue

                x1 , y1 = to_pixel(start,width,height)
                x2 , y2 = to_pixel(end,width,height)

                cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            for lm in landmarks:
                if lm.visibility < 0.5:
                    continue
                pixel_x , pixel_y = to_pixel(lm,width,height)
                cv.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)

            if all(l.visibility > 0.5 for l in [landmarks[RIGHT_ELBOW], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_WRIST]]):
                
                re_x, re_y = to_pixel(landmarks[RIGHT_ELBOW],width,height)
                re_x_image = (width - 1) - re_x
                position_r = tuple((re_x_image , re_y))
                
                re = np.array((re_x , re_y))
                rs = np.array(to_pixel(landmarks[RIGHT_SHOULDER],width,height))
                rw = np.array(to_pixel(landmarks[RIGHT_WRIST],width,height))
                
                right_arm_angle = round(calculate_angle(rs , re , rw))
                right_arm_angle_s = str(right_arm_angle)
            if all(l.visibility > 0.5 for l in [landmarks[LEFT_ELBOW], landmarks[LEFT_SHOULDER], landmarks[LEFT_WRIST]]):
                
                le_x, le_y = to_pixel(landmarks[LEFT_ELBOW],width,height)
                le_x_image = (width - 1) - le_x
                position_l = tuple((le_x_image , le_y))
                
                le = np.array((le_x , le_y))
                ls = np.array(to_pixel(landmarks[LEFT_SHOULDER],width,height))
                lw = np.array(to_pixel(landmarks[LEFT_WRIST],width,height))
                
                left_arm_angle = round(calculate_angle(ls , le , lw))
                left_arm_angle_s = str(left_arm_angle)
                
        frame = cv.flip(frame, 1)
        
        text_to_display = f"FPS :{str(fps)}"
        cv.putText(frame, text_to_display, coordinates, font, font_scale, color, thickness, cv.LINE_AA)

        if right_arm_angle is not None:
            cv.putText(frame, right_arm_angle_s, position_r, font, font_scale, color, thickness, cv.LINE_AA)
        if left_arm_angle is not None:
                    cv.putText(frame, left_arm_angle_s, position_l, font, font_scale, color, thickness, cv.LINE_AA)
        
        text_to_display_angle = f"R :{str(right_arm_angle)}    L :{str(left_arm_angle)}"
        cv.putText(frame, text_to_display_angle, coordinates_1, font, font_scale, color, thickness, cv.LINE_AA)


        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

        
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

