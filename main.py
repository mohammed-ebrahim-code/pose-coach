import csv
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np

import hud


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    top = np.dot(ba, bc)
    ba_len = np.linalg.norm(ba)
    bc_len = np.linalg.norm(bc)
    val = np.clip(top / (ba_len * bc_len + 1e-6), -1, 1) 
    angle_radians = np.arccos(val)
    angle_degree = np.degrees(angle_radians)
    return angle_degree


def to_pixel(landmark, width, height):
    pixel_x = int(landmark.x * width)
    pixel_y = int(landmark.y * height)
    return (pixel_x, pixel_y)


VIDEO_PATH = None
SAVE_OUTPUT = False
SAVE_LOG = False
CAMERA_SIZE = (1280, 720)
MAX_WIDTH = 1280
OUTPUT_FPS = 30

DOWN = 90
UP = 160
SAG_ANGLE = 155
BODY_CHECK_ANGLE = 130
VISIBILITY = 0.5
FEEDBACK_SECONDS = 1

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23
LEFT_ANKLE = 27

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24
RIGHT_ANKLE = 28

CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
               (23, 25), (24, 26), (25, 27), (26, 28), (11, 12)]

BASE_DIR = Path(__file__).parent
model_path = str((BASE_DIR / "models/pose_landmarker_full.task").resolve())
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

good_reps = 0
bad_reps = 0
stage = "up"
lowest_body_angle = 1000
time_end_rep = -100
last_rep = None
fps = 0

USE_WEBCAM = VIDEO_PATH is None
MIRROR = USE_WEBCAM

with PoseLandmarker.create_from_options(options) as landmarker:
    if USE_WEBCAM:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    else:
        cap = cv.VideoCapture(str(BASE_DIR / VIDEO_PATH))
    if not cap.isOpened():
        print("Cannot open camera / video")
        exit()
    video_fps = cap.get(cv.CAP_PROP_FPS) or 30

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = None
    frames_written = 0
    log_file = None
    if SAVE_LOG:
        (BASE_DIR / "results").mkdir(exist_ok=True)
        log_file = open(BASE_DIR / f"results/reps_{stamp}.csv", "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["rep", "result", "lowest_body_angle", "time_s"])

    cv.namedWindow("Pose Coach", cv.WINDOW_NORMAL)
    start_time = time.monotonic()
    last_clock = start_time
    last_frame_timestamp_ms = -1
    frame_number = 0

    while True:
        right_arm_angle = None
        left_arm_angle = None
        angle = None
        body_angle = None
        body_visible = False
        points = {}
        highlight = ()
        body_line = None
        elbow_pixel = None

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        height, width = frame.shape[:2]
        if width > MAX_WIDTH:
            frame = cv.resize(frame, (MAX_WIDTH, int(height * MAX_WIDTH / width)))
            height, width = frame.shape[:2]
        frame_number += 1

        now = time.monotonic()
        if USE_WEBCAM:
            frame_timestamp_ms = int((now - start_time) * 1000)
        else:
            frame_timestamp_ms = int(frame_number * 1000 / video_fps)
        if frame_timestamp_ms <= last_frame_timestamp_ms:
            continue
        last_frame_timestamp_ms = frame_timestamp_ms
        t = frame_timestamp_ms / 1000  

        if now > last_clock:
            fps = round(0.9 * fps + 0.1 * (1 / (now - last_clock))) if fps else round(1 / (now - last_clock))
        last_clock = now

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if pose_landmarker_result.pose_landmarks:
            landmarks = pose_landmarker_result.pose_landmarks[0]

            for i, lm in enumerate(landmarks):
                if lm.visibility >= VISIBILITY and (11 <= i <= 16 or 23 <= i <= 28):
                    points[i] = to_pixel(lm, width, height)

            vis_r = sum(landmarks[i].visibility for i in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)) / 3
            vis_l = sum(landmarks[i].visibility for i in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)) / 3
            if vis_l > vis_r:
                shoulder, elbow, wrist, hip, ankle = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_ANKLE
            else:
                shoulder, elbow, wrist, hip, ankle = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_ANKLE

            if all(landmarks[i].visibility > VISIBILITY for i in (shoulder, elbow, wrist)):
                s = np.array(to_pixel(landmarks[shoulder], width, height))
                e = np.array(to_pixel(landmarks[elbow], width, height))
                wr = np.array(to_pixel(landmarks[wrist], width, height))
                angle = round(calculate_angle(s, e, wr))
                elbow_pixel = to_pixel(landmarks[elbow], width, height)
                highlight = ((shoulder, elbow), (elbow, wrist))

            if all(landmarks[i].visibility > VISIBILITY for i in (shoulder, hip, ankle)):
                s = np.array(to_pixel(landmarks[shoulder], width, height))
                hp = np.array(to_pixel(landmarks[hip], width, height))
                an = np.array(to_pixel(landmarks[ankle], width, height))
                body_angle = round(calculate_angle(s, hp, an))
                # only judge the body during a real rep, not while setting up or resting at the top
                if angle is not None and angle < BODY_CHECK_ANGLE:
                    lowest_body_angle = min(body_angle, lowest_body_angle)
                body_visible = True
                body_line = (tuple(s), tuple(hp), tuple(an))

        if angle is not None and body_visible:
            if angle < DOWN and stage == "up":
                stage = "down"
            if angle > UP and stage == "down":
                stage = "up"
                if lowest_body_angle < SAG_ANGLE:
                    bad_reps += 1
                    last_rep = "bad"
                else:
                    good_reps += 1
                    last_rep = "good"
                rep_number = good_reps + bad_reps
                print(f"Rep {rep_number}: {last_rep}, lowest body angle {lowest_body_angle}")
                if log_file:
                    log.writerow([rep_number, last_rep, lowest_body_angle, round(t, 2)])
                    log_file.flush()
                lowest_body_angle = 1000
                time_end_rep = t

        hud.draw_skeleton(frame, points, CONNECTIONS, highlight, body_line,
                          body_ok=body_angle is None or body_angle >= SAG_ANGLE)
        if MIRROR:
            frame = cv.flip(frame, 1)
            if elbow_pixel is not None:
                elbow_pixel = ((width - 1) - elbow_pixel[0], elbow_pixel[1])

        if angle is None or not body_visible:
            warning = "Show your whole body from the side"
        else:
            warning = None
        feedback = last_rep if t - time_end_rep < FEEDBACK_SECONDS else None

        hud.draw_hud(frame, {
            "good": good_reps, "bad": bad_reps, "stage": stage, "fps": fps,
            "angle": angle, "body_angle": body_angle,
            "sag_angle": SAG_ANGLE, "down": DOWN, "up": UP,
            "feedback": feedback, "warning": warning, "elbow_pos": elbow_pixel,
        })

        cv.imshow("Pose Coach", frame)

        if SAVE_OUTPUT:
            if writer is None:
                (BASE_DIR / "output").mkdir(exist_ok=True)
                out_fps = OUTPUT_FPS if USE_WEBCAM else video_fps
                writer = cv.VideoWriter(str(BASE_DIR / f"output/run_{stamp}.mp4"), cv.VideoWriter_fourcc(*"mp4v"),
                                        out_fps, (width, height))
            if USE_WEBCAM:

                frames_due = int((now - start_time) * OUTPUT_FPS) + 1 - frames_written
                for _ in range(max(0, frames_due)):
                    writer.write(frame)
                    frames_written += 1
            else:
                writer.write(frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv.getWindowProperty("Pose Coach", cv.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            good_reps = 0
            bad_reps = 0
            stage = "up"
            lowest_body_angle = 1000
            last_rep = None

cap.release()
if writer is not None:
    writer.release()
if log_file:
    log_file.close()
cv.destroyAllWindows()