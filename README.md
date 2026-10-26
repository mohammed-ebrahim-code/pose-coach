# Pose Coach: a push-up counter that checks your form

Pose Coach watches push-ups from the side through a webcam or a video file. It counts only **full** reps and flags reps where the **hips sag**. I built it from scratch in Python for my replicate project at TKS (The Knowledge Society) Dubai.


## Results

**My own tests** (laptop camera on the floor, side view):

| Test | Result |
|---|---|
| Full push-ups counted | **15 / 15** |
| Half reps rejected | **5 / 5** |
| Reps with sagging hips flagged | **5 / 5** |
| Good reps judged correctly | **10 / 10** after tuning the threshold to 155° (8 / 10 before) |

**Someone else's video**  **4 / 4** reps counted, all judged good. This test also found a bug that my own tests missed: getting into position with raised hips was being counted against rep 1. I fixed it with `BODY_CHECK_ANGLE`.

## How it works

1. **Find the joints:** MediaPipe Pose Landmarker (VIDEO mode) finds 33 body points in each frame. Points with visibility below 0.5 are ignored.
2. **Pick the side:** it uses the side of the body facing the camera (the one with higher visibility) for both the arm and the body.
3. **Measure two angles:** both come from the dot product and are calculated in pixels.
   - **elbow** = shoulder–elbow–wrist, used for counting
   - **body line** = shoulder–hip–ankle, used for form
4. **Count reps with a two-threshold state machine:** the elbow goes below 90° → "down"; it comes back above 160° → "up", and the rep counts. The 70° gap means the ±2° jitter can never fake a rep.
5. **Judge the whole rep:** once the elbow is below 130°, it keeps the lowest body angle. When the rep ends, below 155° = hips sagging.

## Run it

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the model, [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task), into a `models/` folder. Then:

```
python main.py
```

**Q** quits (always quit with Q, so the saved video finishes properly). **R** resets the count.

### Settings (top of `main.py`)

| Setting | What it does |
|---|---|
| `VIDEO_PATH` | `None` for the webcam, or a file such as `"videos/pushups.mp4"` |
| `SAVE_OUTPUT` | Saves what the window shows to `output/` |
| `SAVE_LOG` | Saves every rep (good/bad, lowest body angle) to `results/` |
| `MAX_WIDTH` | Shrinks large videos, so they're faster to process and the saved file is smaller |
| `DOWN`, `UP` | Elbow thresholds for counting (90°, 160°) |
| `SAG_ANGLE` | Body-line threshold for sagging hips (155°) |
| `BODY_CHECK_ANGLE` | The body is only judged once the elbow is below this (130°) |

## Limitations

- **Side view only.** The angles are 2D, so an arm pointing at the camera isn't measured properly.
- **Sagging and piking look the same.** Both make the body angle smaller, so it can only say "hips not straight", not which way.
- **Tuned on one person:** me, in one room, in loose clothes. Other bodies, cameras and lighting may need different thresholds.
- **The whole body must be visible,** feet included. If not, it pauses instead of guessing.
- **One person, one exercise.**

## Files

- `main.py`: the pose logic, angle math, rep counter and form check
- `hud.py`: the on-screen display

## Credits

- **Pose model:** [Google MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Apache 2.0)
- The pose logic, angle math, rep counter and form check are my own work. The on-screen display (`hud.py`) was built with AI assistance.

## Read more

- Learn article: [Your Camera as a Coach](https://medium.com/@m.ebrahim.alzaabi/your-camera-as-a-coach-how-ai-pose-estimation-could-help-beginners-train-with-correct-form-325d17dc24f6)
- Replicate article: [FILL: link]
- Demo video: [FILL: link]

Mohammed Ebrahim AlZaabi · TKS Dubai