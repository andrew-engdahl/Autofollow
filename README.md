# Autofollow: Intelligent Virtual PTZ Camera

A macOS application that turns a fixed wide camera into a virtual PTZ camera. It detects people with YOLOv8-Pose, picks a subject, and crops a smooth 16:9 shot that follows them — with a control panel for live operation, a fullscreen program output for a second display, and a diagnostics window that shows what the tracker is thinking.

## Features

- **Real-time pose detection** with YOLOv8-Pose (GPU-accelerated via Apple Metal when available)
- **Primary Focus mode** — follows the closest person, hands off to someone closer or more active, and re-acquires automatically when the subject leaves
- **Virtual switcher** — rotates between people on a timer, or on demand with per-person buttons
- **Cut or crossfade** transitions, with a pre-travel phase so the camera is already settled on the new subject
- **Shot types** — Full Body, Waist Up, Medium, Close-Up — computed from actual body landmarks
- **Smooth camera movement** — deadzone, eased panning, heavily damped tilt and zoom
- **Audience exclusion zone** to ignore people standing in front of the stage
- **Fullscreen output** on any connected display; settings remembered between launches
- **Diagnostics window** with color-coded skeleton preview, per-person scores, and a switch-event log
- **Headless CLI mode** for recording to MP4 without the GUI
- **People profiles** *(optional)* — enroll named people with reference photos; InsightFace recognizes them on camera and biases subject selection toward higher-priority people (e.g. the pastor)
- **Speaker recognition** *(optional)* — attach voice samples to a profile; while that person is talking they get a temporary priority boost
- **Music detection** *(optional)* — YAMNet tells music from speech; during a performance the switcher follows the most active performer with shorter dwell, and a recognized speaker vetoes music mode

## Requirements

- macOS 12+
- Python 3.10+ (Homebrew's `python@3.13` is what the Dock launcher expects)
- Webcam, capture card, or other camera visible to macOS

## Installation

```bash
./setup.sh                      # core app
./setup.sh --with-recognition   # + face / speaker / music recognition (large download)
```

This creates the `.venv` virtual environment, installs dependencies, removes any conflicting OpenCV builds, and re-signs `Autofollow.app`. It's safe to re-run. The recognition extras (`requirements-recognition.txt`) are optional: without them the People and Audio features simply report *unavailable* and everything else works.

## Usage

### Launch the control panel

```bash
./run.sh
```

or double-click `Autofollow.app` (it uses the same `.venv`).

The panel shows up immediately; the pose model loads in the background and the status bar reads *Loading pose model…* until it's ready.

### Control panel

| Section | What it does |
|---------|--------------|
| **Camera** | Choose the input device. *Refresh* rescans without interrupting the live camera. |
| **Shot Type** | Full Body / Waist Up / Medium / Close-Up. |
| **Mode** | *Disabled* shows the whole frame. *Primary* follows the closest person. *Time* rotates between people every N seconds. *Manual* switches only when you press a person button. |
| **Transition** | Cut or Crossfade, with fade duration. Used for every subject change, including Primary hand-offs. |
| **Display / Fullscreen** | Pick a display and open the program output fullscreen there. Esc, Q, or a double-click on the output closes it. |
| **Audience Exclusion** | Ignore anyone whose torso is in the bottom N% of the frame. The zone is drawn in yellow in Diagnostics. |
| **Max Tracked Persons** | Cap on simultaneously tracked people. |
| **Audio** | Enable analysis of a microphone input for speaker recognition and music detection (needs the recognition extras). |
| **Manage People** | Enroll people by face and voice and set their priority. |

All of these are saved and restored on the next launch (the audio device is remembered by name).

### Diagnostics

*Open Diagnostics* shows the raw camera view with skeletons, bounding boxes, the primary indicator and the exclusion zone; a per-person table of the scores that drive switching; and a log of every switch and why it happened. The overlay is only rendered while the window is open, so leaving it closed costs nothing.

### People profiles (optional)

**Manage People** opens the profile window.

1. **Add Person…**, enter a name and a priority (0–10). Higher priority means the auto-switcher prefers this person and dwells on them longer; 0 behaves like an unknown person.
2. Add reference photos from disk or **Capture from camera**. Saving embeds them (the first time downloads the InsightFace `buffalo_l` models, ~300 MB, to `~/.insightface/models/`).
3. Optionally add **Voice Samples** — WAV/FLAC files, or **Record 5 s from mic** while Audio is enabled in the main panel. Aim for ≥10 s of clean speech per person.

Once a profile has face embeddings, recognition runs on a background worker (a few times a second, on upper-body crops so distant faces are still legible) and the diagnostics overlay labels the person by name — `★8 Pastor Mike`, with `●` while their voice is recognized. A match is forgotten after 5 s without re-confirmation, so identities don't stick to the wrong track after people cross.

Profiles live in `~/Library/Application Support/Autofollow/profiles/`.

### Audio (optional)

Tick **Enabled** in the Audio section and pick an input. While on:

- A recognized enrolled voice gives that tracked person a +5 priority boost for ~3 s.
- Sustained music (≥5 s) with no recognized speaker switches to **music mode**: Primary follows the most active performer, the time switcher halves its interval and picks by activity. Speech, or any recognized voice, exits music mode.
- Model downloads (SpeechBrain ECAPA ~80 MB, YAMNet via TF Hub) happen on first use and are reported in the status bar.

### Headless / recording

```bash
./run.sh --headless --output show.mp4
./run.sh --headless --camera 1 --shot-type full_body --no-preview --max-frames 3000
./run.sh --list-cameras
```

## How It Works

1. **Detect** — YOLOv8-Pose runs every `DETECTION_INTERVAL` frames on a downscaled copy of the input and returns 17 COCO keypoints per person. People with no visible hips (typically foreground audience cut off at the waist) are ignored.
2. **Track** — Detections are matched to existing tracks by IoU, with a center-distance fallback for fast movers, giving each person a stable ID. Tracks expire after `TRACK_DROPOUT_SECONDS` unseen. Each track carries an EMA-smoothed *foreground score* (bbox area) and *activity score* (torso movement).
3. **Frame** — For the chosen shot type, the zoom is computed from the head and the shot's bottom landmark (hips, knees, ankles…) so the framing is consistent regardless of how far away the person is. Zoom is bounded between "the whole camera frame" and `MAX_ZOOM`.
4. **Smooth** — A per-subject PTZ smoother applies a horizontal deadzone, quadratic ease-out panning, and slow tilt/zoom, all capped by `MAX_*_SPEED`.
5. **Switch** — In Primary mode a hand-off needs the candidate to be ≥1.5× larger or ≥2× more active, and only after `PRIMARY_DWELL_SECONDS`. Profile priority shifts that threshold (a higher-priority person who is at least as close takes over immediately; a lower-priority one must be much closer) but never stops an unknown person who is clearly nearer. If the subject disappears, the camera holds and slowly widens; after `PRIMARY_REACQUIRE_DELAY` the best remaining person is adopted with the configured transition.

## Configuration

`config.py` holds the startup defaults. The most useful knobs:

```python
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720   # program output size
CAPTURE_WIDTH, CAPTURE_HEIGHT = 0, 0      # requested camera mode (0 = camera default)

CONFIDENCE_THRESHOLD = 0.7   # keypoint / person confidence
DETECTION_SCALE = 0.5        # run YOLO on this fraction of the input resolution
DETECTION_INTERVAL = 2       # detect every N frames

SHOT_TYPE = 'waist_up'
MAX_ZOOM = 4.0               # relative to the output size
DEADZONE = 0.4               # fraction of the viewport the subject can roam without panning
SMOOTHING = 0.5              # 0 = responsive … 1 = very smooth
MAX_PAN_SPEED = 15           # source px / frame
MAX_TILT_SPEED = 3
MAX_ZOOM_SPEED = 0.015

PRIMARY_DWELL_SECONDS = 3.0
PRIMARY_REACQUIRE_DELAY = 1.0
TRACK_DROPOUT_SECONDS = 2.0
```

## Performance Notes

- On Apple Silicon the model runs on Metal (`mps`); a 1080p camera processes at 30+ fps with detection every other frame.
- Lower `DETECTION_SCALE` or raise `DETECTION_INTERVAL` for more headroom on slower machines; raise `DETECTION_SCALE` toward 1.0 if small, distant people are missed.
- Output frames are dropped rather than queued if the UI falls behind, so latency stays bounded.
- Face recognition and audio analysis run on their own threads and never block the video pipeline; one shared InsightFace / SpeechBrain instance serves both live recognition and profile enrollment.

## Troubleshooting

**No cameras found / "Could not open camera"** — check that Terminal (or the app) has Camera permission in System Settings → Privacy & Security → Camera. Use `./run.sh --list-cameras` to see what macOS exposes.

**"No signal from camera — reconnecting…"** — the device stopped delivering frames (unplugged, or grabbed by another app). Autofollow retries every few seconds and resumes automatically.

**People aren't detected** — improve lighting, or lower `CONFIDENCE_THRESHOLD`. Make sure the Audience Exclusion slider isn't hiding them (check the yellow zone in Diagnostics).

**The shot ping-pongs between two people** — raise `PRIMARY_DWELL_SECONDS`, or use Manual mode.

**"Face recognition unavailable" / "Music detection unavailable"** — the optional stack isn't installed: `./setup.sh --with-recognition`.

**Dock app won't launch** — run `./setup.sh` again; it rebuilds the environment and re-signs the bundle.

## Project Structure

```
Autofollow/
├── setup.sh               # One-click setup (venv + dependencies + re-sign app)
├── run.sh                 # Launch the app
├── main.py                # Entry point: GUI (default), --headless, --list-cameras
├── control_ui.py          # PyQt5 control panel, video thread, output + diagnostics windows
├── video_processor.py     # Headless processing pipeline
├── camera.py              # Camera discovery / opening
├── pose_detector.py       # YOLOv8-Pose wrapper
├── tracker.py             # Multi-person tracking with stable IDs
├── framing_engine.py      # Shot-type framing and crop
├── smoothing.py           # PTZ smoothing / deadzone
├── switcher.py            # Virtual switcher state machine
├── profiles.py            # People profile store (faces, voices, priority)
├── face_recognizer.py     # InsightFace wrapper (shared instance)
├── face_worker.py         # Background face-recognition worker
├── audio_capture.py       # Microphone ring buffer
├── audio_thread.py        # VAD + speaker recognition + music-mode state machine
├── audio_classifier.py    # YAMNet music/speech scores
├── speaker_recognizer.py  # SpeechBrain ECAPA wrapper (shared instance)
├── people_ui.py           # Manage People window
├── config.py              # Defaults
├── make_icon.py           # Generates AppIcon.icns for the bundle
├── requirements.txt       # Core dependencies
├── requirements-recognition.txt  # Optional recognition stack
├── Autofollow.app/        # Dock launcher (uses .venv)
└── yolov8n-pose.pt        # Model weights
```

## License

MIT License - Feel free to use and modify
