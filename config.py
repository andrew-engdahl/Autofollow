"""Configuration settings for the Autofollow app."""

# Output resolution (16:9)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_ASPECT_RATIO = 16 / 9

# Pose detection
CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL = 'yolov8n-pose.pt'    # nano=n, small=s, medium=m, large=l
DETECTION_SCALE = 0.5             # Run YOLO on this fraction of input resolution (0.25–1.0)
DETECTION_INTERVAL = 1            # Run pose detection every N frames (1 = every frame)
MAX_PERSONS = 12                  # Maximum simultaneous tracked people

# Framing
PADDING_RATIO = 0.15              # Padding around detected person bounding box
SHOT_TYPE = 'waist_up'            # 'full_body' | 'waist_up' | 'medium' | 'close_up'
MAX_ZOOM = 4.0                    # Maximum zoom factor
DEADZONE = 0.4                    # Horizontal deadzone (0–1): fraction of viewport where subject moves without panning

# Framing settings
PADDING_RATIO = 0.15  # 15% padding around detected person
SHOT_TYPE = 'waist_up'  # Type of shot: 'full_body', 'waist_up', 'medium', 'close_up'
MAX_ZOOM = 4.0  # Maximum zoom factor (prevents over-zooming)
GROUP_FRAMING = True  # Frame multiple people together if detected

# Shot type zoom targets (before MAX_ZOOM clamping)
SHOT_TYPE_ZOOM = {
    'full_body': 1.0,
    'waist_up': 1.5,
    'medium': 2.0,
    'close_up': 2.25,
}

# PTZ smoothing
# SMOOTHING: 0 = minimal extra smoothing (still has significant baseline),
#            1 = very smooth / noticeably delayed movement.
# Panning (X) is the primary motion axis; tilt (Y) and zoom (Z) are secondary
# and are smoothed much more aggressively to keep them nearly static.
SMOOTHING = 0.3                   # 0–1 user-facing smoothing dial
MAX_PAN_SPEED = 15                # Maximum pan movement in pixels per frame
MAX_TILT_SPEED = 3                # Maximum tilt movement in pixels per frame (slow)
MAX_ZOOM_SPEED = 0.015            # Maximum zoom change per frame (very slow)

# Camera
CAMERA_INDEX = 0                  # Default camera device (0 = built-in)

# Tracking mode
TRACKING_MODE = 'primary'         # 'primary' (follow foreground person) | 'switcher' (virtual switching)

# Virtual switcher
SWITCH_MODE = 'crossfade'         # 'cut' | 'crossfade'
SWITCH_TRIGGER = 'time'           # 'time' | 'activity' | 'manual'
SWITCH_INTERVAL = 8.0             # Seconds between auto-switches (time trigger)
CROSSFADE_DURATION = 1.0          # Seconds for crossfade transition

# Foreground audience exclusion
# Detections whose bbox bottom-edge is in the lower FOREGROUND_EXCLUSION_Y fraction
# of the frame (0.0 = disabled, 1.0 = exclude everything).  Audience members standing
# in front of the stage are typically in the lower portion of the frame; performers on
# stage are higher up.  Set to 0.0 to disable.
FOREGROUND_EXCLUSION_Y = 0.20     # fraction of frame height from bottom to ignore

# Virtual switcher displacement gate
# A subject switch is only triggered when the new subject's center is at least this
# fraction of the current crop width away from the current shot center.
# 0.0 = switch on any candidate; 0.75 = require significant displacement before switching.
SWITCHER_MIN_DISPLACEMENT_RATIO = 0.75

# UI
SHOW_OVERLAY = False              # Show frame info overlay text
SHOW_DIAGNOSTICS = False          # Overlay pose skeleton and per-person tracking circles
