"""Configuration settings for the Autofollow app."""

# Output resolution (16:9)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_ASPECT_RATIO = 16 / 9

# Pose detection
CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL = 'yolov8n-pose.pt'   # nano=n, small=s, medium=m, large=l
DETECTION_SCALE = 0.5             # Run YOLO on this fraction of input resolution (0.25–1.0)
DETECTION_INTERVAL = 1            # Run pose detection every N frames (1 = every frame)
MAX_PERSONS = 5                   # Maximum simultaneous tracked people

# Framing
PADDING_RATIO = 0.15              # Padding around detected person bounding box
SHOT_TYPE = 'waist_up'           # 'full_body' | 'waist_up' | 'medium' | 'close_up'
MAX_ZOOM = 4.0                    # Maximum zoom factor
DEADZONE = 0.4                    # Horizontal deadzone (0–1): fraction of viewport where subject moves without panning

SHOT_TYPE_ZOOM = {
    'full_body': 1.0,
    'waist_up': 1.5,
    'medium': 2.0,
    'close_up': 2.5,
}

# PTZ smoothing
SMOOTHING_FACTOR = 0.05           # Minimum easing factor (close to target)
MAX_PAN_SPEED = 30                # Maximum pan movement in pixels per frame
MAX_ZOOM_SPEED = 0.05             # Maximum zoom change per frame

# Camera
CAMERA_INDEX = 0                  # Default camera device (0 = built-in)

# Tracking mode
TRACKING_MODE = 'primary'         # 'primary' (follow foreground person) | 'switcher' (virtual switching)

# Virtual switcher
SWITCH_MODE = 'crossfade'               # 'cut' | 'crossfade'
SWITCH_TRIGGER = 'time'           # 'time' | 'activity' | 'manual'
SWITCH_INTERVAL = 8.0             # Seconds between auto-switches (time trigger)
CROSSFADE_DURATION = 1.0          # Seconds for crossfade transition

# UI
SHOW_OVERLAY = False              # Show frame info overlay text
