"""Configuration settings for the Autofollow app.

Values here are the startup defaults.  Most of them can be changed live from
the control panel, and those choices are remembered between launches.
"""

# Output resolution (16:9)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_ASPECT_RATIO = 16 / 9

# Camera
CAMERA_INDEX = 0                  # Default camera device (0 = built-in)
CAPTURE_WIDTH = 0                 # Requested capture size; 0 = leave the camera at its default.
CAPTURE_HEIGHT = 0                # The driver picks the closest mode it supports.
# The input resolution is never assumed: it is detected from the frames the
# camera actually delivers (re-detected if it changes mid-stream), and the
# framing pipeline adapts to it.  Constants below that are given in pixels or
# zoom units are *reference* values tuned for a 1280×720 input; they are scaled
# automatically for the detected size (see geometry.py), so a 4K camera gets
# the same on-screen motion as a 720p one without retuning.

# Pose detection
CONFIDENCE_THRESHOLD = 0.7
YOLO_MODEL = 'yolov8n-pose.pt'    # nano=n, small=s, medium=m, large=l
DETECTION_WIDTH = 640             # Downscale input to this width for YOLO (0 = full resolution)
DETECTION_INTERVAL = 2            # Run pose detection every N frames (1 = every frame)
MAX_PERSONS = 10                  # Maximum simultaneous tracked people
TRACK_DROPOUT_SECONDS = 2.0       # Forget a person after they have been unseen this long

# Framing
PADDING_RATIO = 0.15              # Padding below the shot's bottom landmark (fraction of body height)
SHOT_TYPE = 'waist_up'            # 'full_body' | 'waist_up' | 'medium' | 'close_up'
MAX_ZOOM = 4.0                    # Maximum zoom (relative to the output size, i.e. max upscale factor)
DEADZONE = 0.4                    # Horizontal deadzone (0–1): fraction of viewport where subject moves without panning

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
# The smoother works on the crop center + zoom, so zooms stay anchored on the
# subject.  Panning (X) is the primary motion axis; tilt (Y) and zoom (Z) are
# secondary, with their own deadzones, and are smoothed much more aggressively
# to keep them nearly static.
SMOOTHING = 0.5                  # 0–1 user-facing smoothing dial
# Pan, tilt and zoom speeds all scale with how far the shot is pushed in, so
# a tight shot keeps up with its subject the same way the wide shot does.
MAX_PAN_SPEED = 15                # Max pan on the wide shot, reference (720p) px/frame (× zoom when pushed in)
MAX_TILT_SPEED = 3                # Max tilt on the wide shot, reference px/frame (× zoom when pushed in)
MAX_ZOOM_SPEED = 0.012            # Max zoom change per frame as a fraction of the current zoom

# Tracking mode
TRACKING_MODE = 'primary'         # 'primary' (follow foreground person) | 'switcher' (virtual switching)

# Primary Focus mode
PRIMARY_DWELL_SECONDS = 3.0       # Minimum time on a subject before another can take over
PRIMARY_REACQUIRE_DELAY = 1.0     # Seconds after a lost subject's track expires before picking someone else

# Virtual switcher
SWITCH_MODE = 'crossfade'         # 'cut' | 'crossfade'
SWITCH_TRIGGER = 'time'           # 'time' | 'activity' | 'manual'
SWITCH_INTERVAL = 8.0             # Seconds between auto-switches (time trigger)
CROSSFADE_DURATION = 1.0          # Seconds for crossfade transition

# Foreground audience exclusion
# Detections whose torso center is in the lower FOREGROUND_EXCLUSION_Y fraction
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
SHOW_OVERLAY = False              # Show frame info overlay text (headless mode)
SHOW_DIAGNOSTICS = True           # Draw skeleton overlays in the diagnostics preview
