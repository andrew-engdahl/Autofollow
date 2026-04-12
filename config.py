"""Configuration settings for the Autofollow app."""

# Video settings
VIDEO_WIDTH = 2560  # 4K width (supports up to 4K)
VIDEO_HEIGHT = 1440  # 4K height
VIDEO_FPS = 30
VIDEO_CODEC = 'mp4v'

# Output settings
OUTPUT_WIDTH = 1920  # 16:9 aspect ratio
OUTPUT_HEIGHT = 1080
OUTPUT_ASPECT_RATIO = 16 / 9  # 16:9

# Pose detection settings
CONFIDENCE_THRESHOLD = 0.5
POSE_DETECTION_MODEL = 'pose'  # MediaPipe pose detection model

# Framing settings
PADDING_RATIO = 0.15  # 15% padding around detected person
SHOT_TYPE = 'waist_up'  # Type of shot: 'full_body', 'waist_up', 'medium', 'close_up'
MAX_ZOOM = 4.0  # Maximum zoom factor (prevents over-zooming)

# Shot type zoom targets (before MAX_ZOOM clamping)
SHOT_TYPE_ZOOM = {
    'full_body': 1.0,   # Show entire person, no zoom
    'waist_up': 1.5,    # Show waist/hips to head
    'medium': 2.0,      # Show chest/shoulders to head
    'close_up': 2.5,    # Show head and shoulders only
}

MIN_FACE_SCALE = 0.15  # Minimum portion of frame that should be person
MAX_FACE_SCALE = 0.35  # Maximum portion of frame that should be person

# Smoothing settings
SMOOTHING_FACTOR = 0.05  # 0-1, lower = more smoothing (% of movement per frame)
MAX_PAN_SPEED = 50  # pixels per frame
MAX_ZOOM_SPEED = 0.05  # scale units per frame

# Camera settings
CAMERA_INDEX = 0  # Default camera device (0 = built-in webcam)
AUTO_CAMERA_DETECT = True  # Try to find best available camera

# UI settings
SHOW_OVERLAY = False  # Show frame info overlay text

# Performance settings
SKIP_FRAMES = 0  # Process every Nth frame (0 = process every frame)
DETECTION_INTERVAL = 1  # Run pose detection every N frames
