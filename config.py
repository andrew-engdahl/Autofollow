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
TARGET_ZOOM = 1.5  # 1.5x zoom for medium close-up shot
MIN_FACE_SCALE = 0.15  # Minimum portion of frame that should be person
MAX_FACE_SCALE = 0.7  # Maximum portion of frame that should be person

# Smoothing settings
SMOOTHING_FACTOR = 0.15  # 0-1, lower = more smoothing (15% of movement per frame)
MAX_PAN_SPEED = 100  # pixels per frame
MAX_ZOOM_SPEED = 0.05  # scale units per frame

# Camera settings
CAMERA_INDEX = 0  # Default camera device (0 = built-in webcam)
AUTO_CAMERA_DETECT = True  # Try to find best available camera

# Performance settings
SKIP_FRAMES = 0  # Process every Nth frame (0 = process every frame)
DETECTION_INTERVAL = 1  # Run pose detection every N frames
