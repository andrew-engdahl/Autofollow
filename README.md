# Autofollow: Intelligent Video Framing App

A macOS application that intelligently crops video from a camera device using AI-powered pose detection. Automatically frames a person in a 16:9 medium close-up shot with smooth camera movements.

## Features

- **Real-time pose detection** using MediaPipe (Google's ML Kit equivalent)
- **4K video support** (up to 4K resolution input)
- **Intelligent framing** that automatically crops to 16:9 with optimal zoom
- **Multi-person framing** automatically frames multiple people together with centered panning
- **Smooth camera movements** using exponential smoothing to eliminate jitter
- **Live preview** with pose detection visualization
- **Video output** export to MP4 file format
- **Multi-camera support** for different camera devices
- **People profiles** — save reference images of named individuals; InsightFace
  matches them on-camera and biases the auto-switcher toward higher-priority
  people (e.g. the pastor at a church)

## Requirements

- macOS 10.14+
- Python 3.10+
- Webcam or video capture device

## Installation

### 1. Install Python 3.10+

If you don't have Python installed, download from [python.org](https://www.python.org/downloads/)

### 2. One-Click Setup

Run the setup script once. It creates the virtual environment and installs all dependencies automatically — safe to re-run, it skips steps that are already done.

```bash
./setup.sh
```

That's it. After setup completes, use `./run.sh` to launch the app.

## Usage

### Quick Start

```bash
./run.sh
```

Or with options:
```bash
./run.sh --output video.mp4
./run.sh --list-cameras
```

### Alternative: Manual Activation

If you prefer to activate the virtual environment manually:

```bash
cd /Users/techbooth/Documents/Autofollow
source .venv/bin/activate
python main.py
```

### Live Preview (Default)

```bash
./run.sh
```

Press `q` to quit the preview window.

### List Available Cameras

```bash
./run.sh --list-cameras
```

### Use Specific Camera

```bash
./run.sh --camera 1
```

### Save Output to File

```bash
./run.sh --output output.mp4
```

### Full Example with Options

```bash
./run.sh --camera 0 --output cropped_video.mp4 --max-frames 1000
```

## Command Line Options

```
--camera DEVICE_ID       Camera device index (default: 0)
--list-cameras          List available camera devices  
--output FILE           Save output to MP4 file
--max-frames N          Limit frames processed to N
--no-preview            Disable live preview
--help                  Show this help message
```

## How It Works

### 1. **Pose Detection**
   - Uses MediaPipe to detect human pose landmarks (33 key points)
   - Identifies the person's position and scale in the frame
   - Gracefully handles frames where no person is detected

### 2. **Intelligent Framing**
   - Calculates optimal crop region centered on detected person
   - Maintains 16:9 aspect ratio for standard video format
   - Applies 1.5x zoom for a medium close-up shot
   - Adds 15% padding around person for comfortable framing

### 3. **Smooth Movement**
   - Applies exponential smoothing to camera position and zoom
   - Prevents jittery movements frame-to-frame
   - Limits pan speed (max 100 pixels/frame) and zoom speed
   - Creates cinematic, professional-grade camera movements

### 4. **Output**
   - Crops and resizes to 1920x1080 (16:9 1080p)
   - Supports live preview with visualization
   - Optionally saves to MP4 file

## Configuration

Edit `config.py` to customize behavior:

```python
# Video settings
VIDEO_WIDTH = 2560          # Input width (supports 4K)
OUTPUT_WIDTH = 1920         # Output width (16:9)
OUTPUT_HEIGHT = 1080

# Framing settings
SHOT_TYPE = 'medium'        # Shot type: 'full_body', 'waist_up', 'medium', 'close_up'
MAX_ZOOM = 2.5              # Maximum zoom factor (prevents over-zooming)
PADDING_RATIO = 0.15        # 15% padding around person

# Smoothing settings
SMOOTHING_FACTOR = 0.15     # Lower = more smoothing
MAX_PAN_SPEED = 100         # Pixels per frame
MAX_ZOOM_SPEED = 0.05       # Scale units per frame
```

## Performance Notes

- **Real-time processing**: Typically 25-30 FPS on modern Macs
- **GPU acceleration**: Uses CPU; can be optimized with CoreML
- **Memory usage**: ~200-300 MB typical
- **Frame skipping**: Set `SKIP_FRAMES` in config.py to process fewer frames if needed

## Troubleshooting

### Camera not found
```bash
python main.py --list-cameras
# Use camera number in output:
python main.py --camera 1
```

### Poor pose detection
- Ensure good lighting
- Position subject with clear view of body
- Adjust `CONFIDENCE_THRESHOLD` in `config.py` (lower = more sensitive)

### Choppy output
- Reduce input resolution
- Increase `SMOOTHING_FACTOR` in `config.py`
- Increase `MAX_PAN_SPEED` for faster response

### Output file issues
- Ensure output directory exists and is writable
- Use absolute path: `/Users/username/Desktop/video.mp4`

## Project Structure

```
Autofollow/
├── setup.sh               # One-click setup (venv + dependencies)
├── run.sh                 # Launch the app
├── main.py                # Entry point and CLI
├── video_processor.py     # Main processing pipeline
├── pose_detector.py       # Pose detection (MediaPipe)
├── framing_engine.py      # Crop calculation and framing logic
├── smoothing.py           # Camera movement smoothing
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Advanced Features

### Custom Camera Settings

Modify the camera initialization in `VideoProcessor.initialize_camera()`:

```python
# Set custom resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 4K width
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4K height
self.cap.set(cv2.CAP_PROP_FPS, 30)
```

### Real-time Parameter Tuning

Edit config values while running to experiment:
- Use `--shot-type close_up` for tighter framing, or increase `MAX_ZOOM` in config.py
- Decrease `SMOOTHING_FACTOR` for smoother but slower response
- Adjust `PADDING_RATIO` for more/less head space

## Limitations

- Requires front-facing camera position
- Works best with single person in frame
- Pose detection confidence varies with lighting and body position
- Video output limited to approximately 1080p resolution

## People Profiles

Click **Manage People** in the Output section to open the profiles window.

1. Click **Add Person…**.
2. Enter a name and a priority (0–10). Higher priority = the auto-switcher
   prefers and dwells longer on this person.
3. Add reference images either from disk (**Add from file…**) or by snapping
   the current camera frame (**Capture from camera**).
4. Save. The first profile triggers a one-time download of the InsightFace
   `buffalo_l` model (~300 MB) to `~/.insightface/models/`.

Once a profile is embedded, the diagnostics overlay shows the matched name
(e.g. `★8 Pastor Mike`) on the body bbox in place of `personN`.

Profiles, images, and embeddings live in
`~/Library/Application Support/Autofollow/profiles/`.

## Future Enhancements

- [ ] Audio capture + speaker recognition (give known voices switcher priority)
- [ ] GPU acceleration with CoreML
- [ ] Face detection for better head positioning  
- [ ] Custom framing presets (tight close-up, medium shot, wide shot)
- [ ] Real-time parameter UI
- [ ] Scene detection and adaptive framing
- [ ] 4K output support

## License

MIT License - Feel free to use and modify

## Support

For issues or questions, check the troubleshooting section or review config.py settings.
