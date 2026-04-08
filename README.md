# Autofollow: Intelligent Video Framing App

A macOS application that intelligently crops video from a camera device using AI-powered pose detection. Automatically frames a person in a 16:9 medium close-up shot with smooth camera movements.

## Features

- **Real-time pose detection** using MediaPipe (Google's ML Kit equivalent)
- **4K video support** (up to 4K resolution input)
- **Intelligent framing** that automatically crops to 16:9 with optimal zoom
- **Smooth camera movements** using exponential smoothing to eliminate jitter
- **Live preview** with pose detection visualization
- **Video output** export to MP4 file format
- **Multi-camera support** for different camera devices

## Requirements

- macOS 10.14+
- Python 3.8+
- Webcam or video capture device

## Installation

### 1. Install Python 3.8+

If you don't have Python installed, download from [python.org](https://www.python.org/downloads/)

### 2. Create Virtual Environment & Install Dependencies

A virtual environment has been set up in the `.venv` directory with all dependencies pre-installed.

## Usage

### Quick Start (Using Launcher Script)

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
TARGET_ZOOM = 1.5           # Zoom level (1.5x = medium close-up)
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
├── main.py                 # Entry point and CLI
├── video_processor.py      # Main processing pipeline
├── pose_detector.py        # Pose detection (MediaPipe)
├── framing_engine.py       # Crop calculation and framing logic
├── smoothing.py            # Camera movement smoothing
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── README.md              # This file
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
- Increase `TARGET_ZOOM` for tighter framing
- Decrease `SMOOTHING_FACTOR` for smoother but slower response
- Adjust `PADDING_RATIO` for more/less head space

## Limitations

- Requires front-facing camera position
- Works best with single person in frame
- Pose detection confidence varies with lighting and body position
- Video output limited to approximately 1080p resolution

## Future Enhancements

- [ ] Multi-person tracking with person selection
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
