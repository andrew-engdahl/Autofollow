# Autofollow - Quick Start Guide

## ✅ Setup Complete!

All dependencies have been installed in the virtual environment. You're ready to use Autofollow.

## Getting Started in 3 Steps

### 1. Navigate to the Project
```bash
cd /Users/techbooth/Documents/Autofollow
```

### 2. Run the App
```bash
./run.sh
```

Press **`q`** to exit the preview.

### 3. (Optional) Save Video Output
```bash
./run.sh --output my_video.mp4
```

## Common Commands

| Command | Purpose |
|---------|---------|
| `./run.sh` | Live preview of intelligent framing |
| `./run.sh --list-cameras` | Find available camera devices |
| `./run.sh --camera 1` | Use alternate camera (device 1) |
| `./run.sh --output video.mp4` | Save processed video to file |
| `./run.sh --max-frames 300` | Process only 300 frames |

## How It Works

1. **Captures video** from your camera (up to 4K support)
2. **Detects your pose** using Google's MediaPipe ML Kit
3. **Auto-frames you** in 16:9 medium close-up
4. **Smooths movement** to create professional camera movement
5. **Outputs** high-quality 1080p (16:9) video

## Configuration

Edit `config.py` to customize:
- **SMOOTHING_FACTOR** (0.15) - Lower = smoother but slower response
- **TARGET_ZOOM** (1.5) - Higher = tighter frame
- **PADDING_RATIO** (0.15) - Head space padding around person
- **MAX_PAN_SPEED** (100) - Maximum camera pan pixels/frame
- **MAX_ZOOM_SPEED** (0.05) - Maximum zoom change per frame

## Troubleshooting

### Camera not found
```bash
./run.sh --list-cameras
# Use found camera index:
./run.sh --camera 1
```

### Poor detection (low light)
→ Increase lighting or adjust `CONFIDENCE_THRESHOLD` in config.py

### Choppy output
→ Increase `SMOOTHING_FACTOR` in config.py (try 0.25-0.35)

### Tight framing
→ Decrease `TARGET_ZOOM` in config.py (try 1.2)

## System Information

- **Python Version**: 3.13.3
- **Virtual Environment**: `.venv`
- **Packages Installed**:
  - OpenCV 4.13.0
  - MediaPipe (latest)
  - NumPy 2.4.4

## Next Steps

1. **Test it**: Run `./run.sh` to see it in action
2. **Adjust settings**: Edit `config.py` to fine-tune framing
3. **Record videos**: Use `--output video.mp4` to save
4. **Explore code**: Check individual modules:
   - `pose_detector.py` - ML Kit pose detection
   - `framing_engine.py` - Intelligent crop logic
   - `smoothing.py` - Camera movement smoothing
   - `config.py` - All tunable parameters

## Need Help?

- Check README.md for detailed documentation
- Review config.py comments for detailed options
- Check individual .py files for function documentation

Happy filming! 🎥
