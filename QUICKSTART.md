# Autofollow — Quick Start

## First time

```bash
cd /Users/techbooth/Documents/Autofollow
./setup.sh
```

## Every time

```bash
./run.sh
```

…or double-click **Autofollow.app**.

The control panel appears right away; wait for the status bar to change from *Loading pose model…* to *Camera 0 ready*.

## Running a show

1. **Camera** — pick the input. The label next to it shows the resolution/fps it opened at.
2. **Display** — choose the projector/monitor, then **Open Fullscreen Output**. Esc, Q or double-click on the output closes it.
3. **Mode**
   - **Primary** — follows the closest person and hands off automatically. Good default for a single presenter or a panel.
   - **Time** — rotates between everyone on stage every *Interval* seconds.
   - **Manual** — press **P1 / P2 / …** to pick who's on screen.
   - **Disabled** — full wide shot, no tracking.
4. **Shot Type** and **Transition** can be changed live.
5. If people in the front row keep getting picked up, raise **Audience Exclusion** until they fall inside the yellow zone in **Diagnostics**.

Everything you set is remembered for next time.

## Command line

| Command | Purpose |
|---------|---------|
| `./run.sh` | Control panel (GUI) |
| `./run.sh --list-cameras` | Show available camera indices |
| `./run.sh --headless --output show.mp4` | Record to a file without the GUI |
| `./run.sh --headless --camera 1 --shot-type full_body` | Headless with options |
| `./run.sh --headless --no-preview --max-frames 300` | Process 300 frames, no window |

## Tuning (`config.py`)

| Setting | Effect |
|---------|--------|
| `SHOT_TYPE` | Default shot: `full_body`, `waist_up`, `medium`, `close_up` |
| `MAX_ZOOM` | How tight the camera is allowed to go |
| `SMOOTHING` | 0 = responsive, 1 = very smooth/slow |
| `DEADZONE` | How far the subject can drift before the camera pans |
| `PRIMARY_DWELL_SECONDS` | Minimum time on a subject before a hand-off |
| `DETECTION_INTERVAL` / `DETECTION_SCALE` | Trade detection accuracy for speed |
| `CAPTURE_WIDTH` / `CAPTURE_HEIGHT` | Ask the camera for a specific mode (0 = its default) |

## Troubleshooting

- **No cameras found** → System Settings → Privacy & Security → Camera: allow Terminal / Autofollow.
- **"No signal from camera — reconnecting…"** → the device dropped out; it reconnects on its own once it's back.
- **Choppy** → raise `DETECTION_INTERVAL` to 3, or lower `DETECTION_SCALE` to 0.4.
- **Too tight / too loose** → change Shot Type, or adjust `MAX_ZOOM`.
- **Dock app won't open** → re-run `./setup.sh`.
