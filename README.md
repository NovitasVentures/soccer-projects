# Soccer Ball Tracker

Real-time soccer ball detection and speed estimation from a YouTube broadcast stream, using YOLOv8 and a Kalman filter.

## How it works

1. **Stream ingestion** — yt-dlp resolves the YouTube URL and ffmpeg pipes raw 1080p frames directly into the tracker, bypassing any camera or screen-capture noise.
2. **Detection** — YOLOv8n runs on CUDA, inferring at 1280px. Only the detection spatially closest to the Kalman-predicted position is used, so false positives from white kits don't hijack the track.
3. **Tracking** — A Kalman filter with a time-aware transition matrix (`dt` updated each frame) maintains smooth position and velocity even when the ball is briefly occluded.
4. **Calibration** — On startup, click the left and right edges of the center circle (FIFA diameter: 18.3m) to establish pixels-per-meter for the current zoom level.
5. **Overlay** — Trail, velocity arrow, and speed in km/h are drawn on each frame. Speed label dims when the ball is stationary.

## Requirements

- NVIDIA GPU (tested on RTX 4060)
- Python 3.12
- CUDA-capable PyTorch

```bash
pip install ultralytics yt-dlp opencv-python numpy
```

ffmpeg must be on your PATH:

```bash
sudo apt install ffmpeg
```

## Usage

```bash
python ball_tracker.py
```

On startup, wait for a shot showing the center circle, then click its left and right edges to calibrate. Press `q` to quit.

## Calibration note

Calibration is valid for the current broadcast zoom level. If the director cuts to a significantly different zoom, re-run to recalibrate. A kickoff shot or any centered midfield view works well.

## License

MIT
