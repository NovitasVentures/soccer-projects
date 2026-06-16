# Lessons Learned — Ball Tracker

## Signal quality is everything

Pointing a webcam at a TV (camera → TV → broadcast) produced terrible results (~4/10). Switching to a direct YouTube stream via yt-dlp/ffmpeg immediately jumped quality to the point where tracking was visibly useful. Eliminate artifact layers before tuning any model parameters.

## Calibration reference matters

Clicking the TV bezel gave an inaccurate pixels-per-meter because the bezel-to-screen mapping depends on viewing angle and distance. The center circle (FIFA diameter: 18.3m) is a known physical reference that appears in-frame on nearly every broadcast shot of midfield play — use field markings, not camera hardware.

## Default camera resolution is not the max resolution

OpenCV defaulted to 640×480 (YUYV) even though the Anker C200 supports 1920×1080. Must explicitly set `FOURCC=MJPG` and the target resolution before the first frame read — YUYV tops out at 640×480 on this camera. A soccer ball at 640×480 is ~6px wide; at 1080p it's ~20px, which is the difference between reliable detection and near-zero.

## Kalman velocity units must match actual elapsed time

The original implementation had `vx/vy` in pixels-per-frame but multiplied by camera FPS to get speed — when inference ran slower than the camera, the speed estimate was off by the ratio. The fix: use real `dt` (seconds) in the transition matrix so `vx/vy` come out in pixels/second regardless of inference rate.

## Multiple detections need spatial continuity, not just confidence

When multiple "sports ball" detections fire per frame (white kits, player heads), taking the highest-confidence one causes the tracker to jump across the field. Selecting the detection closest to the Kalman-predicted position keeps the track locked on the true ball even at lower confidence.

## GPU warmup cost is real

First inference on CUDA is ~10× slower than subsequent ones due to JIT compilation. Run a dummy inference before the main loop or the first frame's `dt` will be wildly wrong and poison the Kalman velocity state.

## YOLOv8n vs YOLOv8s

On an RTX 4060, nano runs at ~130fps vs ~75fps for small, with acceptable detection quality on a 1080p broadcast frame. For this use case (single ball, high contrast against green grass) nano is the right tradeoff.
