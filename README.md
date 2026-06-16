# soccer-projects

Real-time soccer match analysis pipeline using computer vision. Runs on a Windows workstation with WSL2, targeting broadcast video input (recorded or live).

## Hardware

- **GPU**: RTX 4060 (8GB VRAM) — all inference runs here via CUDA
- **CPU**: i7-14700 / **RAM**: 32GB
- **OS**: WSL2 (Ubuntu) for all ML/CV work; Windows only for capture
- **Python**: `pip` + `venv` — no conda, no uv

## Architecture

```
[Source]  YouTube stream (yt-dlp) or OBS clip
               ↓
          Frame ingestion (OpenCV VideoCapture / ffmpeg pipe)
               ↓
          Player + ball detection (YOLOv8, CUDA)
               ↓
          Tracking (ByteTrack, built into ultralytics)
               ↓
          Homography → pitch coordinates (roboflow/sports)
               ↓
          Per-module analysis (geometry, Kalman, Voronoi, etc.)
               ↓
          Overlay rendering (OpenCV draw calls)
               ↓
          Output: cv2.imshow via WSLg, or write to file
```

## Projects

| Folder | Description | Status |
|---|---|---|
| [ball_tracker](./ball_tracker/) | Real-time ball detection and speed estimation from a YouTube stream using YOLOv8 + Kalman filter | Working |

## Planned Modules

Build in this sequence — each stage unblocks the next:

### Stage 1 — Ball-centric (no homography needed)
| # | Module | Key dependency |
|---|---|---|
| 1 | Ball velocity vector | Kalman filter |
| 4 | Shot power estimator | Module 1 + velocity threshold |

### Stage 2 — Homography unlock
| # | Module | Key dependency |
|---|---|---|
| 2 | Offside line | Homography + team separation |
| 3 | Press heatmap | Homography + all player positions |
| 6 | Pass network | Homography + possession attribution |
| 7 | Space creation (Voronoi) | Homography + scipy.spatial |

### Stage 3 — Player-centric
| # | Module | Key dependency |
|---|---|---|
| 5 | Keeper positioning grade | Goal post detection + module 2 geometry |
| 9 | Fatigue proxy | Stable player ID across full match |
| 10 | Set piece analyzer | Ball stationary detection + geometry |

### Stage 4 — ML-assisted
| # | Module | Key dependency |
|---|---|---|
| 8 | Danger zone classifier | StatsBomb open data + sklearn |

## Key Design Decisions

**Detection model**: YOLOv8n/s. COCO class 32 = sports ball. Expect ~70% recall in open play, ~30% during fast kicks. Kalman filter bridges dropout frames.

**Team separation**: K-means (k=3: team A, team B, referee) on HSV histogram of each player bounding box. Falls apart with color-similar kits.

**Homography**: `roboflow/sports` — YOLOv8 keypoint detection on pitch markings → `cv2.findHomography` → pitch coordinates. Re-solved every 15 frames to handle broadcast pans and zoom. `SoccerFieldConfiguration` provides the canonical 105×68m pitch template. Hold last valid H matrix up to 30 frames on keypoint dropout.

**Speed calibration**: center circle diameter = 18.3m. Click left/right edges interactively to set `PIXELS_PER_METER`.

**Scene cut detection**: frame-diff threshold resets Kalman state to prevent phantom velocity vectors after broadcast cuts.

## Known Failure Modes

| Situation | Symptom | Mitigation |
|---|---|---|
| Ball occluded by players | Phantom Kalman prediction drifts | Reset on scene cut |
| Camera pan mid-play | Homography goes stale | Re-solve every 15 frames |
| Slow-motion replay | Speed reads 3–4x too low | Flag if speed > 150 km/h |
| Similar kit colors | Team K-means misclassifies | Manual color override |
| Fewer than 4 pitch landmarks | Homography solve fails | Hold last valid H matrix |

## What Is Not Attempted

- **Player names**: jersey number OCR at broadcast resolution is ~30% accuracy
- **Live capture**: build vision logic on recorded clips first
- **Multi-camera**: single broadcast feed only
- **Sub-frame accuracy**: speed/position numbers are directionally correct, not precise

## License

MIT
