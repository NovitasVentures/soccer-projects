import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import subprocess
import time

YOUTUBE_URL = "https://www.youtube.com/watch?v=zQ5x1AlImTI"
FRAME_W = 1920
FRAME_H = 1080
INFER_SIZE = 1280

# FIFA center circle diameter: 9.15m radius = 18.3m diameter
CENTER_CIRCLE_D_M = 18.3

PIXELS_PER_METER = None   # set by calibrate()

# Kalman tuning
MEAS_NOISE = 25.0
PROC_NOISE_POS = 0.5
PROC_NOISE_VEL = 0.05
SPEED_SMOOTH_N = 8


def make_kalman(dt):
    kf = cv2.KalmanFilter(4, 2)  # state: x,y,vx,vy  measure: x,y
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0,  1, 0],
        [0, 0,  0, 1]], np.float32)
    kf.processNoiseCov = np.diag([
        PROC_NOISE_POS, PROC_NOISE_POS,
        PROC_NOISE_VEL, PROC_NOISE_VEL]).astype(np.float32)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEAS_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
    return kf


class VideoStream:
    """Pipes a YouTube video through yt-dlp + ffmpeg into raw BGR frames."""

    def __init__(self, url, width, height):
        print("Resolving stream URL...")
        result = subprocess.run(
            ["yt-dlp", "-f", "137", "--get-url", url],
            capture_output=True, text=True, check=True
        )
        stream_url = result.stdout.strip()
        print("Starting ffmpeg pipe...")
        self._proc = subprocess.Popen([
            "ffmpeg", "-i", stream_url,
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-an", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.width = width
        self.height = height
        self.fps = 25.0  # YouTube 1080p stream is 25fps
        self._frame_bytes = width * height * 3
        # Read and discard first few frames to let ffmpeg stabilize
        for _ in range(5):
            self._proc.stdout.read(self._frame_bytes)
        print("Stream ready.")

    def read(self):
        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:
            return None
        return np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3)).copy()

    def release(self):
        self._proc.terminate()


def calibrate(stream):
    """Click left then right edge of the center circle to set pixels-per-meter.
    Center circle diameter = 18.3m. Works on any shot that shows the center circle."""
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            if len(clicks) == 0 or abs(x - clicks[0]) > 20:
                clicks.append(x)
                print(f"  click {len(clicks)}: x={x}")

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibrate", on_click)
    print("CALIBRATION: wait for a shot showing the center circle.")
    print("Click LEFT edge of center circle, then RIGHT edge.")

    frame = None
    while len(clicks) < 2:
        new_frame = stream.read()
        if new_frame is not None:
            frame = new_frame.copy()
        if frame is None:
            continue
        disp = frame.copy()
        msg = "Click LEFT edge of center circle" if len(clicks) == 0 else "Now click RIGHT edge of center circle"
        cv2.putText(disp, msg, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(disp, f"clicks: {len(clicks)}/2", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        for cx in clicks:
            cv2.line(disp, (cx, 0), (cx, disp.shape[0]), (0, 255, 0), 2)
        cv2.imshow("Calibrate", disp)
        cv2.waitKey(16)

    cv2.setMouseCallback("Calibrate", lambda *a: None)  # stop accepting clicks

    pixel_width = abs(clicks[1] - clicks[0])
    ppm = pixel_width / CENTER_CIRCLE_D_M
    print(f"Center circle spans {pixel_width}px = {CENTER_CIRCLE_D_M}m  =>  {ppm:.2f} px/m")

    disp = frame.copy()
    cv2.putText(disp, f"Calibrated: {ppm:.2f} px/m", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Calibrate", disp)
    cv2.waitKey(1500)
    cv2.destroyWindow("Calibrate")
    return ppm


stream = VideoStream(YOUTUBE_URL, FRAME_W, FRAME_H)
PIXELS_PER_METER = calibrate(stream)

model = YOLO("yolov8n.pt")
model.to("cuda")
model(np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8), verbose=False)

scale = INFER_SIZE / max(FRAME_W, FRAME_H)
infer_w = int(FRAME_W * scale)
infer_h = int(FRAME_H * scale)

# Measure inference dt
warmup_times = []
dummy = np.zeros((infer_h, infer_w, 3), dtype=np.uint8)
for _ in range(5):
    t0 = time.perf_counter()
    model(dummy, classes=[32], conf=0.25, verbose=False, imgsz=INFER_SIZE)
    warmup_times.append(time.perf_counter() - t0)
measured_dt = float(np.median(warmup_times))
print(f"Inference dt: {measured_dt*1000:.1f}ms  ({1/measured_dt:.1f} fps)")

kf = make_kalman(measured_dt)
kalman_initialized = False
trail = deque(maxlen=30)
speed_history = deque(maxlen=SPEED_SMOOTH_N)
frame_times = deque(maxlen=30)
last_t = time.perf_counter()

try:
    while True:
        frame = stream.read()
        if frame is None:
            print("Stream ended.")
            break

        t0 = time.perf_counter()
        dt = t0 - last_t
        last_t = t0

        kf.transitionMatrix[0, 2] = dt
        kf.transitionMatrix[1, 3] = dt

        small = cv2.resize(frame, (infer_w, infer_h))
        results = model(small, classes=[32], conf=0.25, verbose=False, imgsz=INFER_SIZE)

        detected = False
        best_cx, best_cy = None, None
        best_dist = float("inf")
        pred_x = float(kf.statePost[0, 0]) if kalman_initialized else None
        pred_y = float(kf.statePost[1, 0]) if kalman_initialized else None

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2 / scale
            cy = (y1 + y2) / 2 / scale
            if pred_x is None:
                best_cx, best_cy = cx, cy
                break
            dist = (cx - pred_x) ** 2 + (cy - pred_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cx, best_cy = cx, cy

        if best_cx is not None:
            measurement = np.array([[np.float32(best_cx)], [np.float32(best_cy)]])
            if not kalman_initialized:
                kf.statePre  = np.array([[best_cx], [best_cy], [0.0], [0.0]], np.float32)
                kf.statePost = np.array([[best_cx], [best_cy], [0.0], [0.0]], np.float32)
                kalman_initialized = True
            kf.correct(measurement)
            detected = True

        if kalman_initialized:
            predicted = kf.predict()
            sx, sy = int(predicted[0, 0]), int(predicted[1, 0])
            vx, vy = float(predicted[2, 0]), float(predicted[3, 0])  # px/s

            trail.append((sx, sy))

            speed_ms = np.sqrt(vx**2 + vy**2) / PIXELS_PER_METER
            speed_history.append(speed_ms)
            smoothed_ms = float(np.mean(speed_history))

            for i in range(1, len(trail)):
                alpha = i / len(trail)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, trail[i-1], trail[i], color, 2)

            ARROW_SCALE = 0.3
            cv2.arrowedLine(frame, (sx, sy),
                            (int(sx + vx * ARROW_SCALE), int(sy + vy * ARROW_SCALE)),
                            (0, 255, 255), 2, tipLength=0.3)

            dot_color = (0, 255, 0) if detected else (0, 100, 255)
            cv2.circle(frame, (sx, sy), 8, dot_color, 2)

            smoothed_kph = smoothed_ms * 3.6
            label_color = (0, 255, 255) if smoothed_kph > 2.0 else (0, 140, 140)
            cv2.putText(frame, f"{smoothed_kph:.1f} km/h",
                        (sx + 12, sy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

        frame_times.append(time.perf_counter() - t0)
        display_fps = 1.0 / (sum(frame_times) / len(frame_times))
        cv2.putText(frame, f"{display_fps:.1f} fps  {PIXELS_PER_METER:.1f} px/m", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Ball Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    stream.release()
    cv2.destroyAllWindows()
