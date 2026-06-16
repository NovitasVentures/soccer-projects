import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import threading
import time

CAMERA_INDEX = 1
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
INFER_SIZE = 1280
TV_WIDTH_M = 1.218        # 55" 16:9 TV physical width in meters

PIXELS_PER_METER = None   # set by calibrate()

# Kalman tuning
# Position noise: how much we trust the detector (lower = smoother position)
MEAS_NOISE = 25.0
# Process noise: how much we expect velocity to change frame-to-frame
# Lower = velocity estimate coasts longer without a detection
PROC_NOISE_POS = 0.5
PROC_NOISE_VEL = 0.05

# Speed smoothing: rolling average over this many frames to dampen spikes
SPEED_SMOOTH_N = 8


def make_kalman(dt):
    """Build Kalman filter with transition matrix matched to actual frame dt."""
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


class CameraStream:
    """Grabs frames on a background thread so inference never waits on V4L2."""

    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index} — check usbipd attach")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {actual_w}x{actual_h} @ {self.fps:.0f}fps")
        ret, self.frame = self.cap.read()
        if not ret:
            raise RuntimeError("First frame grab failed")
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


def calibrate(stream):
    """Click left edge then right edge of the TV to compute pixels-per-meter."""
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) == 0 or abs(x - clicks[0]) > 50:
                clicks.append(x)
                print(f"  click {len(clicks)}: x={x}")

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibrate", on_click)

    print("CALIBRATION: click the LEFT edge of the TV, then the RIGHT edge.")

    while len(clicks) < 2:
        frame = stream.read()
        msg = "Click LEFT edge of TV" if len(clicks) == 0 else "Now click RIGHT edge of TV"
        cv2.putText(frame, msg, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"clicks: {len(clicks)}/2", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        for cx in clicks:
            cv2.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 255, 0), 2)
        cv2.imshow("Calibrate", frame)
        cv2.waitKey(16)

    pixel_width = abs(clicks[1] - clicks[0])
    ppm = pixel_width / TV_WIDTH_M
    print(f"TV spans {pixel_width}px = {TV_WIDTH_M}m  =>  {ppm:.1f} px/m")

    frame = stream.read()
    cv2.putText(frame, f"Calibrated: {ppm:.1f} px/m", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Calibrate", frame)
    cv2.waitKey(1500)
    cv2.destroyWindow("Calibrate")

    return ppm


stream = CameraStream(CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT)
PIXELS_PER_METER = calibrate(stream)

model = YOLO("yolov8n.pt")
model.to("cuda")
model(np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8), verbose=False)

full_h, full_w = stream.read().shape[:2]
scale = INFER_SIZE / max(full_w, full_h)
infer_w = int(full_w * scale)
infer_h = int(full_h * scale)

# Measure real inference dt over a few warmup frames before building Kalman
warmup_times = []
for _ in range(5):
    t0 = time.perf_counter()
    model(cv2.resize(stream.read(), (infer_w, infer_h)),
          classes=[32], conf=0.25, verbose=False, imgsz=INFER_SIZE)
    warmup_times.append(time.perf_counter() - t0)
measured_dt = float(np.median(warmup_times))
print(f"Measured inference dt: {measured_dt*1000:.1f}ms  ({1/measured_dt:.1f} fps)")

kf = make_kalman(measured_dt)
kalman_initialized = False
trail = deque(maxlen=30)
speed_history = deque(maxlen=SPEED_SMOOTH_N)

frame_times = deque(maxlen=30)
last_t = time.perf_counter()

try:
    while True:
        t0 = time.perf_counter()
        dt = t0 - last_t
        last_t = t0

        # Keep transition matrix in sync with actual elapsed time
        kf.transitionMatrix[0, 2] = dt
        kf.transitionMatrix[1, 3] = dt

        frame = stream.read()
        small = cv2.resize(frame, (infer_w, infer_h))
        results = model(small, classes=[32], conf=0.25, verbose=False, imgsz=INFER_SIZE)

        # Pick the detection closest to the Kalman prediction (spatial continuity).
        # Before first detection, fall back to highest confidence.
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
                # No prior — take highest conf (boxes sorted desc by conf)
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
            # vx/vy are now pixels/second (because dt is in seconds)
            vx, vy = float(predicted[2, 0]), float(predicted[3, 0])

            trail.append((sx, sy))

            # pixels/s -> m/s -> km/h, then smooth over recent frames
            speed_ms = np.sqrt(vx**2 + vy**2) / PIXELS_PER_METER
            speed_history.append(speed_ms)
            smoothed_ms = float(np.mean(speed_history))

            for i in range(1, len(trail)):
                alpha = i / len(trail)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, trail[i-1], trail[i], color, 2)

            ARROW_SCALE = 0.3  # seconds of travel shown as arrow
            end_x = int(sx + vx * ARROW_SCALE)
            end_y = int(sy + vy * ARROW_SCALE)
            cv2.arrowedLine(frame, (sx, sy), (end_x, end_y),
                            (0, 255, 255), 2, tipLength=0.3)

            dot_color = (0, 255, 0) if detected else (0, 100, 255)
            cv2.circle(frame, (sx, sy), 8, dot_color, 2)

            label_color = (0, 255, 255) if smoothed_ms > 0.5 else (0, 140, 140)
            cv2.putText(frame, f"{smoothed_ms:.1f} m/s",
                        (sx + 12, sy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

        frame_times.append(time.perf_counter() - t0)
        display_fps = 1.0 / (sum(frame_times) / len(frame_times))
        cv2.putText(frame, f"{display_fps:.1f} fps  {PIXELS_PER_METER:.0f} px/m", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Ball Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    stream.release()
    cv2.destroyAllWindows()
