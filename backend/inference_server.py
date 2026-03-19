"""
inference_server.py  —  AI Sentinel Theft Detection Server
==========================================================
SPEED FIXES:
  1. Skip frames properly      — process every Nth frame, not ratio math
  2. Resize frame before YOLO  — smaller = faster detection
  3. Half precision (FP16)     — 2x faster on GPU
  4. JPEG quality 70 not 82    — smaller frame = faster streaming
  5. Non-blocking frame queue  — no waiting on encode
  6. Threaded YOLO inference   — separate thread for detection
  7. Torch no_grad always on   — never computes gradients

Run:
    pip install flask ultralytics torch opencv-python numpy
    python inference_server.py
Open:  http://localhost:5000
"""

import os, time, queue, threading, json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, Response, send_from_directory
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────
#  ▶  EDIT: paths to your models
# ──────────────────────────────────────────────────────────────────
MODEL_PATH = r"best_theft_lstm_model_v6.pth"
YOLO_MODEL = "yolov8m-pose.pt"
# ──────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16    = (DEVICE == "cuda")           # FP16 only on GPU — 2x speed boost

TARGET_FPS  = 12                           # process 12 frames/sec (was 24 — halved work)
CONF_TH     = 0.3
IOU_TH      = 0.5
SPATIAL_TH  = 0.25
MAX_REMEMBER= 100

# ── YOLO INPUT SIZE — smaller = faster ───────────────────────────
YOLO_IMGSZ  = 416                          # was default 640 — 40% faster

# ── LSTM SETTINGS ────────────────────────────────────────────────
WINDOW_SIZE = 30
MIN_SEQ_LEN = 25
INPUT_SIZE  = 51
HIDDEN_SIZE = 48
NUM_LAYERS  = 1
DROPOUT     = 0.6

# ── STABILITY SETTINGS ───────────────────────────────────────────
MAX_SEQ_BUFFER = 90
SMOOTH_WINDOW  = 8
THEFT_VOTE_MIN = 5
ALERT_COOLDOWN = 3.0

# ── COLORS BGR ───────────────────────────────────────────────────
RED    = (0,   0, 220)
GREEN  = (0, 200,  60)
YELLOW = (20, 180, 220)


# ══════════════════════════════════════════════════════════════════
#  LSTM MODEL
# ══════════════════════════════════════════════════════════════════
class TheftDetectionLSTM(nn.Module):
    def __init__(self, hidden=HIDDEN_SIZE, layers=NUM_LAYERS):
        super().__init__()
        self.lstm    = nn.LSTM(INPUT_SIZE, hidden, layers, batch_first=True, dropout=0.0)
        self.bn      = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        return self.fc(self.dropout(self.bn(out)))


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def normalize_kps(kps, w, h):
    k = kps.copy().astype(np.float32)
    k[:, 0] = np.clip(k[:, 0] / w, 0.0, 1.0)
    k[:, 1] = np.clip(k[:, 1] / h, 0.0, 1.0)
    return k.flatten()


def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    if inter == 0: return 0.0
    return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-6)


def get_latest_window(kps_deque):
    frames = list(kps_deque)[-WINDOW_SIZE:]
    arr = np.array(frames, dtype=np.float32)
    if len(arr) < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - len(arr), INPUT_SIZE), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr


# ══════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════
print(f"[INIT] Device: {DEVICE}  |  FP16: {USE_FP16}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n  Model not found: {MODEL_PATH}\n"
        f"  Place best_theft_lstm_model_v6.pth in the same folder.\n"
    )

print(f"[INIT] Loading LSTM ...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
_hs  = ckpt.get("hidden_size", HIDDEN_SIZE)
_nl  = ckpt.get("num_layers",  NUM_LAYERS)
WINDOW_SIZE = ckpt.get("window_size", WINDOW_SIZE)

lstm_net = TheftDetectionLSTM(_hs, _nl)
lstm_net.load_state_dict(ckpt["model_state_dict"])
lstm_net.eval().to(DEVICE)

# FP16 on GPU = 2x faster inference
if USE_FP16:
    lstm_net = lstm_net.half()

# Wrap in torch.no_grad() context permanently
torch.set_grad_enabled(False)

THRESHOLD = ckpt.get("threshold", 0.55)
print(f"[INIT] LSTM ready  epoch={ckpt.get('epoch','?')}  "
      f"val_f1={ckpt.get('val_f1',0):.4f}  threshold={THRESHOLD}")

print(f"[INIT] Loading YOLO ...")
yolo_model = YOLO(YOLO_MODEL)

# FP16 on GPU for YOLO too
if USE_FP16:
    yolo_model.model.half()

print(f"[INIT] All ready — YOLO imgsz={YOLO_IMGSZ}  TARGET_FPS={TARGET_FPS}\n")


# ══════════════════════════════════════════════════════════════════
#  TRACKER  with smoothing
# ══════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.seqs         = {}
        self.history      = {}
        self.nid          = 0
        self.prob_history = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
        self.last_alert   = {}

    def resolve(self, box, fi):
        best, bs = None, 0.0
        for pid, h in self.history.items():
            diff = fi - h["last"]
            if diff > MAX_REMEMBER: continue
            thr  = 0.15 if diff > 30 else 0.20 if diff > 10 else SPATIAL_TH
            ious = [calc_iou(box.tolist(), b) for b in h["boxes"][-10:]]
            avg  = float(np.mean(ious)) if ious else 0.0
            if avg > thr and avg > bs: bs = avg; best = pid
        if best is not None: return best
        nid = self.nid
        self.history[nid] = {"boxes": [], "last": fi}
        self.seqs[nid]    = deque(maxlen=MAX_SEQ_BUFFER)
        self.nid += 1
        return nid

    def update(self, pid, box, fi, kp_flat):
        if pid not in self.history:
            self.history[pid] = {"boxes": [], "last": fi}
            self.seqs[pid]    = deque(maxlen=MAX_SEQ_BUFFER)
        h = self.history[pid]
        h["boxes"].append(box.tolist())
        h["last"] = fi
        if len(h["boxes"]) > 10:
            h["boxes"].pop(0)
        self.seqs[pid].append(kp_flat)

    def predict_smoothed(self, pid):
        kps  = self.seqs[pid]
        blen = len(kps)
        if blen < MIN_SEQ_LEN:
            return None, 0.0, blen

        window = get_latest_window(kps)
        X = torch.tensor(window[np.newaxis], dtype=torch.float32).to(DEVICE)

        # FP16 on GPU
        if USE_FP16:
            X = X.half()

        prob = torch.sigmoid(lstm_net(X)).squeeze().item()

        self.prob_history[pid].append(prob)
        probs       = list(self.prob_history[pid])
        theft_votes = sum(1 for p in probs if p >= THRESHOLD)
        total       = len(probs)

        is_theft = (total >= SMOOTH_WINDOW and theft_votes >= THEFT_VOTE_MIN)
        disp_prob = float(np.mean(probs))
        return is_theft, disp_prob, blen

    def can_alert(self, pid):
        now = time.time()
        if now - self.last_alert.get(pid, 0) >= ALERT_COOLDOWN:
            self.last_alert[pid] = now
            return True
        return False


# ══════════════════════════════════════════════════════════════════
#  DRAW BOX
# ══════════════════════════════════════════════════════════════════
def draw_box(frame, x1, y1, x2, y2, color, label, is_theft=False):
    th = 3 if is_theft else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
    c = 15
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px+dx*c, py), color, th+1)
        cv2.line(frame, (px, py), (px, py+dy*c), color, th+1)
    font = cv2.FONT_HERSHEY_SIMPLEX; sc = 0.52; lt = 1
    (tw, tth), _ = cv2.getTextSize(label, font, sc, lt)
    pad = 5; ly = max(y1, tth + 2*pad)
    cv2.rectangle(frame, (x1, ly-tth-2*pad), (x1+tw+2*pad, ly), color, -1)
    cv2.putText(frame, label, (x1+pad, ly-pad), font, sc, (0,0,0), lt, cv2.LINE_AA)
    if is_theft:
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, 0.10, frame, 0.90, 0, frame)


# ══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════════════════
state = {
    "running": False, "status": "idle",
    "fps": 0.0, "alerts": 0,
    "frame_buf": 0, "prob": 0.0,
    "is_theft": False, "threshold": THRESHOLD,
    "filename": "",
    "log": deque(maxlen=60),
    "frame_queue":  queue.Queue(maxsize=2),   # smaller = less lag
    "event_queue":  queue.Queue(maxsize=30),
}


def log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    state["log"].appendleft(line)
    try: state["event_queue"].put_nowait({"type": "log", "msg": line})
    except queue.Full: pass
    print(line)


def push_stats():
    try:
        state["event_queue"].put_nowait({
            "type":    "stats",
            "prob":    round(state["prob"] * 100, 1),
            "fps":     round(state["fps"], 1),
            "alerts":  state["alerts"],
            "buf":     state["frame_buf"],
            "buf_max": WINDOW_SIZE,
            "is_theft":state["is_theft"],
            "status":  state["status"],
        })
    except queue.Full: pass


# ══════════════════════════════════════════════════════════════════
#  MAIN PROCESSING THREAD
# ══════════════════════════════════════════════════════════════════
def process_video_thread(video_path):
    state.update({
        "running": True, "status": "loading", "alerts": 0,
        "prob": 0.0, "is_theft": False, "frame_buf": 0
    })
    log(f"Loading: {Path(video_path).name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        state["status"] = "error"
        log("ERROR: Cannot open video")
        state["running"] = False
        return

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # ── FRAME SKIP: process every Nth frame ──────────────────────
    # Much simpler and faster than ratio math
    skip = max(1, round(src_fps / TARGET_FPS))
    log(f"Video: {fw}x{fh} @ {src_fps:.0f}fps  |  {total} frames  |  processing every {skip} frames")
    log(f"Threshold: {THRESHOLD}  |  Votes needed: {THEFT_VOTE_MIN}/{SMOOTH_WINDOW}")
    state["status"] = "running"

    tracker   = Tracker()
    frame_idx = 0
    proc_idx  = 0
    fps_times = deque(maxlen=20)

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            break

        # ── SKIP FRAMES — only process every Nth frame ───────────
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        t0 = time.time()

        # ── YOLO with smaller input size = faster ─────────────────
        r = yolo_model(
            frame,
            conf=CONF_TH,
            iou=IOU_TH,
            imgsz=YOLO_IMGSZ,          # 416 instead of 640
            verbose=False,
            half=USE_FP16              # FP16 on GPU
        )[0]

        best_prob    = 0.0
        any_theft    = False
        person_count = 0

        if r.boxes is not None and r.keypoints is not None and len(r.boxes) > 0:
            boxes    = r.boxes.xyxy.cpu().numpy()
            kpts_all = r.keypoints.data.cpu().numpy()

            for box, kpts in zip(boxes, kpts_all):
                pid     = tracker.resolve(box, proc_idx)
                kp_flat = normalize_kps(kpts, fw, fh)
                tracker.update(pid, box, proc_idx, kp_flat)
                person_count += 1

                is_t, prob_val, buf_len = tracker.predict_smoothed(pid)
                state["frame_buf"] = buf_len
                x1, y1, x2, y2 = map(int, box)

                if is_t is None:
                    color = YELLOW; label = f"P{pid+1} {buf_len}/{MIN_SEQ_LEN}"; is_t = False
                elif is_t:
                    color = RED; label = f"THEFT  P{pid+1}  {prob_val:.0%}"
                    if tracker.can_alert(pid):
                        state["alerts"] += 1
                        log(f"THEFT: Person {pid+1} | confidence={prob_val:.2f}")
                else:
                    color = GREEN; label = f"NORMAL  P{pid+1}  {prob_val:.0%}"

                if prob_val > best_prob: best_prob = prob_val
                if is_t: any_theft = True
                draw_box(frame, x1, y1, x2, y2, color, label, is_t)

        state["prob"]     = best_prob
        state["is_theft"] = any_theft

        # ── FPS calculation ────────────────────────────────────────
        elapsed = time.time() - t0
        fps_times.append(elapsed)
        state["fps"] = 1.0 / (sum(fps_times) / len(fps_times) + 1e-6)

        # ── HUD overlay ────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (fw, 30), (0, 0, 0), -1)
        cv2.putText(frame,
            f"FPS:{state['fps']:.0f}  Persons:{person_count}  "
            f"Prob:{best_prob:.0%}  Alerts:{state['alerts']}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 210, 255), 1, cv2.LINE_AA)
        if any_theft:
            cv2.putText(frame, "!! THEFT DETECTED",
                (fw-230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 220), 2, cv2.LINE_AA)

        # ── Encode JPEG — quality 70 for faster streaming ─────────
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            data = buf.tobytes()
            # Non-blocking: drop old frame if queue full
            if state["frame_queue"].full():
                try: state["frame_queue"].get_nowait()
                except: pass
            try: state["frame_queue"].put_nowait(data)
            except: pass

        push_stats()
        proc_idx  += 1
        frame_idx += 1

    cap.release()
    state["status"]  = "done"
    state["running"] = False
    log(f"Done — {state['alerts']} theft alerts")
    push_stats()


# ══════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=".")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Stop existing processing
    state["running"] = False
    time.sleep(0.3)

    # Clear queues
    for q in (state["frame_queue"], state["event_queue"]):
        while not q.empty():
            try: q.get_nowait()
            except: break

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))
    state["filename"] = f.filename

    threading.Thread(
        target=process_video_thread,
        args=(str(save_path),),
        daemon=True
    ).start()
    return jsonify({"ok": True, "filename": f.filename})


@app.route("/stream")
def stream():
    def gen():
        while True:
            try:
                fr = state["frame_queue"].get(timeout=3.0)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fr + b"\r\n"
            except queue.Empty:
                if state["status"] in ("done", "error", "idle"):
                    break
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def events():
    def gen():
        yield "data: {\"type\":\"connected\"}\n\n"
        while True:
            try:
                ev = state["event_queue"].get(timeout=2.0)
                yield f"data: {json.dumps(ev)}\n\n"
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/status")
def get_status():
    return jsonify({
        "status":    state["status"],
        "fps":       round(state["fps"], 1),
        "prob":      round(state["prob"] * 100, 1),
        "alerts":    state["alerts"],
        "threshold": THRESHOLD,
        "is_theft":  state["is_theft"],
        "buf":       state["frame_buf"],
        "buf_max":   WINDOW_SIZE,
        "log":       list(state["log"])[:20],
    })


@app.route("/stop", methods=["POST"])
def stop():
    state["running"] = False
    log("Stopped by user")
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  AI SENTINEL — Theft Detection Server")
    print(f"  Device    : {DEVICE}  |  FP16: {USE_FP16}")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Voting    : need {THEFT_VOTE_MIN}/{SMOOTH_WINDOW} to confirm theft")
    print(f"  YOLO size : {YOLO_IMGSZ}px  (faster than default 640)")
    print(f"  Skip      : every {max(1, round(25/TARGET_FPS))} frames at 25fps src")
    print("  Open      : http://localhost:5000")
    print("="*52+"\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
