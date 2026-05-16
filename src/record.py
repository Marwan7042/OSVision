"""
TIGHTLY-COUPLED VIO — OAK-D S2 — KEYFRAME GATED & HW OPTIMIZED & HUD ENABLED
=============================================================================
"""

import depthai as dai
import cv2
import numpy as np
import os, json, time, threading, queue
from collections import deque
from datetime import timedelta
import sys, select, termios, tty
import math
import ctypes
import asyncio
from aiohttp import web

# IMPORT OUR CUSTOM MODULES
from load_config import CFG
from utils import DepthDoubleBuffer, HAS_NUMBA, fast_underwater_restore
from ekf import VIO_EKF

# ============================================================
# CONFIG (DERIVED FROM JSON)
# ============================================================
SAVE_DIR   = CFG["paths"]["scan_dir"]
RGB_DIR    = os.path.join(SAVE_DIR, "rgb")
DEPTH_DIR  = os.path.join(SAVE_DIR, "depth")
POSES_FILE = os.path.join(SAVE_DIR, "poses.json")
LAYOUT_FILE = os.path.join(SAVE_DIR, "hud_layout.json")
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

TARGET_FPS = CFG["hardware"]["target_fps"]
IMU_RATE   = CFG["hardware"]["imu_rate_hz"]
USE_IR_PROJECTOR   = CFG["hardware"]["use_ir_projector"]
IR_DOT_BRIGHTNESS  = CFG["hardware"]["ir_dot_brightness_ma"]
LOCK_EXPOSURE      = CFG["hardware"]["lock_exposure"]
EXPOSURE_TIME_US   = CFG["hardware"]["exposure_time_us"]
ISO_SENSITIVITY    = CFG["hardware"]["iso_sensitivity"]
CALIBRATION_MODE   = CFG.get("hardware", {}).get("calibration_mode", "custom").lower()

GATE_MIN_FRAME_GAP   = CFG["keyframe_gating"]["min_frame_gap"]
GATE_MAX_FRAME_GAP   = CFG["keyframe_gating"]["max_frame_gap"]
GATE_MIN_DEPTH_VALID = CFG["keyframe_gating"]["min_depth_valid_ratio"]
GATE_MAX_BLUR_PIXELS = CFG["keyframe_gating"]["max_blur_pixels"]
LAPLACIAN_PASS_THRESHOLD = 50.0

QC_CFG = CFG.get("quality_control", {})
TARGET_SCORE_GOOD = QC_CFG.get("score_good_threshold", 0.75)
TARGET_DEPTH_PCT = QC_CFG.get("ideal_depth_ratio", 0.40) * 100.0
INFO_CFG = CFG.get("information_gating", {})
INFO_SCORE_MIN = INFO_CFG.get("min_information_score", 0.45)

STATIC_VAR_THR   = CFG["ekf_tuning"]["static_variance_threshold"]
MIN_GRAV_SAMPLES = CFG["ekf_tuning"]["min_gravity_samples"]
MIN_FEAT_UPDATE  = CFG["ekf_tuning"]["min_feature_update"]
DEPTH_MIN_MM     = CFG["ekf_tuning"]["depth_min_mm"]
DEPTH_MAX_MM     = CFG["ekf_tuning"]["depth_max_mm"]

BUDGET_CFG = CFG.get("runtime_budgets", {})
VIS_P95_BUDGET_MS = BUDGET_CFG.get("visual_p95_ms", 22.0)
DISK_P95_BUDGET_MS = BUDGET_CFG.get("disk_p95_ms", 14.0)
LOOP_P95_BUDGET_MS = BUDGET_CFG.get("main_loop_p95_ms", 35.0)
BUDGET_REPORT_SEC = BUDGET_CFG.get("report_interval_sec", 8.0)
QUEUE_PRESSURE_RAISE = BUDGET_CFG.get("queue_pressure_raise", 0.70)
QUEUE_PRESSURE_RECOVER = BUDGET_CFG.get("queue_pressure_recover", 0.35)

SYNC_CFG = CFG.get("time_sync", {})
SYNC_WARN_ABS_S = SYNC_CFG.get("warn_abs_offset_s", 0.030)
SYNC_WARN_DRIFT_SPS = SYNC_CFG.get("warn_drift_s_per_s", 0.004)
SYNC_EWMA_ALPHA = SYNC_CFG.get("offset_ewma_alpha", 0.08)

RECOVERY_CFG = CFG.get("recovery_modes", {})
VISION_WEAK_MIN_GAP = RECOVERY_CFG.get("vision_weak_min_frame_gap", 4)
IMU_ONLY_MIN_GAP = RECOVERY_CFG.get("imu_only_min_frame_gap", 8)
IMU_ONLY_HOLD_SEC = RECOVERY_CFG.get("imu_only_hold_sec", 6.0)
DISK_PRESSURE_MIN_GAP = RECOVERY_CFG.get("disk_pressure_min_frame_gap", 10)
DISK_PRESSURE_SAVE_STRIDE = max(1, int(RECOVERY_CFG.get("disk_pressure_save_stride", 2)))
DISK_PRESSURE_QFILL = RECOVERY_CFG.get("disk_pressure_queue_fill", 0.7)
VISION_WEAK_INFO_BONUS = RECOVERY_CFG.get("vision_weak_info_bonus", 0.08)

CR_CFG = CFG.get("color_restore", {})
CR_ENABLED = CR_CFG.get("enabled", False)
CR_R_MAX = CR_CFG.get("r_max_gain", 3.0)
CR_G_MAX = CR_CFG.get("g_max_gain", 1.2)
CR_HUD = CR_CFG.get("apply_to_hud", True)
CR_REC = CR_CFG.get("apply_to_recording", True)

ISP_WIDTH  = 960
ISP_HEIGHT = 540

# ============================================================
# SHARED STATE & BUFFERS
# ============================================================
latest_jpeg  = None
recording_event = threading.Event()
visual_queue = queue.Queue(maxsize=1)
stop_event   = threading.Event()

lk_queue = queue.Queue(maxsize=1)
disk_health = {"consecutive_drops": 0}

cam_ctrl_lock = threading.Lock()
cam_state = {
    "wb": 4600,
    "exp": EXPOSURE_TIME_US,
    "iso": ISO_SENSITIVITY
}

hud_telemetry = {
    "state": "IDLE",
    "score": 1.0,
    "blur": 0.0,
    "depth_pct": 0.0,
    "message": "AWAITING GRAVITY CALIBRATION",
    "mode": "BOOT",
    "visual_nis": 0.0,
    "sync_offset_ms": 0.0,
    "sync_drift_ms_s": 0.0,
    "adaptive_gap": GATE_MIN_FRAME_GAP
}
hud_lock = threading.Lock()
runtime_state = {"bad_streak_counter": 0}
adaptive_gate = {"min_frame_gap": int(GATE_MIN_FRAME_GAP)}
imu_sync = {"last_imu_ts": None, "last_imu_host_ts": None}
sync_state = {"offset_ewma": None, "drift_ewma": 0.0, "last_cam_ts": None, "last_host_ts": None}
recovery_state = {"imu_only_until": 0.0}
lat_lock = threading.Lock()
latency_samples = {
    "visual_ms": deque(maxlen=600),
    "disk_ms": deque(maxlen=800),
    "main_loop_ms": deque(maxlen=600),
}
runtime_mode = {"mode": "BOOT"}

def _observe_latency(key, value_ms):
    with lat_lock:
        latency_samples[key].append(float(value_ms))

def _latency_percentiles(key):
    with lat_lock:
        arr = np.array(latency_samples[key], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }

def _feature_coverage_score(features, width, height):
    if not features:
        return 0.0
    xs = [f.position.x for f in features]
    ys = [f.position.y for f in features]
    if len(xs) < 4:
        return 0.0
    spread = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return float(np.clip(spread / float(width * height), 0.0, 1.0))

def _parallax_score(curr_fd, prev_fd):
    if not curr_fd or not prev_fd:
        return 0.0
    shared = [fid for fid in curr_fd.keys() if fid in prev_fd]
    if len(shared) < 8:
        return 0.0
    disp = []
    for fid in shared:
        p0 = prev_fd[fid]
        p1 = curr_fd[fid]
        disp.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    med = float(np.median(np.array(disp, dtype=np.float64)))
    return float(np.clip(med / 20.0, 0.0, 1.0))

# ============================================================
# ZERO-COPY BUFFERS
# ============================================================
MAX_FEATURES = 150
pp_buffer = np.empty((MAX_FEATURES, 2), dtype=np.float32)
pc_buffer = np.empty((MAX_FEATURES, 2), dtype=np.float32)

# ============================================================
# THREAD AFFINITY
# ============================================================
def pin_thread(core_id):
    """Pin calling thread to a specific core."""
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        pass  # non-Linux systems

# ============================================================
# THREADS
# ============================================================
def visual_worker(ekf, K_mat, T_ic, T_ci):
    ok = 0
    fail = 0
    while not stop_event.is_set():
        try: 
            item = visual_queue.get(timeout=0.1)
        except queue.Empty: 
            continue
            
        if item is None: 
            break

        t0 = time.perf_counter()
        r, n = ekf.update_visual(item[0], item[1], item[2], K_mat, T_ic, T_ci, item[3])
        _observe_latency("visual_ms", (time.perf_counter() - t0) * 1000.0)
        if r: ok += 1
        else: fail += 1

        health = ekf.get_visual_health()
        with hud_lock:
            hud_telemetry["visual_nis"] = float(health.get("nis_ema", 0.0))
            hud_telemetry["mode"] = health.get("mode", runtime_mode["mode"])
        visual_queue.task_done()

def imu_worker(device, ekf):
    pin_thread(3)  # Core 3 dedicated to IMU/EKF
    q = device.getOutputQueue("imu", maxSize=100, blocking=False)
    ab = np.zeros(3, dtype=np.float64)
    gb = np.zeros(3, dtype=np.float64)
    
    while not stop_event.is_set():
        try:
            m = q.tryGet() 
            if m is not None:
                for pkt in m.packets:
                    a = pkt.acceleroMeter
                    g = pkt.gyroscope
                    try:
                        ts = a.getTimestampDevice().total_seconds()
                    except AttributeError:
                        try: ts = a.timestamp.get().total_seconds()
                        except AttributeError: ts = time.monotonic()
                            
                    ab[0], ab[1], ab[2] = a.x, a.y, a.z
                    gb[0], gb[1], gb[2] = g.x, g.y, g.z
                    ekf.feed_imu(ab, gb, ts)
                    imu_sync["last_imu_ts"] = float(ts)
                    imu_sync["last_imu_host_ts"] = time.monotonic()
            else:
                time.sleep(0.001)
        except Exception as e:
            if stop_event.is_set(): break
            time.sleep(0.01)

def lk_worker():
    # Gutted LK Fallback Thread
    while not stop_event.is_set():
        try:
            lk_queue.get(timeout=0.1)
        except queue.Empty:
            continue

# ============================================================
# ASYNC WEB SERVER (aiohttp)
# ============================================================
routes = web.RouteTableDef()

@routes.get('/')
async def index(request):
    html = f'''<html><head><style>
        :root {{
            --panel: rgba(20, 24, 30, 0.68);
            --panel-strong: rgba(16, 20, 26, 0.88);
            --panel-border: rgba(136, 164, 200, 0.24);
            --bg: #080b12;
            --text: #e6edf7;
            --muted: #92a6c3;
            --accent: #3b82f6;
            --accent-2: #22d3ee;
        }}
        body {{
            background:
                radial-gradient(1200px 700px at 10% -15%, rgba(59, 130, 246, 0.22), transparent 55%),
                radial-gradient(900px 600px at 100% 0%, rgba(34, 211, 238, 0.17), transparent 50%),
                var(--bg);
            color: var(--text);
            font-family: Inter, Segoe UI, Roboto, sans-serif;
            margin: 0;
            padding: 14px;
            text-align: center;
            overflow: hidden;
        }}
        h2 {{
            font-size: 0.86em;
            margin: 6px 0 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 700;
        }}
        .panel-title {{
            font-size: 0.74em;
            margin: 0 0 10px;
            color: #b6c5dc;
            text-transform: uppercase;
            letter-spacing: 1.6px;
            font-weight: 700;
            text-align: left;
        }}
        .app {{ max-width: 1440px; margin: 0 auto; position: relative; }}
        .control-bar {{
            background: var(--panel-strong);
            padding: 12px 14px;
            border-radius: 14px;
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-bottom: 12px;
            border: 1px solid var(--panel-border);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.32);
        }}
        .control-item {{ display: flex; align-items: center; gap: 8px; font-size: 13px; }}
        input[type=range] {{ width: 140px; cursor: pointer; accent-color: var(--accent); }}
        .flex-row {{ display: flex; justify-content: center; align-items: flex-start; width: 100%; }}
        .video-container {{
            flex: 1;
            max-width: 100%;
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 10px;
            overflow: auto;
            min-width: 760px;
            min-height: 440px;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32);
        }}
        .video-feed {{ width: 100%; aspect-ratio: 16 / 9; border: 1px solid rgba(166, 190, 223, 0.28); border-radius: 10px; background: #000; }}
        .frame-panel {{
            width: 100%;
            aspect-ratio: 16 / 9;
            min-width: 320px;
            min-height: 180px;
            border: 1px solid rgba(166, 190, 223, 0.28);
            border-radius: 10px;
            overflow: hidden;
            background: #000;
        }}
        .video-canvas {{ width: 100%; height: 100%; border: 0; border-radius: 10px; background: #000; display: block; }}
        #hud-source {{ display: none; }}
        .hud-layout {{ display: flex; gap: 10px; align-items: stretch; }}
        .hud-main {{ flex: 1; min-width: 0; overflow: auto; min-width: 520px; min-height: 300px; }}
        .side-stack {{ width: 320px; min-width: 260px; display: flex; flex-direction: column; gap: 10px; }}
        .side-controls {{ margin-bottom: 0; display: block; }}
        .side-controls .control-item {{ justify-content: space-between; margin-bottom: 8px; }}
        .side-controls .control-item:last-child {{ margin-bottom: 0; }}
        .side-controls input[type=range] {{ width: 145px; }}
        .telemetry-panel {{
            width: 300px;
            padding: 10px 12px;
            background: linear-gradient(180deg, rgba(18, 24, 34, 0.9), rgba(13, 18, 26, 0.95));
            border: 1px solid rgba(127, 154, 190, 0.28);
            border-radius: 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 14px;
            text-align: left;
            color: #d6e1f0;
            line-height: 1.7;
            white-space: pre-line;
            overflow: auto;
            min-width: 230px;
            min-height: 160px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }}
        .interactive-panel {{ position: relative; touch-action: none; user-select: none; }}
        .interactive-panel .drag-handle {{ position: absolute; top: 0; left: 0; right: 0; height: 16px; cursor: grab; opacity: 0; }}
        .interactive-panel.active .drag-handle {{ cursor: grabbing; }}
        .interactive-panel.active {{ z-index: 40; }}
        @media (max-width: 1100px) {{
            .hud-layout {{ flex-direction: column; }}
            .side-stack {{ width: auto; }}
            .telemetry-panel {{ width: auto; }}
        }}
        .toggle-btn {{
            background: linear-gradient(180deg, rgba(59, 130, 246, 0.95), rgba(37, 99, 235, 0.95));
            color: #fff;
            border: 1px solid rgba(99, 179, 255, 0.7);
            border-radius: 9px;
            padding: 6px 11px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(25, 74, 150, 0.32);
            transition: transform 120ms ease, filter 120ms ease;
        }}
        .toggle-btn:hover {{ filter: brightness(1.07); transform: translateY(-1px); }}
        .toggle-btn.off {{
            background: linear-gradient(180deg, rgba(71, 85, 105, 0.9), rgba(51, 65, 85, 0.9));
            border-color: rgba(148, 163, 184, 0.55);
            box-shadow: none;
        }}
        #wb-lbl {{ color: #00eeff; }} #exp-lbl {{ color: #ffaa00; }} #iso-lbl {{ color: #ff4444; }}
    </style></head>
    <body>
        <div class="app">
        <div class="flex-row">
            <div id="video-panel" class="video-container interactive-panel" data-min-w="760" data-min-h="440">
                <div class="drag-handle"></div>
                <h2>Pilot HUD (Client Rendered)</h2>
                <div class="hud-layout">
                    <div id="hud-main-panel" class="hud-main interactive-panel" data-min-w="520" data-min-h="300">
                        <div class="drag-handle"></div>
                        <div id="frame-panel" class="frame-panel interactive-panel" data-min-w="320" data-min-h="180">
                            <div class="drag-handle"></div>
                            <canvas id="hud-canvas" class="video-canvas" width="640" height="360"></canvas>
                        </div>
                        <img id="hud-source" src="/stream">
                    </div>
                    <div class="side-stack">
                        <div id="control-panel" class="control-bar side-controls interactive-panel" data-min-w="250" data-min-h="190">
                            <div class="drag-handle"></div>
                            <div class="panel-title">Camera Controls</div>
                            <div class="control-item">
                                <label id="wb-lbl">WB: <span id="wbVal">4600</span>K</label>
                                <input type="range" min="2500" max="8000" step="100" value="4600" oninput="document.getElementById('wbVal').innerText = this.value; fetch('/set_wb?v=' + this.value);">
                            </div>
                            <div class="control-item">
                                <label id="exp-lbl">EXP: <span id="expVal">{EXPOSURE_TIME_US}</span>us</label>
                                <input type="range" min="1000" max="33000" step="500" value="{EXPOSURE_TIME_US}" oninput="document.getElementById('expVal').innerText = this.value; fetch('/set_exp?v=' + this.value);">
                            </div>
                            <div class="control-item">
                                <label id="iso-lbl">ISO: <span id="isoVal">{ISO_SENSITIVITY}</span></label>
                                <input type="range" min="100" max="1600" step="100" value="{ISO_SENSITIVITY}" oninput="document.getElementById('isoVal').innerText = this.value; fetch('/set_iso?v=' + this.value);">
                            </div>
                            <div class="control-item">
                                <button id="annotations-toggle" class="toggle-btn" type="button">Annotations: ON</button>
                            </div>
                            <div class="control-item">
                                <button id="reset-layout-btn" class="toggle-btn" type="button">Reset Layout</button>
                            </div>
                        </div>
                        <div class="telemetry-panel interactive-panel" id="telemetry-panel" data-min-w="230" data-min-h="160">
                            <div class="drag-handle"></div>
                            <div class="panel-title">Mission Telemetry</div>
                            STATE: IDLE (target: GOOD)
SCORE: 1.00 (target: above {TARGET_SCORE_GOOD:.2f})
BLUR(LAPLV): 0.0 (target: above {LAPLACIAN_PASS_THRESHOLD:.1f})
DEPTH: 0.0% (target: above {TARGET_DEPTH_PCT:.1f}%)
REC: NO (target: YES while recording)
                        </div>
                    </div>
                </div>
            </div>
        </div>
        </div>
        <script>
            const hudCanvas = document.getElementById('hud-canvas');
            const hudCtx = hudCanvas.getContext('2d');
            const hudSrc = document.getElementById('hud-source');
            const telemetryPanel = document.getElementById('telemetry-panel');
            const annotationsToggle = document.getElementById('annotations-toggle');
            const resetLayoutBtn = document.getElementById('reset-layout-btn');
            const LAYOUT_STORAGE_KEY = "slam_hud_layout_v1";
            const SERVER_LAYOUT_ENDPOINT = "/layout";
            let zCounter = 20;
            let annotationsEnabled = true;
            const defaultPanelState = new Map();
            let telem = {{
                state: "IDLE",
                score: 1.0,
                blur: 0.0,
                depth_pct: 0.0,
                message: "AWAITING GRAVITY CALIBRATION",
                recording: false
            }};

            function syncAnnotationsToggleUI() {{
                annotationsToggle.textContent = "Annotations: " + (annotationsEnabled ? "ON" : "OFF");
                annotationsToggle.className = annotationsEnabled ? "toggle-btn" : "toggle-btn off";
            }}

            function resizeCursor(dir) {{
                const map = {{
                    "n": "ns-resize", "s": "ns-resize", "e": "ew-resize", "w": "ew-resize",
                    "ne": "nesw-resize", "sw": "nesw-resize", "nw": "nwse-resize", "se": "nwse-resize"
                }};
                return map[dir] || "default";
            }}

            function getResizeDir(e, el) {{
                const r = el.getBoundingClientRect();
                const pad = 8;
                const nearLeft = (e.clientX - r.left) <= pad;
                const nearRight = (r.right - e.clientX) <= pad;
                const nearTop = (e.clientY - r.top) <= pad;
                const nearBottom = (r.bottom - e.clientY) <= pad;
                let dir = "";
                if (nearTop) dir += "n";
                else if (nearBottom) dir += "s";
                if (nearLeft) dir += "w";
                else if (nearRight) dir += "e";
                return dir;
            }}

            function isInteractiveTarget(target) {{
                return !!target.closest("input, button, label, canvas");
            }}

            function getTranslate(el) {{
                const m = /translate\\(([-\\d.]+)px,\\s*([-\\d.]+)px\\)/.exec(el.style.transform || "");
                return {{
                    x: m ? Number(m[1]) : 0,
                    y: m ? Number(m[2]) : 0
                }};
            }}

            function setTranslate(el, x, y) {{
                el.style.transform = "translate(" + Math.round(x) + "px, " + Math.round(y) + "px)";
            }}

            function rectInParent(el, parentRect) {{
                const r = el.getBoundingClientRect();
                return {{
                    left: r.left - parentRect.left,
                    top: r.top - parentRect.top,
                    right: r.right - parentRect.left,
                    bottom: r.bottom - parentRect.top,
                    width: r.width,
                    height: r.height
                }};
            }}

            function overlaps(a, b, gap = 6) {{
                return !(a.right <= b.left + gap || a.left >= b.right - gap || a.bottom <= b.top + gap || a.top >= b.bottom - gap);
            }}

            function clampTranslateToParent(el, tx, ty) {{
                const parent = el.parentElement;
                if (!parent) return {{ x: tx, y: ty }};
                const parentRect = parent.getBoundingClientRect();
                const r = el.getBoundingClientRect();
                const w = r.width;
                const h = r.height;
                const base = getTranslate(el);
                const currentLeft = r.left - parentRect.left;
                const currentTop = r.top - parentRect.top;
                const proposedLeft = currentLeft + (tx - base.x);
                const proposedTop = currentTop + (ty - base.y);
                const pad = 2;
                const maxLeft = Math.max(pad, parentRect.width - w - pad);
                const maxTop = Math.max(pad, parentRect.height - h - pad);
                const clampedLeft = Math.min(maxLeft, Math.max(pad, proposedLeft));
                const clampedTop = Math.min(maxTop, Math.max(pad, proposedTop));
                return {{
                    x: tx + (clampedLeft - proposedLeft),
                    y: ty + (clampedTop - proposedTop)
                }};
            }}

            function movePanelAway(mover, blocker, parentRect) {{
                const mRect = rectInParent(mover, parentRect);
                const bRect = rectInParent(blocker, parentRect);
                if (!overlaps(mRect, bRect)) return false;

                const t = getTranslate(mover);
                const gap = 8;
                const options = [
                    {{ dx: (bRect.right + gap) - mRect.left, dy: 0 }},
                    {{ dx: (bRect.left - gap) - mRect.right, dy: 0 }},
                    {{ dx: 0, dy: (bRect.bottom + gap) - mRect.top }},
                    {{ dx: 0, dy: (bRect.top - gap) - mRect.bottom }}
                ];

                let best = null;
                for (const opt of options) {{
                    const candidate = clampTranslateToParent(mover, t.x + opt.dx, t.y + opt.dy);
                    const cost = Math.abs(candidate.x - t.x) + Math.abs(candidate.y - t.y);
                    if (!best || cost < best.cost) best = {{ ...candidate, cost }};
                }}

                if (!best) return false;
                setTranslate(mover, best.x, best.y);
                return true;
            }}

            function resolveCollisions(parent, anchorEl) {{
                if (!parent) return;
                const panels = Array.from(parent.children).filter((c) => c.classList && c.classList.contains("interactive-panel"));
                if (!panels.includes(anchorEl)) return;
                const ordered = [anchorEl, ...panels.filter((p) => p !== anchorEl)];
                const parentRect = parent.getBoundingClientRect();

                for (let iter = 0; iter < 24; iter++) {{
                    let changed = false;
                    for (let i = 0; i < ordered.length; i++) {{
                        const blocker = ordered[i];
                        for (let j = 0; j < ordered.length; j++) {{
                            if (i === j) continue;
                            const mover = ordered[j];
                            if (movePanelAway(mover, blocker, parentRect)) {{
                                changed = true;
                                const minW = Number(mover.dataset.minW || 160);
                                const minH = Number(mover.dataset.minH || 120);
                                clampPanelToParent(mover, minW, minH);
                            }}
                        }}
                    }}
                    if (!changed) break;
                }}
            }}

            function clampPanelToParent(el, minW, minH) {{
                const parent = el.parentElement;
                if (!parent) return;

                const parentRect = parent.getBoundingClientRect();
                const maxW = Math.max(minW, parentRect.width - 6);
                const maxH = Math.max(minH, parentRect.height - 6);

                const currentW = el.getBoundingClientRect().width;
                const currentH = el.getBoundingClientRect().height;
                const clampedW = Math.min(Math.max(minW, currentW), maxW);
                const clampedH = Math.min(Math.max(minH, currentH), maxH);
                el.style.width = clampedW + "px";
                el.style.height = clampedH + "px";

                const txMatch = /translate\\(([-\\d.]+)px,\\s*([-\\d.]+)px\\)/.exec(el.style.transform || "");
                let tx = txMatch ? Number(txMatch[1]) : 0;
                let ty = txMatch ? Number(txMatch[2]) : 0;

                let rect = el.getBoundingClientRect();
                const pad = 2;
                if (rect.left < parentRect.left + pad) tx += (parentRect.left + pad - rect.left);
                if (rect.right > parentRect.right - pad) tx -= (rect.right - (parentRect.right - pad));
                if (rect.top < parentRect.top + pad) ty += (parentRect.top + pad - rect.top);
                if (rect.bottom > parentRect.bottom - pad) ty -= (rect.bottom - (parentRect.bottom - pad));
                setTranslate(el, tx, ty);
            }}

            function applyLayoutState(layout) {{
                Object.entries(layout).forEach(([id, state]) => {{
                    const el = document.getElementById(id);
                    if (!el) return;
                    if (typeof state.width === "string" && state.width) el.style.width = state.width;
                    if (typeof state.height === "string" && state.height) el.style.height = state.height;
                    if (typeof state.transform === "string" && state.transform) el.style.transform = state.transform;
                    const minW = Number(el.dataset.minW || 160);
                    const minH = Number(el.dataset.minH || 120);
                    clampPanelToParent(el, minW, minH);
                }});
            }}

            function saveLayoutState() {{
                try {{
                    const layout = {{}};
                    document.querySelectorAll(".interactive-panel[id]").forEach((el) => {{
                        layout[el.id] = {{
                            width: el.style.width || "",
                            height: el.style.height || "",
                            transform: el.style.transform || "translate(0px, 0px)"
                        }};
                    }});
                    const payload = JSON.stringify(layout);
                    localStorage.setItem(LAYOUT_STORAGE_KEY, payload);
                    fetch(SERVER_LAYOUT_ENDPOINT, {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: payload,
                        keepalive: true
                    }}).catch(() => {{}});
                    return payload;
                }} catch (_) {{}}
                return null;
            }}

            async function loadLayoutState() {{
                let layout = null;
                try {{
                    const r = await fetch(SERVER_LAYOUT_ENDPOINT, {{ cache: "no-store" }});
                    if (r.ok) {{
                        const remote = await r.json();
                        if (remote && typeof remote === "object" && Object.keys(remote).length > 0) {{
                            layout = remote;
                        }}
                    }}
                }} catch (_) {{}}
                if (!layout) {{
                    try {{
                        const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
                        if (raw) layout = JSON.parse(raw);
                    }} catch (_) {{}}
                }}
                if (layout && typeof layout === "object") applyLayoutState(layout);
            }}

            function makePanelInteractive(el) {{
                const rect = el.getBoundingClientRect();
                el.style.width = rect.width + "px";
                el.style.height = rect.height + "px";
                if (!el.style.transform) el.style.transform = "translate(0px, 0px)";

                let mode = null;
                let dir = "";
                let sx = 0, sy = 0;
                let startW = 0, startH = 0;
                let startTX = 0, startTY = 0;
                let startRect = null;
                let parentRect = null;
                const minW = Number(el.dataset.minW || 160);
                const minH = Number(el.dataset.minH || 120);
                defaultPanelState.set(el, {{
                    width: rect.width,
                    height: rect.height,
                    transform: "translate(0px, 0px)"
                }});

                const onMouseMove = (e) => {{
                    if (mode) {{
                        const dx = e.clientX - sx;
                        const dy = e.clientY - sy;
                        if (mode === "drag") {{
                            let nextTX = startTX + dx;
                            let nextTY = startTY + dy;
                            const minTX = startTX + (parentRect.left + 2 - startRect.left);
                            const maxTX = startTX + (parentRect.right - 2 - startRect.right);
                            const minTY = startTY + (parentRect.top + 2 - startRect.top);
                            const maxTY = startTY + (parentRect.bottom - 2 - startRect.bottom);
                            nextTX = Math.min(maxTX, Math.max(minTX, nextTX));
                            nextTY = Math.min(maxTY, Math.max(minTY, nextTY));
                            setTranslate(el, nextTX, nextTY);
                            return;
                        }}

                        let newW = startW;
                        let newH = startH;
                        let newTX = startTX;
                        let newTY = startTY;

                        if (dir.includes("e")) newW = Math.max(minW, startW + dx);
                        if (dir.includes("s")) newH = Math.max(minH, startH + dy);
                        if (dir.includes("w")) {{
                            const w = Math.max(minW, startW - dx);
                            newTX += (startW - w);
                            newW = w;
                        }}
                        if (dir.includes("n")) {{
                            const h = Math.max(minH, startH - dy);
                            newTY += (startH - h);
                            newH = h;
                        }}

                        el.style.width = newW + "px";
                        el.style.height = newH + "px";
                        setTranslate(el, newTX, newTY);
                        clampPanelToParent(el, minW, minH);
                        return;
                    }}

                    const d = getResizeDir(e, el);
                    if (d) el.style.cursor = resizeCursor(d);
                    else if (e.target.closest(".drag-handle")) el.style.cursor = "grab";
                    else el.style.cursor = "default";
                }};

                const onMouseUp = () => {{
                    if (mode && el.parentElement) {{
                        resolveCollisions(el.parentElement, el);
                    }}
                    if (mode) saveLayoutState();
                    mode = null;
                    el.classList.remove("active");
                }};

                el.addEventListener("mousedown", (e) => {{
                    if (e.button !== 0) return;
                    const d = getResizeDir(e, el);
                    const txMatch = /translate\\(([-\\d.]+)px,\\s*([-\\d.]+)px\\)/.exec(el.style.transform || "");
                    startTX = txMatch ? Number(txMatch[1]) : 0;
                    startTY = txMatch ? Number(txMatch[2]) : 0;
                    sx = e.clientX;
                    sy = e.clientY;
                    startRect = el.getBoundingClientRect();
                    parentRect = el.parentElement.getBoundingClientRect();
                    startW = startRect.width;
                    startH = startRect.height;
                    el.style.zIndex = String(++zCounter);
                    el.classList.add("active");

                    if (d) {{
                        mode = "resize";
                        dir = d;
                        e.stopPropagation();
                        e.preventDefault();
                        return;
                    }}

                    if (isInteractiveTarget(e.target) && !e.target.closest(".drag-handle")) return;
                    mode = "drag";
                    e.stopPropagation();
                    e.preventDefault();
                }});

                window.addEventListener("mousemove", onMouseMove);
                window.addEventListener("mouseup", onMouseUp);
            }}

            document.querySelectorAll(".interactive-panel").forEach(makePanelInteractive);
            loadLayoutState();
            resetLayoutBtn.addEventListener("click", () => {{
                defaultPanelState.forEach((state, el) => {{
                    el.style.width = state.width + "px";
                    el.style.height = state.height + "px";
                    el.style.transform = state.transform;
                    const minW = Number(el.dataset.minW || 160);
                    const minH = Number(el.dataset.minH || 120);
                    clampPanelToParent(el, minW, minH);
                }});
                localStorage.removeItem(LAYOUT_STORAGE_KEY);
                fetch(SERVER_LAYOUT_ENDPOINT, {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: "{{}}",
                    keepalive: true
                }}).catch(() => {{}});
            }});
            window.addEventListener("beforeunload", saveLayoutState);

            annotationsToggle.addEventListener('click', () => {{
                annotationsEnabled = !annotationsEnabled;
                syncAnnotationsToggleUI();
            }});

            async function pollTelemetry() {{
                try {{
                    const r = await fetch('/telemetry', {{ cache: 'no-store' }});
                    if (r.ok) {{
                        telem = await r.json();
                        telemetryPanel.textContent = "STATE: " + telem.state + " (target: GOOD)\\n" +
                            "SCORE: " + Number(telem.score || 0).toFixed(2) + " (target: above {TARGET_SCORE_GOOD:.2f})\\n" +
                            "BLUR(LAPLV): " + Number(telem.blur || 0).toFixed(1) + " (target: above {LAPLACIAN_PASS_THRESHOLD:.1f})\\n" +
                            "DEPTH: " + (Number(telem.depth_pct || 0) * 100).toFixed(1) + "%" + " (target: above {TARGET_DEPTH_PCT:.1f}%)\\n" +
                            "REC: " + (telem.recording ? "YES" : "NO") + " (target: YES while recording)";
                    }}
                }} catch (_) {{}}
                setTimeout(pollTelemetry, 200);
            }}

            function drawHud() {{
                const cssW = Math.max(2, hudCanvas.clientWidth);
                const cssH = Math.max(2, hudCanvas.clientHeight);
                if (hudCanvas.width !== cssW || hudCanvas.height !== cssH) {{
                    hudCanvas.width = cssW;
                    hudCanvas.height = cssH;
                }}

                const w = hudCanvas.width;
                const h = hudCanvas.height;

                try {{
                    hudCtx.drawImage(hudSrc, 0, 0, w, h);
                }} catch (_) {{}}

                if (!annotationsEnabled) {{
                    requestAnimationFrame(drawHud);
                    return;
                }}

                if (telem.recording) {{
                    hudCtx.fillStyle = "rgba(0, 0, 0, 0.42)";
                    hudCtx.fillRect(w - 124, 8, 112, 34);
                    hudCtx.font = "bold 24px sans-serif";
                    hudCtx.fillStyle = "#ff0000";
                    hudCtx.beginPath();
                    hudCtx.arc(w - 100, 27, 8, 0, Math.PI * 2);
                    hudCtx.fill();
                    hudCtx.fillText("REC", w - 80, 35);
                }}

                hudCtx.strokeStyle = "#ffffff";
                hudCtx.lineWidth = 1;
                hudCtx.beginPath();
                hudCtx.moveTo((w / 2) - 10, h / 2);
                hudCtx.lineTo((w / 2) + 10, h / 2);
                hudCtx.moveTo(w / 2, (h / 2) - 10);
                hudCtx.lineTo(w / 2, (h / 2) + 10);
                hudCtx.stroke();

                if (telem.message) {{
                    hudCtx.fillStyle = "rgba(0, 0, 0, 0.55)";
                    hudCtx.fillRect(0, h - 58, w, 58);
                    hudCtx.font = "bold 28px sans-serif";
                    hudCtx.fillStyle = "#ff0000";
                    const textW = hudCtx.measureText(telem.message).width;
                    hudCtx.fillText(telem.message, (w - textW) / 2, h - 30);
                }}

                requestAnimationFrame(drawHud);
            }}

            syncAnnotationsToggleUI();
            pollTelemetry();
            requestAnimationFrame(drawHud);
        </script>
    </body></html>'''
    return web.Response(text=html, content_type='text/html')

@routes.get('/set_wb')
async def set_wb(request):
    try:
        val = int(request.query.get('v', 4600))
        with cam_ctrl_lock: cam_state["wb"] = max(2500, min(8000, val))
    except Exception: pass
    return web.Response(text="OK")

@routes.get('/set_exp')
async def set_exp(request):
    try:
        val = int(request.query.get('v', 15000))
        with cam_ctrl_lock: cam_state["exp"] = max(1000, min(33000, val))
    except Exception: pass
    return web.Response(text="OK")

@routes.get('/set_iso')
async def set_iso(request):
    try:
        val = int(request.query.get('v', 800))
        with cam_ctrl_lock: cam_state["iso"] = max(100, min(1600, val))
    except Exception: pass
    return web.Response(text="OK")

@routes.get('/stream')
async def stream(request):
    response = web.StreamResponse(headers={
        'Cache-Control': 'no-cache,private',
        'Content-Type': 'multipart/x-mixed-replace;boundary=FRAME'
    })
    await response.prepare(request)
    try:
        while not stop_event.is_set():
            fd = latest_jpeg  # Lock-free atomic reference read
            if fd:
                await response.write(b'--FRAME\r\nContent-Type:image/jpeg\r\n')
                await response.write(f'Content-Length:{len(fd)}\r\n\r\n'.encode())
                await response.write(fd)
                await response.write(b'\r\n')
            await asyncio.sleep(0.033)  # Yield to event loop (~30fps limit)
    except Exception:
        pass # Handle client disconnects gracefully without crashing the loop
    return response

@routes.get('/telemetry')
async def telemetry(request):
    with hud_lock:
        payload = hud_telemetry.copy()
    payload["recording"] = recording_event.is_set()
    return web.json_response(payload)

@routes.get('/layout')
async def get_layout(request):
    if not os.path.exists(LAYOUT_FILE):
        return web.json_response({})
    try:
        with open(LAYOUT_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return web.json_response(data)
    except (OSError, json.JSONDecodeError):
        pass
    return web.json_response({})

@routes.post('/layout')
async def set_layout(request):
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    if not isinstance(payload, dict):
        return web.Response(status=400, text="Layout must be an object")

    sanitized = {}
    for panel_id, state in payload.items():
        if not isinstance(panel_id, str) or not isinstance(state, dict):
            continue
        width = state.get("width", "")
        height = state.get("height", "")
        transform = state.get("transform", "translate(0px, 0px)")
        if not isinstance(width, str) or not isinstance(height, str) or not isinstance(transform, str):
            continue
        sanitized[panel_id] = {
            "width": width[:32],
            "height": height[:32],
            "transform": transform[:64]
        }

    try:
        with open(LAYOUT_FILE, "w") as f:
            json.dump(sanitized, f)
    except OSError as e:
        return web.Response(status=500, text=f"Failed to save layout: {e}")

    return web.json_response({"ok": True})

def start_async_server():
    """Runs the asyncio event loop in a dedicated background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    loop.run_until_complete(site.start())
    
    async def watch_stop():
        while not stop_event.is_set():
            await asyncio.sleep(1)
        await runner.cleanup()
        loop.stop()
        
    loop.create_task(watch_stop())
    try:
        loop.run_forever()
    finally:
        loop.close()

# Start the async web server in its own thread
web_thread = threading.Thread(target=start_async_server, daemon=True)
web_thread.start()


# ============================================================
# ASYNC DISK I/O WORKER
# ============================================================
disk_queue = queue.Queue(maxsize=200)

def disk_worker():
    pin_thread(2) # Core 2 dedicated to Disk I/O
    while not stop_event.is_set() or not disk_queue.empty():
        try:
            item = disk_queue.get(timeout=0.1)
            if item is None: break
            filepath, img = item[0], item[1]
            t0 = time.perf_counter()
            
            # Save raw Depth maps to .u16 Binary Array (Bypasses PNG compression CPU load)
            if filepath.endswith('.png'):
                filepath = filepath.replace('.png', '.u16')
                img.tofile(filepath)
            else:
                cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            _observe_latency("disk_ms", (time.perf_counter() - t0) * 1000.0)
            disk_queue.task_done()
        except queue.Empty: 
            continue

disk_thread = threading.Thread(target=disk_worker, daemon=False)
disk_thread.start()

# ============================================================
# PIPELINE CONFIG
# ============================================================
pipeline = dai.Pipeline()

calib = None
cal_source = "UNKNOWN"

if CALIBRATION_MODE == "factory":
    try:
        with dai.Device() as temp_device:
            calib = temp_device.readFactoryCalibration()
            cal_source = "FACTORY"
            print("[CAL] ✓ Factory calibration loaded (as requested in config)")
    except Exception as e:
        print(f"[CAL] ⚠ Factory cal requested but not available: {e}. Falling back.")

if calib is not None:
    pipeline.setCalibrationData(calib)

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(1, 2)
cam.setFps(TARGET_FPS)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(640, 360)
cam.setVideoSize(640, 360)

if LOCK_EXPOSURE:
    cam.initialControl.setManualExposure(cam_state["exp"], cam_state["iso"])
    cam.initialControl.setManualWhiteBalance(cam_state["wb"])
    print(f"[HW] Locked Camera Exposure: {cam_state['exp']/1000:.1f}ms / ISO {cam_state['iso']} / WB {cam_state['wb']}K")

controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(cam.inputControl)

jpeg_enc = pipeline.create(dai.node.VideoEncoder)
jpeg_enc.setDefaultProfilePreset(TARGET_FPS, dai.VideoEncoderProperties.Profile.MJPEG)
jpeg_enc.setQuality(80)
cam.video.link(jpeg_enc.input)

xout_jpeg = pipeline.create(dai.node.XLinkOut)
xout_jpeg.setStreamName("mjpeg")
xout_jpeg.input.setBlocking(False)
xout_jpeg.input.setQueueSize(1)
jpeg_enc.bitstream.link(xout_jpeg.input)

mono_l = pipeline.create(dai.node.MonoCamera)
mono_r = pipeline.create(dai.node.MonoCamera)
mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_l.setFps(TARGET_FPS)
mono_r.setFps(TARGET_FPS)

stereo = pipeline.create(dai.node.StereoDepth)
if hasattr(dai.node.StereoDepth.PresetMode, "DEFAULT"):
    stereo_preset = dai.node.StereoDepth.PresetMode.DEFAULT
else:
    stereo_preset = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
stereo.setDefaultProfilePreset(stereo_preset)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)

# Drop down Depth resolution directly to 640x360 to save vast USB bandwidth 
stereo.setOutputSize(640, 360) 

cfg = stereo.initialConfig.get()
cfg.postProcessing.spatialFilter.enable = True
cfg.postProcessing.spatialFilter.holeFillingRadius = 2
cfg.postProcessing.temporalFilter.enable = True
cfg.postProcessing.speckleFilter.enable = True
cfg.postProcessing.speckleFilter.speckleRange = 50
try:
    cfg.postProcessing.thresholdFilter.minRange = DEPTH_MIN_MM
    cfg.postProcessing.thresholdFilter.maxRange = DEPTH_MAX_MM
except Exception: 
    pass
stereo.initialConfig.set(cfg)
mono_l.out.link(stereo.left)
mono_r.out.link(stereo.right)

# Empower the VPU Feature Tracker to completely avoid the CPU LK fallback
feat = pipeline.create(dai.node.FeatureTracker)
feat.setHardwareResources(2, 2)  
feat_cfg = feat.initialConfig.get()
if hasattr(feat_cfg, "pyramidLevels"):
    feat_cfg.pyramidLevels = 5

if hasattr(feat_cfg, "cornerDetector") and hasattr(feat_cfg.cornerDetector, "cellGridDimension"):
    feat_cfg.cornerDetector.cellGridDimension = 4
feat.initialConfig.set(feat_cfg)
cam.isp.link(feat.inputImage)

imu_n = pipeline.create(dai.node.IMU)
imu_n.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], reportRate=IMU_RATE)
imu_n.setBatchReportThreshold(5)
imu_n.setMaxBatchReports(20)

xout_i = pipeline.create(dai.node.XLinkOut)
xout_i.setStreamName("imu")
xout_i.input.setBlocking(False)
xout_i.input.setQueueSize(10)
imu_n.out.link(xout_i.input)

sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=30))
sync.inputs["rgb"].setBlocking(False)
sync.inputs["rgb"].setQueueSize(1)
sync.inputs["depth"].setBlocking(False)
sync.inputs["depth"].setQueueSize(1)
sync.inputs["features"].setBlocking(False)
sync.inputs["features"].setQueueSize(1)

cam.isp.link(sync.inputs["rgb"])
stereo.depth.link(sync.inputs["depth"])
feat.outputFeatures.link(sync.inputs["features"])

xout_s = pipeline.create(dai.node.XLinkOut)
xout_s.setStreamName("synced")
xout_s.input.setBlocking(False)
xout_s.input.setQueueSize(1)
sync.out.link(xout_s.input)

# ============================================================
# MAIN
# ============================================================
print(f"Starting VIO ({'Numba JIT' if HAS_NUMBA else 'numpy'})...")
print(f"HUD available at http://<PI_IP>:8080/ (client browser renders overlay)")

ekf = VIO_EKF()
saved_poses = []

with dai.Device(pipeline) as device:
    print("[SYS] Booting device and stabilizing pipeline (1.5s)...")
    time.sleep(1.5)

    if USE_IR_PROJECTOR:
        try:
            device.setIrLaserDotProjectorBrightness(IR_DOT_BRIGHTNESS)
            print(f"[HW] ✓ Active Stereo IR Projector ON ({IR_DOT_BRIGHTNESS}mA)")
        except Exception as e:
            print(f"[HW] ⚠ IR Projector NOT SUPPORTED on this model. (Running Passive Stereo)")

    if calib is None:
        try:
            calib = device.readCalibration()
            cal_source = "CUSTOM (EEPROM)"
            print("[CAL] ✓ Using current device calibration (custom EEPROM / fallback)")
        except Exception as e:
            print(f"[CAL] ⚠ Could not read any calibration: {e}")

    intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, ISP_WIDTH, ISP_HEIGHT)
    K_mat = np.array([
        [intr[0][0], 0,          intr[0][2]],
        [0,          intr[1][1], intr[1][2]],
        [0,          0,          1.0       ]], dtype=np.float64)

    with open(os.path.join(SAVE_DIR, "intrinsics.json"), "w") as f:
        json.dump({
            "width": ISP_WIDTH, "height": ISP_HEIGHT,
            "fx": intr[0][0], "fy": intr[1][1],
            "cx": intr[0][2], "cy": intr[1][2],
            "calibration_source": cal_source
        }, f, indent=2)

    try:
        ext = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A, useSpecTranslation=True)
        T_ci = np.array(ext, dtype=np.float64)
        if T_ci.shape == (3,4): 
            T_ci = np.vstack([T_ci, [0,0,0,1]])
        T_ic = np.linalg.inv(T_ci)
    except Exception as e:
        T_ci = np.eye(4)
        T_ic = np.eye(4)

    imu_t = threading.Thread(target=imu_worker, args=(device,ekf), daemon=True)
    vis_t = threading.Thread(target=visual_worker, args=(ekf,K_mat, T_ic, T_ci), daemon=True)
    lk_thread = threading.Thread(target=lk_worker, daemon=True)
    
    imu_t.start()
    vis_t.start()
    lk_thread.start()

    qs = device.getOutputQueue("synced", 1, False)
    qj = device.getOutputQueue("mjpeg", 1, False) 
    control_q = device.getInputQueue("control")

    count = 0
    frame_ok = False
    prev_fd = {}
    depth_dbuf = None
    prev_depth = None  
    prev_ts = 0.0
    ekf_frame_counter = 0     
    last_saved_ekf_idx = 0
    
    current_applied_wb = 0
    current_applied_exp = 0
    current_applied_iso = 0

    gray_curr = None
    prev_gray = None

    print(f"\n[EKF] Hold still for gravity calibration...")

    has_tty = False
    old_settings = None
    try:
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            has_tty = True
        else:
            print("[WARN] No TTY detected.")
    except Exception as e:
        print(f"[WARN] Terminal setup failed: {e}. Key inputs ignored.")

    last_calib_print = 0
    last_heartbeat = time.time()
    
    telemetry_stats = {
        "visual_queue_drops": 0,
        "disk_queue_drops": 0,
        "forced_gaps_accepted": 0,
        "sync_warnings": 0,
        "budget_violations": 0,
        "mode_transitions": [],
        "recovered_from_pressure": 0
    }
    last_budget_report = time.monotonic()
    mode_last = "BOOT"

    try:
        while True:
            loop_t0 = time.perf_counter()
            with cam_ctrl_lock:
                target_wb = cam_state["wb"]
                target_exp = cam_state["exp"]
                target_iso = cam_state["iso"]
            
            if target_wb != current_applied_wb or target_exp != current_applied_exp or target_iso != current_applied_iso:
                ctrl = dai.CameraControl()
                ctrl.setManualWhiteBalance(target_wb)
                if LOCK_EXPOSURE:
                    ctrl.setManualExposure(target_exp, target_iso)
                control_q.send(ctrl)
                current_applied_wb = target_wb
                current_applied_exp = target_exp
                current_applied_iso = target_iso
                print(f"  [HW] Updated Settings -> WB: {target_wb}K | Exp: {target_exp}us | ISO: {target_iso}")

            if has_tty and select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1).lower()
                if k == 'r':
                    if not ekf.is_ready(): 
                        print("  [WAIT] Gravity not calibrated yet — hold absolutely still!")
                    elif not recording_event.is_set():
                        recording_event.set()
                        runtime_state["bad_streak_counter"] = 0
                        print("\n[>>>] REC")
                        with hud_lock: hud_telemetry["state"] = "GOOD"
                elif k == 's' and recording_event.is_set():
                    recording_event.clear()
                    runtime_state["bad_streak_counter"] = 0
                    print(f"\n[|||] STOP {count}")
                    with hud_lock: hud_telemetry["state"] = "IDLE"; hud_telemetry["message"] = "PAUSED"
                elif k == 'q': 
                    break

            mj = qj.tryGet()
            if mj:
                # Lock-free atomic reference read from DepthAI
                latest_jpeg = mj.getData().tobytes()

            sy = qs.tryGet()
            
            if sy or mj:
                last_heartbeat = time.time()
                
            if time.time() - last_heartbeat > 5.0:
                print("\n[FATAL] HARDWARE QUEUE STALL DETECTED! No frames for 5 seconds.")
                with hud_lock:
                    hud_telemetry["state"] = "BAD"
                    hud_telemetry["message"] = "FATAL: HARDWARE SENSOR STALL"
                break
            
            if sy:
                raw_rgb = sy["rgb"].getCvFrame()
                dep = sy["depth"].getFrame().astype(np.uint16)
                fm = sy["features"]
                
                try: rgb_ts = sy["rgb"].getTimestamp().total_seconds()
                except AttributeError: rgb_ts = time.monotonic()
                host_ts = time.monotonic()

                # Continuous camera/IMU time-offset and drift estimation.
                imu_ts = imu_sync["last_imu_ts"]
                if imu_ts is not None:
                    offset = float(rgb_ts - imu_ts)
                    prev_offset = sync_state["offset_ewma"]
                    if prev_offset is None:
                        sync_state["offset_ewma"] = offset
                    else:
                        sync_state["offset_ewma"] = (1.0 - SYNC_EWMA_ALPHA) * prev_offset + SYNC_EWMA_ALPHA * offset
                    if sync_state["last_cam_ts"] is not None and sync_state["last_host_ts"] is not None:
                        dt_host = max(1e-6, host_ts - sync_state["last_host_ts"])
                        drift = (offset - (prev_offset if prev_offset is not None else offset)) / dt_host
                        sync_state["drift_ewma"] = 0.9 * sync_state["drift_ewma"] + 0.1 * drift
                    sync_state["last_cam_ts"] = rgb_ts
                    sync_state["last_host_ts"] = host_ts

                    warn_sync = (
                        abs(sync_state["offset_ewma"]) > SYNC_WARN_ABS_S or
                        abs(sync_state["drift_ewma"]) > SYNC_WARN_DRIFT_SPS
                    )
                    if warn_sync:
                        telemetry_stats["sync_warnings"] += 1
                    with hud_lock:
                        hud_telemetry["sync_offset_ms"] = float(sync_state["offset_ewma"] * 1000.0)
                        hud_telemetry["sync_drift_ms_s"] = float(sync_state["drift_ewma"] * 1000.0)
                
                # Zero-Math Grayscale: Extracting the Green channel directly
                gray_curr = raw_rgb[:, :, 1].copy()
                
                if CR_ENABLED and CR_REC:
                    rgb_to_save = fast_underwater_restore(raw_rgb, CR_R_MAX, CR_G_MAX)
                else:
                    rgb_to_save = raw_rgb
                
                if not frame_ok:
                    print(f"  [CAL] Frame: {raw_rgb.shape[1]}×{raw_rgb.shape[0]}")
                    frame_ok = True
                    
                if depth_dbuf is None:
                    depth_dbuf = DepthDoubleBuffer(dep.shape[0], dep.shape[1])
                depth_dbuf.write(dep)

                if count == 0 and not ekf.is_ready() and frame_ok:
                    now = time.time()
                    if now - last_calib_print > 3.0:
                        var = ekf._var_tracker.variance()
                        samples = len(ekf._still_accels)
                        print(f"  [EKF] Calibrating... var={var:.4f} (needs < {STATIC_VAR_THR}) "
                              f"samples={samples}/{MIN_GRAV_SAMPLES}")
                        last_calib_print = now
                        
                if ekf.is_ready() and not recording_event.is_set():
                    with hud_lock:
                        if hud_telemetry["message"] == "AWAITING GRAVITY CALIBRATION":
                            hud_telemetry["message"] = "READY - PRESS 'r' TO RECORD"

                if recording_event.is_set() and ekf.is_ready():
                    ekf_frame_counter += 1
                    gap = ekf_frame_counter - last_saved_ekf_idx
                    active_mode = runtime_mode["mode"]
                    
                    accept = False
                    reason = ""
                    valid_ratio = 0.0
                    evaluated_for_hud = False
                    is_forced_gap = False
                    
                    quality_score = 1.0       
                    quality_state = "GOOD"
                    info_score = 0.0
                    feature_cov_score = 0.0
                    parallax_score = 0.0
                    n_tracked = 0

                    if fm:
                        tracked = fm.trackedFeatures
                        n_tracked = len(tracked)
                        feature_cov_score = _feature_coverage_score(tracked, raw_rgb.shape[1], raw_rgb.shape[0])
                        cf_tmp = {t.id: (float(t.position.x), float(t.position.y)) for t in tracked}
                        parallax_score = _parallax_score(cf_tmp, prev_fd)
                    
                    # Optimized Temporal Gradient (Laplacian Variance) on center ROI to avoid downsampling aliasing
                    h_g, w_g = gray_curr.shape
                    roi = gray_curr[h_g//4 : 3*h_g//4, w_g//4 : 3*w_g//4]
                    laplacian_var = cv2.Laplacian(roi, cv2.CV_32F).var()

                    if gap >= GATE_MAX_FRAME_GAP:
                        accept = True
                        is_forced_gap = True
                        telemetry_stats["forced_gaps_accepted"] += 1
                        reason = f"forced_gap({gap})"
                        valid_ratio = float(np.count_nonzero(dep)) / dep.size
                        evaluated_for_hud = True
                        
                    elif gap >= adaptive_gate["min_frame_gap"]:
                        valid_ratio = float(np.count_nonzero(dep)) / dep.size
                        if valid_ratio < GATE_MIN_DEPTH_VALID:
                            reason = f"bad_depth({valid_ratio:.2f})"
                        else:
                            if laplacian_var < LAPLACIAN_PASS_THRESHOLD:
                                reason = f"blur_laplacian({laplacian_var:.1f})"
                            else:
                                # Information-aware gating: depth + blur + coverage + parallax.
                                qc_cfg = CFG.get("quality_control", {})
                                ideal_depth = qc_cfg.get("ideal_depth_ratio", 0.40)
                                severe_blur_equiv = qc_cfg.get("severe_blur_px", 10.0)
                                w_depth = INFO_CFG.get("weight_depth", 0.40)
                                w_blur = INFO_CFG.get("weight_blur", 0.25)
                                w_cov = INFO_CFG.get("weight_coverage", 0.20)
                                w_parallax = INFO_CFG.get("weight_parallax", 0.15)
                                q_depth = min(1.0, valid_ratio / ideal_depth) if ideal_depth > 0 else 0.0
                                q_blur = min(1.0, max(0.0, (laplacian_var - severe_blur_equiv) / 100.0))
                                info_score = (
                                    q_depth * w_depth +
                                    q_blur * w_blur +
                                    feature_cov_score * w_cov +
                                    parallax_score * w_parallax
                                )
                                info_thr = INFO_SCORE_MIN
                                if active_mode == "VISION_WEAK":
                                    info_thr = min(0.95, INFO_SCORE_MIN + VISION_WEAK_INFO_BONUS)
                                elif active_mode == "IMU_ONLY":
                                    info_thr = 1.0  # deterministic disable of visual-dependent acceptance

                                if info_score >= info_thr:
                                    accept = True
                                    reason = f"passed_info(I:{info_score:.2f},L:{laplacian_var:.1f},D:{valid_ratio:.2f})"
                                else:
                                    reason = f"low_info(I:{info_score:.2f},thr:{info_thr:.2f})"
                        evaluated_for_hud = True

                    if evaluated_for_hud:
                        qc_cfg = CFG.get("quality_control", {})
                        ideal_depth = qc_cfg.get("ideal_depth_ratio", 0.40)
                        severe_blur_equiv = 20.0
                        good_thresh = qc_cfg.get("score_good_threshold", 0.75)
                        weak_thresh = qc_cfg.get("score_weak_threshold", 0.40)
                        
                        w_depth = qc_cfg.get("weight_depth", 0.6) 
                        w_blur = qc_cfg.get("weight_blur", 0.4)
                        
                        q_depth = min(1.0, valid_ratio / ideal_depth) if ideal_depth > 0 else 0
                        q_blur  = min(1.0, max(0.0, (laplacian_var - severe_blur_equiv) / 100.0))
                        quality_score = (q_depth * w_depth) + (q_blur * w_blur)
                        if info_score > 0.0:
                            quality_score = 0.7 * quality_score + 0.3 * info_score
                        if is_forced_gap:
                            quality_state = "WEAK"
                            quality_score = min(quality_score, weak_thresh)
                        elif quality_score >= good_thresh:
                            quality_state = "GOOD"
                        elif quality_score >= weak_thresh:
                            quality_state = "WEAK"
                        else:
                            quality_state = "BAD"

                        if quality_state == "BAD":
                            runtime_state["bad_streak_counter"] += 1
                        else:
                            runtime_state["bad_streak_counter"] = 0

                        with hud_lock:
                            hud_telemetry["state"] = quality_state
                            hud_telemetry["score"] = quality_score
                            hud_telemetry["blur"] = laplacian_var 
                            hud_telemetry["depth_pct"] = valid_ratio
                            hud_telemetry["adaptive_gap"] = adaptive_gate["min_frame_gap"]
                            
                            if runtime_state["bad_streak_counter"] >= 8:
                                hud_telemetry["message"] = "CRITICAL: REVERSE TO LAST GOOD VIEW!"
                            elif quality_state == "BAD":
                                if laplacian_var < 35.0:
                                    hud_telemetry["message"] = "SLOW YAW / MOTION BLUR!"
                                elif valid_ratio < CFG.get("keyframe_gating", {}).get("min_depth_valid_ratio", 0.25):
                                    hud_telemetry["message"] = "MOVE CLOSER / POOR DEPTH!"
                                else:
                                    hud_telemetry["message"] = "TRACKING LOST!"
                            else:
                                hud_telemetry["message"] = ""

                    if accept:
                        if active_mode == "DISK_PRESSURE" and (ekf_frame_counter % DISK_PRESSURE_SAVE_STRIDE != 0):
                            accept = False
                            reason = f"disk_pressure_stride({DISK_PRESSURE_SAVE_STRIDE})"

                    if accept:
                        ekf.set_keyframe()
                        T, cov6 = ekf.get_pose()
                        Tc = T @ T_ic
                        
                        saved_poses.append({
                            "frame_id": count,
                            "ekf_frame_idx": ekf_frame_counter,
                            "gate_reason": reason,
                            "quality_score": float(quality_score),
                            "quality_state": quality_state,
                            "information_score": float(info_score),
                            "feature_coverage_score": float(feature_cov_score),
                            "parallax_score": float(parallax_score),
                            "tracked_features": int(n_tracked),
                            "sync_offset_ms": float(hud_telemetry.get("sync_offset_ms", 0.0)),
                            "recovery_mode": active_mode,
                            "is_forced_gap": is_forced_gap, 
                            "pose": Tc.tolist(),
                            "cov6": cov6.tolist()
                        })
                        
                        try:
                            disk_queue.put((os.path.join(RGB_DIR, f"{count:04d}.jpg"), rgb_to_save), timeout=0.5)
                            disk_queue.put((os.path.join(DEPTH_DIR, f"{count:04d}.png"), dep), timeout=0.5)
                            disk_health["consecutive_drops"] = 0
                        except queue.Full:
                            telemetry_stats["disk_queue_drops"] += 1
                            disk_health["consecutive_drops"] += 1
                            if disk_health["consecutive_drops"] >= 3:
                                with hud_lock:
                                    hud_telemetry["message"] = "CRITICAL: DISK SLOW! REDUCE SPEED!"
                                    hud_telemetry["state"] = "BAD"
                                print(f"  [CRITICAL] Disk queue saturated. Consider increasing GATE_MIN_FRAME_GAP.")
                            
                        count += 1
                        last_saved_ekf_idx = ekf_frame_counter
                        
                        if count % 10 == 0:
                            print(f"  [{count:04d} | idx:{ekf_frame_counter}] Health: {quality_score:.2f} ({quality_state}) | {reason}")

                if fm and gray_curr is not None:
                    cf = {t.id:(float(t.position.x),float(t.position.y)) for t in fm.trackedFeatures}
                    ds_curr = depth_dbuf.read() if depth_dbuf else None

                    if prev_fd and prev_depth is not None and ekf.is_ready():
                        if runtime_mode["mode"] != "IMU_ONLY":
                            pp, pc = [], []
                            for fid, c in cf.items():
                                if fid in prev_fd: 
                                    pp.append(prev_fd[fid])
                                    pc.append(c)
                                    
                            if len(pp) >= MIN_FEAT_UPDATE:
                                try: 
                                    n_pts = min(len(pp), MAX_FEATURES)
                                    for i in range(n_pts):
                                        pp_buffer[i, 0] = pp[i][0]
                                        pp_buffer[i, 1] = pp[i][1]
                                        pc_buffer[i, 0] = pc[i][0]
                                        pc_buffer[i, 1] = pc[i][1]
                                        
                                    visual_queue.put_nowait((pp_buffer[:n_pts].copy(), pc_buffer[:n_pts].copy(), prev_depth.copy(), prev_ts))
                                except queue.Full: 
                                    telemetry_stats["visual_queue_drops"] += 1
                                    pass
                    
                    prev_fd = cf
                    prev_gray = gray_curr.copy()
                    # Use offset-compensated camera timestamp for visual update timing.
                    if sync_state["offset_ewma"] is not None:
                        prev_ts = rgb_ts - sync_state["offset_ewma"]
                    else:
                        prev_ts = rgb_ts
                    
                    if ds_curr is not None:
                        valid_ratio = float(np.count_nonzero(ds_curr)) / ds_curr.size
                        if valid_ratio > 0.15:
                            prev_depth = ds_curr.copy()

            time.sleep(0.001)

            _observe_latency("main_loop_ms", (time.perf_counter() - loop_t0) * 1000.0)
            now = time.monotonic()
            if now - last_budget_report >= BUDGET_REPORT_SEC:
                vstat = _latency_percentiles("visual_ms")
                dstat = _latency_percentiles("disk_ms")
                lstat = _latency_percentiles("main_loop_ms")
                vq_fill = visual_queue.qsize() / float(max(1, visual_queue.maxsize))
                dq_fill = disk_queue.qsize() / float(max(1, disk_queue.maxsize))
                pressure = max(vq_fill, dq_fill)
                budget_bad = (
                    vstat["p95"] > VIS_P95_BUDGET_MS or
                    dstat["p95"] > DISK_P95_BUDGET_MS or
                    lstat["p95"] > LOOP_P95_BUDGET_MS
                )
                health = ekf.get_visual_health()
                disk_pressure = (
                    dq_fill > DISK_PRESSURE_QFILL or
                    disk_health["consecutive_drops"] >= 2 or
                    dstat["p95"] > DISK_P95_BUDGET_MS
                )

                if health["mode"] == "VISION_DEGRADED":
                    recovery_state["imu_only_until"] = max(
                        recovery_state["imu_only_until"], now + IMU_ONLY_HOLD_SEC
                    )

                if disk_pressure:
                    mode = "DISK_PRESSURE"
                    adaptive_gate["min_frame_gap"] = max(adaptive_gate["min_frame_gap"], DISK_PRESSURE_MIN_GAP)
                elif now < recovery_state["imu_only_until"]:
                    mode = "IMU_ONLY"
                    adaptive_gate["min_frame_gap"] = max(adaptive_gate["min_frame_gap"], IMU_ONLY_MIN_GAP)
                elif health["mode"] == "VISION_ADAPTIVE" or vstat["p95"] > VIS_P95_BUDGET_MS or vq_fill > QUEUE_PRESSURE_RAISE:
                    mode = "VISION_WEAK"
                    adaptive_gate["min_frame_gap"] = max(adaptive_gate["min_frame_gap"], VISION_WEAK_MIN_GAP)
                else:
                    mode = "NOMINAL"
                    if pressure < QUEUE_PRESSURE_RECOVER and adaptive_gate["min_frame_gap"] > GATE_MIN_FRAME_GAP:
                        adaptive_gate["min_frame_gap"] -= 1
                        telemetry_stats["recovered_from_pressure"] += 1
                    elif pressure > QUEUE_PRESSURE_RAISE or budget_bad:
                        adaptive_gate["min_frame_gap"] = min(GATE_MAX_FRAME_GAP, adaptive_gate["min_frame_gap"] + 1)
                        telemetry_stats["budget_violations"] += 1

                runtime_mode["mode"] = mode
                if mode != mode_last:
                    telemetry_stats["mode_transitions"].append({
                        "t_monotonic": now,
                        "from": mode_last,
                        "to": mode,
                        "adaptive_gap": adaptive_gate["min_frame_gap"],
                        "visual_nis_ema": health.get("nis_ema", 0.0),
                        "disk_q_fill": dq_fill,
                        "visual_q_fill": vq_fill,
                    })
                    mode_last = mode

                with hud_lock:
                    hud_telemetry["mode"] = mode
                    if mode == "DISK_PRESSURE":
                        hud_telemetry["message"] = "DISK_PRESSURE: throttling saves"
                    elif mode == "IMU_ONLY":
                        hud_telemetry["message"] = "IMU_ONLY: visual updates paused"
                    elif mode == "VISION_WEAK":
                        hud_telemetry["message"] = "VISION_WEAK: tightening keyframe gate"

                print(
                    f"[BUDGET] mode={mode} gap={adaptive_gate['min_frame_gap']} "
                    f"V(p95={vstat['p95']:.1f}ms) D(p95={dstat['p95']:.1f}ms) "
                    f"L(p95={lstat['p95']:.1f}ms) q(v={vq_fill:.2f},d={dq_fill:.2f}) "
                    f"NIS={health.get('nis_ema', 0.0):.2f}"
                )
                last_budget_report = now

    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        stop_event.set()
        recording_event.clear()
        
        while not visual_queue.empty():
            try: visual_queue.get_nowait()
            except: break
        visual_queue.put(None)
        
        while not disk_queue.empty():
            try: disk_queue.get_nowait()
            except: break
        disk_queue.put(None)

        imu_t.join(timeout=2.0)
        vis_t.join(timeout=2.0)
        lk_thread.join(timeout=2.0)
        web_thread.join(timeout=2.0)
        disk_thread.join(timeout=30.0)
        
        if has_tty and old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
        with open(POSES_FILE, "w") as f:
            json.dump(saved_poses, f, indent=2)
            
        res_file = os.path.join(SAVE_DIR, "vio_residuals.json")
        with open(res_file, "w") as f:
            json.dump(ekf.residual_log, f, indent=2)
            
        telem_file = os.path.join(SAVE_DIR, "session_telemetry.json")
        telemetry_stats["latency_summary_ms"] = {
            "visual": _latency_percentiles("visual_ms"),
            "disk": _latency_percentiles("disk_ms"),
            "main_loop": _latency_percentiles("main_loop_ms"),
        }
        telemetry_stats["final_adaptive_gap"] = adaptive_gate["min_frame_gap"]
        telemetry_stats["final_mode"] = runtime_mode["mode"]
        telemetry_stats["sync"] = {
            "offset_ms": float((sync_state["offset_ewma"] or 0.0) * 1000.0),
            "drift_ms_per_s": float(sync_state["drift_ewma"] * 1000.0),
        }
        telemetry_stats["ekf_visual_health"] = ekf.get_visual_health()
        with open(telem_file, "w") as f:
            json.dump(telemetry_stats, f, indent=2)
            
        print(f"\n[DONE] {count} keyframes saved → {POSES_FILE}")
        print(f"  VIO telemetry saved → {res_file}")
        print(f"  Session telemetry saved → {telem_file}")
        print(f"  Total EKF ticks monitored: {ekf_frame_counter}")
        if ekf_frame_counter > 0:
            print(f"  Retention rate: {(count/ekf_frame_counter)*100:.1f}%")