"""
TIGHTLY-COUPLED MSCKF VIO — OAK-D S2 — KEYFRAME GATED & HW OPTIMIZED
=============================================================================
"""

import depthai as dai
import cv2
import numpy as np
import os, json, time, threading, queue
import http.server, socketserver
from urllib.parse import urlparse, parse_qs
from datetime import timedelta
import sys, select, termios, tty
import math

# IMPORT OUR CUSTOM MODULES
from load_config import CFG
from utils import HAS_NUMBA, fast_underwater_restore
from ekf import VIO_EKF

# ============================================================
# CONFIG (DERIVED FROM JSON)
# ============================================================
SAVE_DIR   = CFG["paths"]["scan_dir"]
RGB_DIR    = os.path.join(SAVE_DIR, "rgb")
DEPTH_DIR  = os.path.join(SAVE_DIR, "depth")
POSES_FILE = os.path.join(SAVE_DIR, "poses.json")
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

TARGET_FPS = CFG["hardware"]["target_fps"]
TIME_OFFSET_MS = CFG["hardware"].get("time_offset_cam_imu_ms", 0.0)
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
jpeg_lock    = threading.Lock()
recording_event = threading.Event()
stop_event   = threading.Event()

disk_queue = queue.Queue(maxsize=200)
disk_health = {"consecutive_drops": 0}

cam_ctrl_lock = threading.Lock()
cam_state = {
    "wb": 4600,
    "exp": EXPOSURE_TIME_US,
    "iso": ISO_SENSITIVITY
}

latest_preview = None
hud_telemetry = {
    "state": "IDLE",
    "score": 1.0,
    "blur": 0.0,
    "depth_pct": 0.0,
    "active_tracks": 0,
    "message": "AWAITING GRAVITY CALIBRATION"
}
hud_lock = threading.Lock()
runtime_state = {"bad_streak_counter": 0} 

# ============================================================
# THREADS
# ============================================================
def imu_worker(device, ekf):
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
                        try: 
                            ts = a.timestamp.get().total_seconds()
                        except AttributeError: 
                            ts = time.monotonic()
                    
                    # Apply hardware time offset for perfect Manifold Integration
                    ts_corrected = ts - (TIME_OFFSET_MS / 1000.0)
                            
                    ab[0], ab[1], ab[2] = a.x, a.y, a.z
                    gb[0], gb[1], gb[2] = g.x, g.y, g.z
                    ekf.feed_imu(ab, gb, ts_corrected)
            else:
                time.sleep(0.001)
        except Exception as e:
            if stop_event.is_set(): 
                break
            print(f"  [IMU] {e}")
            time.sleep(0.01)

def disk_worker():
    while not stop_event.is_set() or not disk_queue.empty():
        try:
            item = disk_queue.get(timeout=0.1)
            if item is None: 
                break
                
            # Throttled Zlib Compression for PNG Depth Maps
            if item[0].endswith('.png'):
                cv2.imwrite(item[0], item[1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                cv2.imwrite(item[0], item[1])
                
            disk_queue.task_done()
        except queue.Empty: 
            continue

# ============================================================
# WEB SERVER (DUAL STREAM + GUI SLIDER + HARDENED PARSING)
# ============================================================
class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/set_wb':
            try:
                val = int(parse_qs(parsed_path.query)['v'][0])
                with cam_ctrl_lock: cam_state["wb"] = max(2500, min(8000, val))
            except Exception: pass
            self.send_response(200); self.end_headers(); return

        if parsed_path.path == '/set_exp':
            try:
                val = int(parse_qs(parsed_path.query)['v'][0])
                with cam_ctrl_lock: cam_state["exp"] = max(1000, min(33000, val))
            except Exception: pass
            self.send_response(200); self.end_headers(); return

        if parsed_path.path == '/set_iso':
            try:
                val = int(parse_qs(parsed_path.query)['v'][0])
                with cam_ctrl_lock: cam_state["iso"] = max(100, min(1600, val))
            except Exception: pass
            self.send_response(200); self.end_headers(); return

        if parsed_path.path == '/stream':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache,private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace;boundary=FRAME')
            self.end_headers()
            try:
                while not stop_event.is_set():
                    with jpeg_lock: 
                        fd = latest_jpeg
                    if fd:
                        self.wfile.write(b'--FRAME\r\nContent-Type:image/jpeg\r\n')
                        self.wfile.write(f'Content-Length:{len(fd)}\r\n\r\n'.encode())
                        self.wfile.write(fd)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.05)
            except Exception: 
                pass
                
        elif parsed_path.path == '/hud':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache,private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace;boundary=FRAME')
            self.end_headers()
            try:
                while not stop_event.is_set():
                    with hud_lock:
                        frame_ref = latest_preview
                        telem = hud_telemetry.copy()
                    
                    frame = frame_ref.copy() if frame_ref is not None else None
                    
                    if frame is not None:
                        if CR_ENABLED and CR_HUD:
                            frame = fast_underwater_restore(frame, CR_R_MAX, CR_G_MAX)
                            
                        h, w = frame.shape[:2]
                        
                        if telem["state"] == "GOOD":   color = (0, 255, 0)      
                        elif telem["state"] == "WEAK": color = (0, 255, 255)    
                        elif telem["state"] == "BAD":  color = (0, 0, 255)      
                        else:                          color = (255, 255, 255)  
                        
                        cv2.rectangle(frame, (0, 0), (w, h), color, 6)
                        cv2.putText(frame, f"STATE: {telem['state']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(frame, f"LAPLV: {telem['blur']:.1f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"DEPTH: {telem['depth_pct']*100:.1f}%", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"TRACKS: {telem.get('active_tracks', 0)}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        if recording_event.is_set():
                            cv2.putText(frame, "REC", (w - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                            cv2.circle(frame, (w - 100, 27), 8, (0, 0, 255), -1)
                        
                        cv2.drawMarker(frame, (w//2, h//2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
                        
                        if telem["message"]:
                            text_size = cv2.getTextSize(telem["message"], cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                            text_x = (w - text_size[0]) // 2
                            cv2.putText(frame, telem["message"], (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        fd = jpeg.tobytes()
                        
                        self.wfile.write(b'--FRAME\r\nContent-Type:image/jpeg\r\n')
                        self.wfile.write(f'Content-Length:{len(fd)}\r\n\r\n'.encode())
                        self.wfile.write(fd)
                        self.wfile.write(b'\r\n')
                        
                    time.sleep(0.1) 
            except Exception: 
                pass
                
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = f'''<html><head><style>
                body {{ background: #0b0b0b; color: #ececec; font-family: sans-serif; margin: 0; padding: 10px; text-align: center; overflow-x: hidden; }}
                h2 {{ font-size: 1.0em; margin: 10px 0; color: #888; text-transform: uppercase; letter-spacing: 2px; }}
                .control-bar {{ background: #1a1a1a; padding: 10px; border-radius: 4px; display: flex; justify-content: center; gap: 40px; margin-bottom: 20px; border-bottom: 2px solid #333; }}
                .control-item {{ display: flex; align-items: center; gap: 10px; }}
                input[type=range] {{ width: 140px; cursor: pointer; }}
                .flex-row {{ display: flex; justify-content: space-evenly; align-items: flex-start; gap: 15px; width: 100vw; }}
                .video-container {{ flex: 1; max-width: 48vw; }}
                .video-feed {{ width: 100%; aspect-ratio: 16 / 9; border: 2px solid #333; border-radius: 2px; background: #000; }}
                #wb-lbl {{ color: #00eeff; }} #exp-lbl {{ color: #ffaa00; }} #iso-lbl {{ color: #ff4444; }}
            </style></head>
            <body>
                <div class="control-bar">
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
                </div>
                <div class="flex-row">
                    <div class="video-container"><h2>Pilot HUD (Restored)</h2><img src="/hud" class="video-feed"></div>
                    <div class="video-container"><h2>Hardware Feed (Raw)</h2><img src="/stream" class="video-feed"></div>
                </div>
            </body></html>'''
            self.wfile.write(html.encode('utf-8'))
                             
    def log_message(self, *a): 
        pass

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer): 
    pass
    
threading.Thread(target=lambda: ThreadedHTTPServer(('', 8080), MJPEGHandler).serve_forever(), daemon=True).start()

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
xout_jpeg.input.setQueueSize(2)
jpeg_enc.bitstream.link(xout_jpeg.input)

xout_prev = pipeline.create(dai.node.XLinkOut)
xout_prev.setStreamName("preview")
xout_prev.input.setBlocking(False)
xout_prev.input.setQueueSize(2)
cam.preview.link(xout_prev.input)

mono_l = pipeline.create(dai.node.MonoCamera)
mono_r = pipeline.create(dai.node.MonoCamera)
mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_l.setFps(TARGET_FPS)
mono_r.setFps(TARGET_FPS)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)
stereo.setOutputSize(ISP_WIDTH, ISP_HEIGHT)

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

feat = pipeline.create(dai.node.FeatureTracker)
feat.setHardwareResources(1, 1)
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
sync.inputs["rgb"].setQueueSize(2)
sync.inputs["depth"].setBlocking(False)
sync.inputs["depth"].setQueueSize(2)
sync.inputs["features"].setBlocking(False)
sync.inputs["features"].setQueueSize(2)

cam.isp.link(sync.inputs["rgb"])
stereo.depth.link(sync.inputs["depth"])
feat.outputFeatures.link(sync.inputs["features"])

xout_s = pipeline.create(dai.node.XLinkOut)
xout_s.setStreamName("synced")
xout_s.input.setBlocking(False)
xout_s.input.setQueueSize(2)
sync.out.link(xout_s.input)

# ============================================================
# MAIN
# ============================================================
print(f"Starting Tightly-Coupled MSCKF VIO ({'Numba JIT' if HAS_NUMBA else 'numpy'})...")
print(f"HUD available at http://<IP>:8080/")

ekf = VIO_EKF()
saved_poses = []

disk_thread = threading.Thread(target=disk_worker, daemon=False)
disk_thread.start()

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
    imu_t.start()

    qs = device.getOutputQueue("synced", 4, False)
    qj = device.getOutputQueue("mjpeg", 2, False) 
    qp = device.getOutputQueue("preview", 2, False) 
    control_q = device.getInputQueue("control")

    count = 0
    frame_ok = False
    ekf_frame_counter = 0     
    last_saved_ekf_idx = 0
    
    current_applied_wb = 0
    current_applied_exp = 0
    current_applied_iso = 0

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
        "disk_queue_drops": 0,
        "forced_gaps_accepted": 0,
    }

    try:
        while True:
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
                with jpeg_lock: 
                    latest_jpeg = mj.getData().tobytes()

            mp = qp.tryGet()
            if mp:
                frame_np = mp.getCvFrame()
                with hud_lock:
                    latest_preview = frame_np

            sy = qs.tryGet()
            
            if sy or mj or mp:
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
                
                # Zero-Math Grayscale: Extracting the Green channel directly
                gray_curr = raw_rgb[:, :, 1].copy()
                
                if CR_ENABLED and CR_REC:
                    rgb_to_save = fast_underwater_restore(raw_rgb, CR_R_MAX, CR_G_MAX)
                else:
                    rgb_to_save = raw_rgb
                
                if not frame_ok:
                    print(f"  [CAL] Frame: {raw_rgb.shape[1]}×{raw_rgb.shape[0]}")
                    frame_ok = True

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
                    
                    accept = False
                    reason = ""
                    valid_ratio = 0.0
                    evaluated_for_hud = False
                    is_forced_gap = False
                    
                    quality_score = 1.0       
                    quality_state = "GOOD"
                    
                    # Optimized Temporal Gradient (Laplacian Variance) using downsampling and float32
                    laplacian_var = cv2.Laplacian(gray_curr[::4, ::4], cv2.CV_32F).var()

                    if gap >= GATE_MAX_FRAME_GAP:
                        accept = True
                        is_forced_gap = True
                        telemetry_stats["forced_gaps_accepted"] += 1
                        reason = f"forced_gap({gap})"
                        valid_ratio = float(np.count_nonzero(dep)) / dep.size
                        evaluated_for_hud = True
                        
                    elif gap >= GATE_MIN_FRAME_GAP:
                        valid_ratio = float(np.count_nonzero(dep)) / dep.size
                        if valid_ratio < GATE_MIN_DEPTH_VALID:
                            reason = f"bad_depth({valid_ratio:.2f})"
                        else:
                            if laplacian_var < 50.0:
                                reason = f"blur_laplacian({laplacian_var:.1f})"
                            else:
                                accept = True
                                reason = f"passed_gate(L:{laplacian_var:.1f},D:{valid_ratio:.2f})"
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
                        # Extract VPU feature tracks directly
                        f_dict = {t.id: (t.position.x, t.position.y) for t in fm.trackedFeatures}
                        
                        # Feed the features to MSCKF. Note: Triangulation is handled internally using the camera poses.
                        ekf.set_keyframe()
                        success, active_tracks = ekf.add_visual_tracks(count, f_dict, K_mat, T_ic)
                        T, cov6 = ekf.get_pose()
                        Tc = T @ T_ic
                        
                        saved_poses.append({
                            "frame_id": count,
                            "ekf_frame_idx": ekf_frame_counter,
                            "gate_reason": reason,
                            "quality_score": float(quality_score),
                            "quality_state": quality_state,
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
                            print(f"  [{count:04d} | idx:{ekf_frame_counter}] "
                                  f"Health: {quality_score:.2f} ({quality_state}) | {reason} | Tracks: {active_tracks}")

                    with hud_lock:
                        hud_telemetry["active_tracks"] = len(ekf.tracks)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        stop_event.set()
        recording_event.clear()
        
        while not disk_queue.empty():
            try: disk_queue.get_nowait()
            except: break
        disk_queue.put(None)

        imu_t.join(timeout=2.0)
        disk_thread.join(timeout=30.0)
        
        if has_tty and old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
        with open(POSES_FILE, "w") as f:
            json.dump(saved_poses, f, indent=2)
            
        res_file = os.path.join(SAVE_DIR, "vio_residuals.json")
        with open(res_file, "w") as f:
            json.dump(ekf.residual_log, f, indent=2)
            
        telem_file = os.path.join(SAVE_DIR, "session_telemetry.json")
        with open(telem_file, "w") as f:
            json.dump(telemetry_stats, f, indent=2)
            
        print(f"\n[DONE] {count} keyframes saved → {POSES_FILE}")
        print(f"  VIO telemetry saved → {res_file}")
        print(f"  Session telemetry saved → {telem_file}")
        print(f"  Total EKF ticks monitored: {ekf_frame_counter}")
        if ekf_frame_counter > 0:
            print(f"  Retention rate: {(count/ekf_frame_counter)*100:.1f}%")