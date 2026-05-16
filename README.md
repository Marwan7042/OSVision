
## Visual SLAM Pipeline for Underwater Mapping

**Visual-Inertial Odometry and Dense Reconstruction Pipeline for Subsea Environments**

[![Python]([https://img.shields.io/badge/Python-3.8%2B-blue.svg](https://www.python.org/))]()
[![Open3D]([https://img.shields.io/badge/Open3D-0.16%2B-lightgrey.svg](https://www.open3d.org/))]()
[![DepthAI]([https://img.shields.io/badge/DepthAI-OAK--D-orange.svg](https://docs.luxonis.com/software-v3/depthai/))]()
[![Numba]([https://img.shields.io/badge/Numba-JIT-green.svg](https://numba.pydata.org/))]()

---

## Overview

This is a Simultaneous Localization and Mapping (SLAM) system designed for underwater ROV navigation, addressing domain-specific challenges such as optical refraction, variable turbidity, and dynamic lighting. 

The pipeline fuses high-frequency inertial data with stereo vision utilizing a 15-Degree-of-Freedom (DOF) Local Error-State Kalman Filter (ESKF) for real-time state estimation. Offline, the estimated trajectory seeds a global pose graph, followed by Truncated Signed Distance Function (TSDF) volumetric integration to generate a dense 3D mesh.

---

## Architecture and Key Features

* **Hardware-Level Sensor Synchronization:** Utilizes Luxonis DepthAI spatial nodes to align RGB, depth, and VPU feature timestamps, mitigating Out-Of-Sequence Measurement (OOSM) errors.
* **15-DOF ESKF Formulation:** Estimates position, velocity, orientation SO(3), and IMU biases. Implements a local error-state Lie algebra formulation to prevent covariance singularities.
* **Dynamic Motion-Blur Gating:** Calculates estimated pixel blur as a function of focal length and real-time angular velocity. Frames exceeding the defined threshold are rejected prior to visual processing.
* **CPU Optical Flow Fallback:** In the event of VPU tracking failure (e.g., rapid illumination changes), the system initiates CPU-bound Lucas-Kanade optical flow to maintain feature tracking and state continuity.
* **Asynchronous Concurrency:** Implements isolated threads for hardware acquisition, visual processing, and disk I/O, utilizing zero-copy double-buffering and Numba JIT compilation to manage execution latency.

---

## Installation

### Prerequisites
* Python 3.8+
* Luxonis OAK-D S2 (or equivalent DepthAI hardware)
* Intel iGPU (Optional, required for OpenCL hardware-accelerated color correction)

### Setup
```bash
git clone [https://github.com/Marwan7042/slam.git](https://github.com/Marwan7042/slam.git)
cd slam
```

Create a dedicated virtual environment on each machine:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install machine-specific dependencies:
```bash
# Raspberry Pi 4 (recording)
pip install -r requirements-pi.txt

# Station laptop (reconstruction)
pip install -r requirements-laptop.txt
```

Or use the included Makefile:
```bash
# Raspberry Pi 4
make install-pi

# Station laptop
make install-laptop
```

---

## Usage

The system operates in two distinct phases: real-time data acquisition and offline reconstruction.

### Phase 1: Real-Time VIO (Acquisition)
Execute this script during ROV operation. It initializes the sensor pipeline, runs the ESKF, and logs the trajectory and gated keyframes to the specified output directory.
```bash
python record.py
# or: make record
```
* **Diagnostic Interface:** Navigate to `http://<ROV_IP>:8080/` to access the real-time HUD, monitor EKF health status, and adjust camera parameters (Exposure/ISO/WB).
* **Terminal Controls:** * `r`: Initiate map recording.
  * `s`: Pause recording.
  * `q`: Terminate pipeline and export session telemetry.

### Phase 2: Global Reconstruction (Offline)
Execute this script post-mission on the host machine to process the acquired dataset, perform loop closure optimization, and extract the 3D mesh.
```bash
python reconstruct.py
# or: make reconstruct
```
* **Output:** `coral_mesh_mm.ply` (Dense 3D point cloud and mesh geometry).

---

## System Components

### 1. Acquisition and Gating (`record.py`)
Handles data ingestion, thread management, and initial data validation.
* **Statistical Gating:** Evaluates spatial point variance to filter unstructured feature tracking and applies the gyroscopic blur threshold.
* **Telemetry Watchdog:** Monitors hardware bus activity. Stalls exceeding 5.0 seconds trigger a safe pipeline termination and data flush.

### 2. State Estimator (`ekf.py`)
The primary numerical filter, with core matrix operations compiled via Numba.
* **IMU Kinematic Clamping:** Restricts input accelerations to physical ROV limits ($\pm 25 m/s^2$) to prevent integration instability during hull impacts.
* **Mahalanobis Gating:** Evaluates incoming visual measurements against the state covariance matrix ($\mathbf{P}$). Innovations exceeding the $\chi^2$ threshold are discarded.
* **Covariance Management:** In prolonged periods of visual denial, the filter marginally inflates the covariance matrix to ensure receptivity to future visual updates.

### 3. Reconstruction Backend (`reconstruct.py`)
Handles global trajectory optimization and volumetric mapping.
* **Color Equalization:** Optionally leverages OpenCL (`cv2.UMat`) for hardware-accelerated red-channel attenuation correction.
* **Pose Graph Optimization:** Utilizes Multi-Scale ICP for scan matching. Applies Singular Value Decomposition (SVD) on local point cloud normals to reject geometrically degenerate (planar) loop closures.
* **TSDF Integration:** Fuses the depth maps into a voxel grid, utilizing CUDA (`o3d.t.geometry.VoxelBlockGrid`) when compatible hardware is present.

---

## Configuration (`config.json`)

System parameters are defined externally in `config.json`.

### Practical tuning guide

| Group | Parameter | What it controls | Typical tweak direction |
| :--- | :--- | :--- | :--- |
| `paths` | `scan_dir` | Output session folder (`rgb/`, `depth/`, `poses.json`). | Change per mission/dataset. |
| `hardware` | `target_fps` | Camera/stereo frame rate on Pi. | Lower if Pi overheats or drops frames; raise only if budgets remain healthy. |
| `hardware` | `imu_rate_hz` | IMU sample rate for propagation. | Keep high for aggressive motion; lower if CPU constrained. |
| `hardware` | `calibration_mode` | `"custom"` EEPROM vs `"factory"` calibration selection. | Use `custom` underwater unless factory is known-good. |
| `hardware` | `exposure_time_us`, `iso_sensitivity` | Brightness/noise tradeoff in low light. | Increase for dark scenes, reduce to limit blur/noise. |
| `hardware` | `decimate_factor` | Reconstruction-side image downscaling factor. | Increase to reduce laptop load, decrease for finer geometry. |
| `keyframe_gating` | `min_frame_gap`, `max_frame_gap` | Lower/upper spacing between accepted keyframes. | Increase `min` for less load; keep `max` as safety to avoid starvation. |
| `keyframe_gating` | `min_depth_valid_ratio` | Minimum valid-depth fraction to accept frame. | Raise in clean water for quality, lower in turbid scenes to keep continuity. |
| `quality_control` | `score_good_threshold`, `score_weak_threshold` | GOOD/WEAK/BAD state boundaries. | Raise for stricter quality; lower if too many frames become BAD. |
| `quality_control` | `ideal_depth_ratio` | Target depth coverage ratio for scoring. | Lower for sparse/degraded water visibility. |
| `ekf_tuning` | `static_variance_threshold`, `min_gravity_samples` | Startup gravity-init strictness. | Relax slightly if init is too slow; tighten if false starts happen. |
| `ekf_tuning` | `min_feature_update` | Minimum shared tracks before visual update attempt. | Raise for robustness; lower if feature count is often weak. |
| `ekf_tuning` | `depth_patch_radius`, `depth_min_mm`, `depth_max_mm` | Depth sampling robustness + valid range for visual update. | Increase patch in noisy depth; tighten min/max to reject bad depth tails. |
| `ekf_tuning` | `imu_vibration_multiplier` | Process noise inflation for vibration-heavy platforms. | Raise if filter is overconfident under vibration; lower if too noisy/drifty. |
| `information_gating` | `min_information_score` | Required combined info score (depth/blur/coverage/parallax). | Raise to keep only high-value frames; lower to avoid starvation. |
| `information_gating` | `weight_depth`, `weight_blur`, `weight_coverage`, `weight_parallax` | Relative weighting of frame usefulness terms. | Rebalance based on failure mode (e.g., more depth weight in low-texture scenes). |
| `runtime_budgets` | `visual_p95_ms`, `disk_p95_ms`, `main_loop_p95_ms` | p95 latency targets driving adaptive throttling. | Increase only if hardware cannot sustain current targets. |
| `runtime_budgets` | `queue_pressure_raise`, `queue_pressure_recover` | Queue fill levels for entering/recovering pressure mode. | Lower raise threshold for earlier protection. |
| `recovery_modes` | `vision_weak_min_frame_gap`, `imu_only_min_frame_gap` | Minimum gap in degraded states. | Increase to shed Pi load faster when unstable. |
| `recovery_modes` | `imu_only_hold_sec` | How long IMU-only mode is held once triggered. | Increase for stability, decrease for quicker visual re-entry. |
| `recovery_modes` | `disk_pressure_min_frame_gap`, `disk_pressure_save_stride` | Deterministic disk-pressure throttling strength. | Raise for slow SD/storage media. |
| `time_sync` | `warn_abs_offset_s`, `warn_drift_s_per_s` | Camera–IMU sync warning sensitivity. | Tighten for high-precision missions; relax if sensor clocks are noisy. |
| `reconstruction` | `voxel_tracking_m`, `voxel_loop_m` | Downsample voxel sizes for tracking/loop closure clouds. | Increase for speed, decrease for tighter alignment detail. |
| `reconstruction` | `voxel_tsdf_m` | Final TSDF resolution. | Smaller = better detail + higher memory/runtime. |
| `reconstruction` | `depth_quantile_pruning` | Prunes far/noisy tail depths before reconstruction. | Lower for aggressive outlier pruning. |
| `reconstruction` | `loop_closure_interval`, `min_loop_frame_distance`, `loop_top_k` | Loop candidate density and search breadth. | Increase interval/lower top-k for speed; opposite for hard loop closures. |
| `reconstruction` | `z_regularize` | Post-opt vertical drift smoothing. | Keep on for long underwater passes with weak vertical observability. |
| `loop_closure_quality` | `max_translation_residual_m`, `max_rotation_residual_deg` | Consistency gates for accepting loops. | Tighten to reduce false loops; loosen if true loops are rejected. |
| `loop_closure_quality` | `switchable_min_weight`, `min_fgr_fitness` | Minimum loop trust and coarse match quality. | Raise for stricter robustness, lower for difficult low-feature datasets. |
| `tsdf_quality` | `min_depth_valid_ratio` | Minimum usable depth ratio to integrate a frame. | Raise to protect mesh quality; lower to keep more frames in poor visibility. |
| `tsdf_quality` | `dynamic_depth_truncation`, `dynamic_trunc_ewma_alpha` | Adaptive depth truncation behavior during fusion. | Enable for varying range scenes; lower alpha for slower/steadier adaptation. |
| `tsdf_quality` | `max_tracking_rmse_skip` | Skip TSDF frames with poor tracking quality. | Tighten for cleaner mesh, loosen if integration becomes too sparse. |

### New reliability/quality controls

- `runtime_budgets`: Enforces p95 latency budgets for visual update, disk I/O, and main loop; drives adaptive keyframe gap under backpressure.
- `recovery_modes`: Explicit degraded-state policies (`VISION_WEAK`, `IMU_ONLY`, `DISK_PRESSURE`) with deterministic throttling/recovery behavior.
- `time_sync`: Continuously tracks camera-IMU offset/drift and raises sync warnings when thresholds are exceeded.
- `information_gating`: Adds coverage/parallax-aware keyframe scoring to avoid low-information frames.
- `loop_closure_quality`: Adds loop consistency gates and switchable loop-edge weighting in pose graph construction.
- `tsdf_quality`: Rejects low-confidence depth frames and enables dynamic depth truncation during TSDF fusion.

## Regression harness

Run full regression (reconstruct + score):
```bash
make regression
```

Score existing outputs only:
```bash
make regression-score
```

Benchmark list and thresholds are in `benchmarks/manifest.json`. The harness writes `benchmarks/scorecard.json` with pass/fail checks for runtime, loop drift proxy, loop precision, mesh completeness, and TSDF frame coverage.

For non-return trajectories, set `"is_closed_loop": false` in the manifest entry. In that mode, the drift gate is skipped and scoring relies on runtime, loop precision, mesh completeness, and TSDF coverage.
