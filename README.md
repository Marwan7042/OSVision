
## Visual SLAM Pipeline for Underwater Mapping

**Visual-Inertial Odometry and Dense Reconstruction Pipeline for Subsea Environments**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.18%2B-lightgrey.svg)](https://www.open3d.org/)
[![DepthAI](https://img.shields.io/badge/DepthAI-OAK--D-orange.svg)](https://docs.luxonis.com/software-v3/depthai/)
[![Numba](https://img.shields.io/badge/Numba-JIT-green.svg)](https://numba.pydata.org/)

---

## Overview

This is a Simultaneous Localization and Mapping (SLAM) system designed for underwater ROV navigation, addressing domain-specific challenges such as optical refraction, variable turbidity, and dynamic lighting.

The pipeline fuses high-frequency inertial data with stereo vision utilizing a 15-Degree-of-Freedom (DOF) Local Error-State Kalman Filter (ESKF) for real-time state estimation. Offline, the estimated trajectory seeds a global pose graph, followed by Truncated Signed Distance Function (TSDF) volumetric integration to generate a dense 3D mesh.

---

## Architecture and Key Features

* **VPU-Offloaded Acquisition:** DepthAI handles RGB/depth synchronization and feature tracking on-device, so the Raspberry Pi mostly orchestrates capture, telemetry, and logging instead of doing the heavy image work itself.
* **15-DOF ESKF / MSCKF Fusion:** Estimates position, velocity, orientation SO(3), and IMU biases with tightly-coupled visual-inertial updates and manifold pre-integration.
* **Information-Aware Keyframe Gating:** Accepts frames using depth validity, Laplacian blur, feature coverage, and parallax scoring instead of a simple blur-only rule.
* **Asynchronous Concurrency:** Uses separate workers for acquisition, IMU propagation, visual updates, and disk I/O, with zero-copy buffering and Numba acceleration where available.
* **CUDA-First Reconstruction:** Offline reconstruction prefers CUDA when available; OpenCL is used as an iGPU fallback for preprocessing and acceleration on non-CUDA hosts.

---

## Installation

### Prerequisites
* Python 3.8+
* Luxonis OAK-D S2 (or equivalent DepthAI hardware)
* CUDA if applicable, else it falls back to Intel iGPU (Optional, required for OpenCL hardware-accelerated color correction)

### Setup
```bash
git clone [https://github.com/Marwan7042/slam.git](https://github.com/Marwan7042/slam.git)
cd slam
```

```bash
# edge/companion computer (Raspberry pi or Jetson)
make install-pi

# Station laptop
make install-laptop
```

---

## Repository layout

- `src/slam/shared/settings.py` — runtime settings loader
- `src/slam/shared/helpers.py` — shared helper utilities
- `src/slam/estimation/state_estimator.py` — state estimation and filtering
- `src/slam/pipelines/acquisition.py` — acquisition pipeline
- `src/slam/pipelines/reconstruction.py` — reconstruction pipeline
- `src/slam/pipelines/benchmarking.py` — benchmark scoring pipeline
- `src/slam/dashboard/client.py` — browser dashboard client
- `src/slam/dashboard/server.py` — browser dashboard server
- `src/slam/cli.py` — command runner for `python -m slam`
- `requirements/` — pinned dependency sets for Pi and laptop environments
- `benchmarks/` — benchmark manifest and scorecard output
- `scan/` — mission data and HUD layout files

## Usage

The system operates in two distinct phases: real-time data acquisition and offline reconstruction.

### Phase 1: Real-Time VIO (Acquisition)
Execute this script during ROV operation. It initializes the sensor pipeline, runs the ESKF, and logs the trajectory and gated keyframes to the specified output directory.
```bash
make record
```
* **Diagnostic Interface:** Navigate to `http://<ROV_IP>:8080/` to access the real-time HUD, monitor EKF health status, and adjust camera parameters (Exposure/ISO/WB).
* **Terminal Controls:** * `r`: Initiate map recording.
  * `s`: Pause recording.
  * `q`: Terminate pipeline and export session telemetry.

### Phase 2: Global Reconstruction (Offline)
Execute this script post-mission on the host machine to process the acquired dataset, perform loop closure optimization, and extract the 3D mesh.
```bash
make reconstruct
```
* **Output:** `coral_mesh_mm.ply` (Dense 3D point cloud and mesh geometry).

---

## System Components

### 1. Acquisition and Gating (`slam/pipelines/acquisition.py`)
Handles data ingestion, hardware control, VPU feature tracking, and initial data validation.
* **Adaptive Gating:** Evaluates depth validity, Laplacian blur, feature coverage, and parallax to decide which frames become keyframes.
* **Telemetry Watchdog:** Monitors hardware bus activity. Stalls exceeding 5.0 seconds trigger a safe pipeline termination and data flush.
* **HUD Control Loop:** Serves the browser dashboard and forwards camera parameter changes back to DepthAI.

### 2. State Estimator (`slam/estimation/state_estimator.py`)
The primary numerical filter, with core matrix operations compiled via Numba.
* **IMU Kinematic Clamping:** Restricts input accelerations to physical ROV limits ($\pm 25 m/s^2$) to prevent integration instability during hull impacts.
* **Mahalanobis Gating:** Evaluates incoming visual measurements against the state covariance matrix ($\mathbf{P}$). Innovations exceeding the $\chi^2$ threshold are discarded.
* **Covariance Management:** In prolonged periods of visual denial, the filter marginally inflates the covariance matrix to ensure receptivity to future visual updates.

### 3. Reconstruction Backend (`slam/pipelines/reconstruction.py`)
Handles global trajectory optimization and volumetric mapping.
* **Color Equalization:** Uses OpenCL (`cv2.UMat`) when available for accelerated preprocessing, otherwise falls back to CPU.
* **Pose Graph Optimization:** Uses multi-scale ICP and colored ICP to refine tracking and loop closures, with acceptance gated by fitness and RMSE.
* **TSDF Integration:** Fuses the depth maps into a voxel grid; CUDA is preferred when available, otherwise the pipeline falls back to the legacy CPU path.

---

## Configuration (`src/config/config.json`)

System parameters are defined externally in `src/config/config.json`.

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
