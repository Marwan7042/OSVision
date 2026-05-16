import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "max_runtime_s": 1200.0,
    "max_loop_drift_m": 1.0,
    "min_loop_precision": 0.05,
    "min_mesh_vertices": 10000,
    "min_mesh_triangles": 20000,
    "min_tsdf_frames": 20,
}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_loop_drift_m(scan_dir: Path) -> float:
    poses_path = scan_dir / "poses.json"
    if not poses_path.exists():
        return float("inf")
    poses = _load_json(poses_path)
    if len(poses) < 2:
        return float("inf")
    p0 = poses[0]["pose"]
    p1 = poses[-1]["pose"]
    a = [float(p0[0][3]), float(p0[1][3]), float(p0[2][3])]
    b = [float(p1[0][3]), float(p1[1][3]), float(p1[2][3])]
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _run_reconstruct(repo_dir: Path, scan_dir: Path) -> int:
    env = os.environ.copy()
    env["TRITON_NONINTERACTIVE"] = "1"
    env["TRITON_SCAN_DIR"] = str(scan_dir)
    cmd = [sys.executable, "reconstruct.py"]
    return subprocess.call(cmd, cwd=str(repo_dir), env=env)


def _score_one(repo_dir: Path, item: dict, run_reconstruct: bool) -> dict:
    scan_dir = Path(item["scan_dir"]).expanduser()
    if not scan_dir.is_absolute():
        scan_dir = (repo_dir / scan_dir).resolve()

    if run_reconstruct:
        rc = _run_reconstruct(repo_dir, scan_dir)
        if rc != 0:
            return {
                "name": item["name"],
                "scan_dir": str(scan_dir),
                "status": "reconstruct_failed",
                "return_code": rc,
            }

    metrics_path = scan_dir / "reconstruction_metrics.json"
    if not metrics_path.exists():
        return {
            "name": item["name"],
            "scan_dir": str(scan_dir),
            "status": "missing_metrics",
        }

    metrics = _load_json(metrics_path)
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(item.get("thresholds", {}))

    is_closed_loop = bool(item.get("is_closed_loop", True))

    runtime_s = float(metrics.get("timing", {}).get("total_seconds", 1e12))
    loops_found = int(metrics.get("phase_2_loop_closure", {}).get("loops_found", 0))
    loops_tried = int(metrics.get("phase_2_loop_closure", {}).get("candidates_tried", 0))
    loop_precision = float(loops_found / max(1, loops_tried))
    vertices = int(metrics.get("phase_5_mesh", {}).get("vertices", 0))
    triangles = int(metrics.get("phase_5_mesh", {}).get("triangles", 0))
    tsdf_frames = int(metrics.get("phase_4_tsdf", {}).get("frames_integrated", 0))
    drift_m = float(_compute_loop_drift_m(scan_dir)) if is_closed_loop else None

    checks = {
        "runtime_ok": runtime_s <= thresholds["max_runtime_s"],
        "loop_precision_ok": loop_precision >= thresholds["min_loop_precision"],
        "mesh_vertices_ok": vertices >= thresholds["min_mesh_vertices"],
        "mesh_triangles_ok": triangles >= thresholds["min_mesh_triangles"],
        "tsdf_frames_ok": tsdf_frames >= thresholds["min_tsdf_frames"],
    }
    if is_closed_loop:
        checks["drift_ok"] = drift_m <= thresholds["max_loop_drift_m"]
    else:
        checks["drift_ok"] = True
    passed = all(checks.values())

    return {
        "name": item["name"],
        "scan_dir": str(scan_dir),
        "status": "pass" if passed else "fail",
        "metrics": {
            "runtime_s": runtime_s,
            "loop_drift_m": drift_m,
            "is_closed_loop": is_closed_loop,
            "loop_precision": loop_precision,
            "mesh_vertices": vertices,
            "mesh_triangles": triangles,
            "tsdf_frames": tsdf_frames,
        },
        "thresholds": thresholds,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression harness for SLAM reconstruction benchmarks.")
    parser.add_argument(
        "--manifest",
        default="benchmarks/manifest.json",
        help="Path to benchmark manifest JSON.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run reconstruct.py; only score existing metrics files.",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_dir / manifest_path).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 2

    manifest = _load_json(manifest_path)
    benches = manifest.get("benchmarks", [])
    if not benches:
        print("[ERROR] No benchmarks found in manifest.")
        return 2

    started = time.time()
    results = []
    for item in benches:
        if "name" not in item or "scan_dir" not in item:
            print(f"[WARN] Skipping malformed benchmark entry: {item}")
            continue
        results.append(_score_one(repo_dir, item, run_reconstruct=not args.no_run))

    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") not in {"pass"})
    print("\nRegression scorecard")
    print("=" * 80)
    for r in results:
        name = r["name"]
        status = r["status"]
        if status in {"missing_metrics", "reconstruct_failed"}:
            print(f"{name:24s}  {status}")
            continue
        m = r["metrics"]
        drift_str = f"{m['loop_drift_m']:.3f}m" if m["loop_drift_m"] is not None else "n/a"
        print(
            f"{name:24s}  {status:4s}  "
            f"runtime={m['runtime_s']:.1f}s  "
            f"drift={drift_str}  "
            f"loop_prec={m['loop_precision']:.3f}  mesh={m['mesh_vertices']}/{m['mesh_triangles']}  "
            f"tsdf={m['tsdf_frames']}"
        )

    summary = {
        "manifest": str(manifest_path),
        "generated_at_unix": time.time(),
        "duration_s": time.time() - started,
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    out_path = repo_dir / "benchmarks" / "scorecard.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-" * 80)
    print(f"Passed: {passed}  Failed: {failed}")
    print(f"Scorecard written to: {out_path}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
