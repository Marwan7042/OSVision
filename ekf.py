import math
import numpy as np
import threading
import cv2
from utils import RunningVariance
from load_config import CFG

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): return args[0]
        def decorator(func): return func
        return decorator

# Import EKF constants securely from CFG
STATIC_VAR_THR   = CFG["ekf_tuning"]["static_variance_threshold"]
STATIC_WIN       = CFG["ekf_tuning"]["static_variance_window"]
MIN_GRAV_SAMPLES = CFG["ekf_tuning"]["min_gravity_samples"]
MIN_FEAT_UPDATE  = CFG["ekf_tuning"]["min_feature_update"]
DEPTH_PATCH_R    = CFG["ekf_tuning"]["depth_patch_radius"]
DEPTH_MIN_MM     = CFG["ekf_tuning"]["depth_min_mm"]
DEPTH_MAX_MM     = CFG["ekf_tuning"]["depth_max_mm"]

# MSCKF Tracking parameters (fallback to defaults if not in CFG yet)
MSCKF_WINDOW     = CFG["ekf_tuning"].get("msckf_window_size", 12)
MIN_TRACK        = CFG["ekf_tuning"].get("min_feature_track_length", 4)

# --- IMU NOISE PARAMETERS ---
_BASE_ACCEL_ND  = 160e-6 * 9.81
_BASE_GYRO_ND   = np.deg2rad(0.007)

# Undertuned Bias Random Walk (BRW)
# Empirically tuned for underwater ROV (thermal gradients + vibration harmonics)
ACCEL_BRW = 2.0e-3 * 9.81         # 2.0 mg 
GYRO_BRW  = np.deg2rad(1.5 / 3600.0) # 0.00042 rad/s

VIB_MULTIPLIER = CFG["ekf_tuning"].get("imu_vibration_multiplier", 15.0)

ACCEL_ND = _BASE_ACCEL_ND * VIB_MULTIPLIER
GYRO_ND  = _BASE_GYRO_ND * VIB_MULTIPLIER

VIS_NOISE_P   = (0.010)**2
VIS_NOISE_PHI = (np.deg2rad(1.0))**2
REORTHO_INTERVAL = 500
VIS_NIS_CHI2_95 = 12.592
VIS_NIS_CHI2_99 = 16.812

# ============================================================
# NUMBA JIT KERNELS (Preserving your custom optimizations)
# ============================================================

@njit(cache=True, fastmath=True)
def skew(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ], dtype=np.float64)

@njit(cache=True)
def project_nullspace(H_f, r):
    """SVD extraction of the left null-space to eliminate 3D landmarks from the state."""
    U, S, Vt = np.linalg.svd(H_f, full_matrices=True)
    A = U[:, 3:] 
    r_o = A.T @ r
    return r_o, A

@njit(cache=True, fastmath=True)
def _rodrigues_jit(wx,wy,wz,out):
    t2=wx*wx+wy*wy+wz*wz; t=math.sqrt(t2)
    if t<1e-9:
        out[0,0]=1.0;out[0,1]=-wz;out[0,2]=wy;out[1,0]=wz;out[1,1]=1.0;out[1,2]=-wx;out[2,0]=-wy;out[2,1]=wx;out[2,2]=1.0;return
    
    if t > 3.13: 
        t = 3.13
        scale = 3.13 / math.sqrt(t2)
        wx *= scale; wy *= scale; wz *= scale
        
    c=math.cos(t);s=math.sin(t);tc=1.0-c;it=1.0/t;x=wx*it;y=wy*it;z=wz*it
    out[0,0]=tc*x*x+c;out[0,1]=tc*x*y-s*z;out[0,2]=tc*x*z+s*y;out[1,0]=tc*x*y+s*z;out[1,1]=tc*y*y+c;out[1,2]=tc*y*z-s*x;out[2,0]=tc*x*z-s*y;out[2,1]=tc*y*z+s*x;out[2,2]=tc*z*z+c

@njit(cache=True, fastmath=True)
def _mat3_mul(A,B,out):
    for i in range(3):
        for j in range(3):
            s=0.0
            for k in range(3): s+=A[i,k]*B[k,j]
            out[i,j]=s

@njit(cache=True, fastmath=True)
def _mat3_vec(A,v,out):
    out[0]=A[0,0]*v[0]+A[0,1]*v[1]+A[0,2]*v[2];out[1]=A[1,0]*v[0]+A[1,1]*v[1]+A[1,2]*v[2];out[2]=A[2,0]*v[0]+A[2,1]*v[1]+A[2,2]*v[2]

@njit(cache=True, fastmath=True)
def _triple_product_15(F,P,Qd,out,tmp):
    for i in range(15):
        for j in range(15):
            s=0.0
            for k in range(15): s+=F[i,k]*P[k,j]
            tmp[i,j]=s
    for i in range(15):
        for j in range(15):
            s=0.0
            for k in range(15): s+=tmp[i,k]*F[j,k]
            out[i,j]=s+Qd[i,j]

@njit(cache=True, fastmath=True)
def _symmetrise_15(P):
    for i in range(15):
        for j in range(i+1,15): avg=0.5*(P[i,j]+P[j,i]);P[i,j]=avg;P[j,i]=avg

@njit(cache=True, fastmath=True)
def _build_F_and_Qd_jit(F,Qd,R,a_b,w_b,dt,na_var,ng_var,nba_var,nbg_var):
    for i in range(15):
        for j in range(15): F[i,j]=0.0;Qd[i,j]=0.0
        F[i,i]=1.0
    F[0,3]=dt;F[1,4]=dt;F[2,5]=dt;ax,ay,az=a_b[0],a_b[1],a_b[2]
    
    for i in range(3):
        c0 = -R[i,1]*az + R[i,2]*ay
        c1 =  R[i,0]*az - R[i,2]*ax
        c2 = -R[i,0]*ay + R[i,1]*ax
        F[3+i, 6] = c0*dt
        F[3+i, 7] = c1*dt
        F[3+i, 8] = c2*dt

    wx,wy,wz=w_b[0],w_b[1],w_b[2]
    F[6,6]=1.0;F[6,7]=wz*dt;F[6,8]=-wy*dt;F[7,6]=-wz*dt;F[7,7]=1.0;F[7,8]=wx*dt;F[8,6]=wy*dt;F[8,7]=-wx*dt;F[8,8]=1.0
    F[6,12]=-dt;F[7,13]=-dt;F[8,14]=-dt
    Qd[3,3]=Qd[4,4]=Qd[5,5]=na_var*dt;Qd[6,6]=Qd[7,7]=Qd[8,8]=ng_var*dt;Qd[9,9]=Qd[10,10]=Qd[11,11]=nba_var*dt;Qd[12,12]=Qd[13,13]=Qd[14,14]=nbg_var*dt

@njit(cache=True, fastmath=True)
def _propagate_state_jit(p,v,R,ba,bg,accel_raw,gyro_raw,dt,gravity_world,F,Qd,P,dR,dR_half,R_mid,a_w_mid,a_b,w_b,w_dt,na_var,ng_var,nba_var,nbg_var,step_count,reortho_interval,tmp15):
    a_b[0]=accel_raw[0]-ba[0];a_b[1]=accel_raw[1]-ba[1];a_b[2]=accel_raw[2]-ba[2];w_b[0]=gyro_raw[0]-bg[0];w_b[1]=gyro_raw[1]-bg[1];w_b[2]=gyro_raw[2]-bg[2]
    w_dt[0]=w_b[0]*dt*0.5;w_dt[1]=w_b[1]*dt*0.5;w_dt[2]=w_b[2]*dt*0.5;_rodrigues_jit(w_dt[0],w_dt[1],w_dt[2],dR_half);_mat3_mul(R,dR_half,R_mid);_mat3_vec(R_mid,a_b,a_w_mid)
    a_w_mid[0]-=gravity_world[0];a_w_mid[1]-=gravity_world[1];a_w_mid[2]-=gravity_world[2];dt2h=0.5*dt*dt
    p[0]+=v[0]*dt+a_w_mid[0]*dt2h;p[1]+=v[1]*dt+a_w_mid[1]*dt2h;p[2]+=v[2]*dt+a_w_mid[2]*dt2h;v[0]+=a_w_mid[0]*dt;v[1]+=a_w_mid[1]*dt;v[2]+=a_w_mid[2]*dt
    w_dt[0]=w_b[0]*dt;w_dt[1]=w_b[1]*dt;w_dt[2]=w_b[2]*dt;_rodrigues_jit(w_dt[0],w_dt[1],w_dt[2],dR);_mat3_mul(R,dR,dR_half)
    for i in range(3):
        for j in range(3): R[i,j]=dR_half[i,j]
    step_count+=1
    if step_count%reortho_interval==0:
        n0=math.sqrt(R[0,0]**2+R[0,1]**2+R[0,2]**2)
        if n0>1e-12: R[0,0]/=n0;R[0,1]/=n0;R[0,2]/=n0
        d=R[1,0]*R[0,0]+R[1,1]*R[0,1]+R[1,2]*R[0,2];R[1,0]-=d*R[0,0];R[1,1]-=d*R[0,1];R[1,2]-=d*R[0,2]
        n1=math.sqrt(R[1,0]**2+R[1,1]**2+R[1,2]**2)
        if n1>1e-12: R[1,0]/=n1;R[1,1]/=n1;R[1,2]/=n1
        R[2,0]=R[0,1]*R[1,2]-R[0,2]*R[1,1];R[2,1]=R[0,2]*R[1,0]-R[0,0]*R[1,2];R[2,2]=R[0,0]*R[1,1]-R[0,1]*R[1,0]
    _build_F_and_Qd_jit(F,Qd,R,a_b,w_b,dt,na_var,ng_var,nba_var,nbg_var);_triple_product_15(F,P,Qd,P,tmp15);_symmetrise_15(P)
    return step_count

@njit(cache=True, fastmath=True)
def _batch_depth_lookup_jit(depth_map,xs,ys,r,h,w,min_mm,max_mm):
    n=len(xs);result=np.zeros(n,dtype=np.float64)
    for idx in range(n):
        xi=int(round(xs[idx]));yi=int(round(ys[idx]))
        if xi<r: xi=r
        if xi>=w-r: xi=w-r-1
        if yi<r: yi=r
        if yi>=h-r: yi=h-r-1
        vc=0;ps=(2*r+1)*(2*r+1);vals=np.empty(ps,dtype=np.float64)
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                d=float(depth_map[yi+dy,xi+dx])
                if d>=min_mm and d<=max_mm: vals[vc]=d;vc+=1
        if vc>0:
            for i in range(vc):
                for j in range(i+1,vc):
                    if vals[j]<vals[i]: vals[i],vals[j]=vals[j],vals[i]
            result[idx]=vals[vc//2]
    return result

@njit(cache=True, fastmath=True)
def _mat_to_rotvec_jit(R):
    val=(R[0,0]+R[1,1]+R[2,2]-1.0)*0.5
    if val>1.0: val=1.0
    if val<-1.0: val=-1.0
    theta=math.acos(val);out=np.zeros(3)
    if theta<1e-9: return out
    k=theta/(2.0*math.sin(theta))
    out[0]=(R[2,1]-R[1,2])*k;out[1]=(R[0,2]-R[2,0])*k;out[2]=(R[1,0]-R[0,1])*k;return out

# ============================================================
# STATE ESTIMATOR (15-DOF EKF + MSCKF & Pre-Integration)
# ============================================================

class VIO_EKF:
    def __init__(self):
        self._lock = threading.Lock()
        
        # 15-DOF State
        self.p=np.zeros(3); self.v=np.zeros(3)
        self.R=np.eye(3); self.ba=np.zeros(3); self.bg=np.zeros(3)
        self.P=np.diag([1e-6]*3+[1e-4]*3+[1e-6]*3+[1e-4]*3+[1e-4]*3).astype(np.float64)
        
        self._R_vis=np.diag([VIS_NOISE_P]*3+[VIS_NOISE_PHI]*3).astype(np.float64)
        self._I15=np.eye(15); self._I3=np.eye(3)
        self._F=np.zeros((15,15)); self._Qd=np.zeros((15,15))
        self._dR=np.eye(3); self._dR_half=np.eye(3); self._R_mid=np.eye(3)
        self._a_w_mid=np.zeros(3); self._a_b=np.zeros(3)
        self._w_b=np.zeros(3); self._w_dt=np.zeros(3)
        self._tmp33=np.zeros((3,3)); self._P_tmp=np.zeros((15,15))
        self._tmp15x15 = np.empty((15, 15), dtype=np.float64)
        
        self.gravity_world=None; self.gravity_ready=False
        self._var_tracker=RunningVariance(STATIC_WIN)
        self._still_accels=[]; self.last_imu_ts=None
        self._kf_p=np.zeros(3); self._kf_R=np.eye(3); self._kf_set=False
        self._step_count=0
        
        # Safety Nets
        self._starvation_ticks = 0 
        self._last_v_p = None
        self._last_v_R = None
        self.residual_log = [] 
        
        # --- MSCKF / Manifold Pre-Integration Buffers ---
        self.window = []
        self.tracks = {}
        self.pre_dp = np.zeros(3, dtype=np.float64)
        self.pre_dv = np.zeros(3, dtype=np.float64)
        self.pre_dR = np.eye(3, dtype=np.float64)
        self.pre_dt = 0.0
        self._last_vis_cam_pose = None
        self._last_vis_ts = None
        self._vis_noise_scale = 1.0
        self._vis_reject_streak = 0
        self._vis_accept_count = 0
        self._vis_reject_count = 0
        self._vis_nis_ema = 0.0
        self._vis_last_nis = 0.0

    def feed_imu(self, a, g, ts):
        with self._lock: 
            a_clipped = np.clip(a, -25.0, 25.0)
            g_clipped = np.clip(g, -5.0, 5.0)
            
            # Manifold Pre-Integration relative to last keyframe
            if self.last_imu_ts is not None and self.gravity_ready:
                dt = ts - self.last_imu_ts
                if 0 < dt < 0.1:
                    unbiased_g = g_clipped - self.bg
                    unbiased_a = a_clipped - self.ba
                    _rodrigues_jit(unbiased_g[0]*dt, unbiased_g[1]*dt, unbiased_g[2]*dt, self._tmp33)
                    
                    self.pre_dp += self.pre_dv * dt + 0.5 * (self.pre_dR @ unbiased_a) * dt**2
                    self.pre_dv += (self.pre_dR @ unbiased_a) * dt
                    
                    _mat3_mul(self.pre_dR, self._tmp33, self._dR)
                    self.pre_dR[:] = self._dR
                    self.pre_dt += dt

            # Global 15-DOF State Propagation
            self._propagate(a_clipped, g_clipped, ts)
            
            if np.isnan(self.p).any() or np.isnan(self.R).any():
                print("  [CRITICAL] NaN detected in IMU Propagation! Reverting state.")
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                    self.v[:] = np.zeros(3)

    def _propagate(self, accel_raw, gyro_raw, ts):
        # Your custom gravity and 15-DOF integration preserved exactly.
        norm=math.sqrt(accel_raw[0]**2+accel_raw[1]**2+accel_raw[2]**2)
        self._var_tracker.push(norm)
        
        if not self.gravity_ready:
            is_s = self._var_tracker.is_full() and self._var_tracker.variance() < STATIC_VAR_THR
            if is_s:
                self._still_accels.append(accel_raw.copy())
                if len(self._still_accels) >= MIN_GRAV_SAMPLES:
                    samples = np.array(self._still_accels)
                    norms = np.linalg.norm(samples, axis=1)
                    median_norm = float(np.median(norms))
                    
                    inlier_mask = np.abs(norms - median_norm) < 0.05 * median_norm
                    inliers = samples[inlier_mask]
                    
                    if len(inliers) >= MIN_GRAV_SAMPLES // 2:
                        gb = np.mean(inliers, axis=0)
                        gm = float(np.linalg.norm(gb))
                        if 9.5 <= gm <= 10.5:
                            gu = gb / gm
                            zd = np.array([0., 0., -1.])
                            v = np.cross(gu, zd)
                            s = np.linalg.norm(v)
                            c = np.dot(gu, zd)
                            if s < 1e-8: 
                                Ra = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
                            else:
                                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                                Ra = np.eye(3) + vx + vx@vx * ((1. - c) / (s * s))
                            self.R[:] = Ra
                            self.gravity_world = np.array([0., 0., -gm])
                            self.gravity_ready = True
                            print(f"\n  [EKF] Gravity: ‖g‖={gm:.4f} m/s^2 ({len(inliers)}/{len(samples)} inliers)")
                        else:
                            print(f"  [WARN] Gravity magnitude {gm:.2f} m/s^2 unrealistic. Recalibrating...")
                            self._still_accels = []
                    else:
                        print(f"  [WARN] Too many gravity outliers. Recollecting...")
                        self._still_accels = []
            else:
                self._still_accels = []
            self.last_imu_ts = ts
            return
            
        if self.last_imu_ts is None: 
            self.last_imu_ts = ts
            return
            
        dt = ts - self.last_imu_ts
        self.last_imu_ts = ts
        
        if dt <= 0 or dt > 0.1: 
            return
            
        self._step_count = _propagate_state_jit(
            self.p, self.v, self.R, self.ba, self.bg, accel_raw, gyro_raw, dt,
            self.gravity_world, self._F, self._Qd, self.P,
            self._dR, self._dR_half, self._R_mid, self._a_w_mid,
            self._a_b, self._w_b, self._w_dt,
            ACCEL_ND**2, GYRO_ND**2, ACCEL_BRW**2, GYRO_BRW**2,
            self._step_count, REORTHO_INTERVAL, self._tmp15x15
        )

    def set_keyframe(self):
        with self._lock: 
            self._kf_p[:] = self.p
            self._kf_R[:] = self.R
            self._kf_set = True
        
    def get_angular_velocity(self):
        with self._lock: 
            return self._w_b.copy()

    def get_visual_health(self):
        with self._lock:
            if self._vis_reject_streak >= 10:
                mode = "VISION_DEGRADED"
            elif self._vis_noise_scale > 2.0:
                mode = "VISION_ADAPTIVE"
            else:
                mode = "NOMINAL"
            return {
                "mode": mode,
                "noise_scale": float(self._vis_noise_scale),
                "reject_streak": int(self._vis_reject_streak),
                "accept_count": int(self._vis_accept_count),
                "reject_count": int(self._vis_reject_count),
                "nis_ema": float(self._vis_nis_ema),
                "nis_last": float(self._vis_last_nis),
            }

    def update_visual(self, prev_pts, curr_pts, prev_depth, K, T_ic, T_ci, frame_ts=None):
        """
        Performs a tightly-coupled visual correction using depth-backed 3D-2D PnP.
        """
        with self._lock:
            if not self.gravity_ready:
                return False, 0

            if prev_pts is None or curr_pts is None or prev_depth is None:
                return False, 0
            if len(prev_pts) < MIN_FEAT_UPDATE or len(curr_pts) < MIN_FEAT_UPDATE:
                return False, 0

            if self._last_vis_ts is not None and frame_ts is not None:
                dt = float(frame_ts - self._last_vis_ts)
                if dt <= 0.0 or dt > 0.5:
                    self._last_vis_cam_pose = None

            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            # Feature tracks come from ISP resolution, depth may be downscaled.
            # Scale feature coordinates into the depth map before depth lookup.
            depth_h, depth_w = prev_depth.shape[:2]
            track_w = max(1.0, 2.0 * cx)
            track_h = max(1.0, 2.0 * cy)
            scale_x = depth_w / track_w
            scale_y = depth_h / track_h

            xs = (prev_pts[:, 0] * scale_x).astype(np.float64, copy=False)
            ys = (prev_pts[:, 1] * scale_y).astype(np.float64, copy=False)
            z_mm = _batch_depth_lookup_jit(
                prev_depth,
                xs,
                ys,
                DEPTH_PATCH_R,
                depth_h,
                depth_w,
                DEPTH_MIN_MM,
                DEPTH_MAX_MM,
            )
            valid = z_mm > 0.0
            if int(np.count_nonzero(valid)) < MIN_FEAT_UPDATE:
                return False, int(np.count_nonzero(valid))

            z = (z_mm[valid] * 1e-3).astype(np.float64, copy=False)
            up = prev_pts[valid, 0].astype(np.float64, copy=False)
            vp = prev_pts[valid, 1].astype(np.float64, copy=False)
            uc = curr_pts[valid, 0].astype(np.float64, copy=False)
            vc = curr_pts[valid, 1].astype(np.float64, copy=False)

            X = (up - cx) * z / fx
            Y = (vp - cy) * z / fy
            obj_pts = np.stack([X, Y, z], axis=1).astype(np.float32)
            img_pts = np.stack([uc, vc], axis=1).astype(np.float32)

            if obj_pts.shape[0] < 6:
                return False, int(obj_pts.shape[0])

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts,
                img_pts,
                K.astype(np.float64),
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=100,
                reprojectionError=3.0,
                confidence=0.99,
            )
            if not ok or inliers is None or len(inliers) < 6:
                return False, 0

            inl = inliers.flatten()
            obj_inl = obj_pts[inl]
            img_inl = img_pts[inl]

            ok_refine, rvec, tvec = cv2.solvePnP(
                obj_inl,
                img_inl,
                K.astype(np.float64),
                None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok_refine:
                return False, int(len(inl))

            reproj, _ = cv2.projectPoints(obj_inl, rvec, tvec, K.astype(np.float64), None)
            reproj_err = img_inl - reproj.reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.sum(reproj_err * reproj_err, axis=1))))

            R_pc, _ = cv2.Rodrigues(rvec)
            t_pc = tvec.reshape(3).astype(np.float64)

            if self._last_vis_cam_pose is None:
                R_wc_init = self.R @ T_ic[:3, :3]
                p_wc_init = self.p + self.R @ T_ic[:3, 3]
                self._last_vis_cam_pose = (R_wc_init.copy(), p_wc_init.copy())
                self._last_vis_ts = frame_ts
                return False, int(len(inl))

            R_wc_prev, p_wc_prev = self._last_vis_cam_pose
            R_wc_meas = R_wc_prev @ R_pc.T
            p_wc_meas = p_wc_prev - (R_wc_meas @ t_pc)

            # Convert measured world camera pose to measured world IMU pose.
            R_wi_meas = R_wc_meas @ T_ci[:3, :3]
            p_wi_meas = p_wc_meas + (R_wc_meas @ T_ci[:3, 3])

            r_p = p_wi_meas - self.p
            r_th = _mat_to_rotvec_jit(self.R.T @ R_wi_meas)
            if not np.isfinite(r_p).all() or not np.isfinite(r_th).all():
                return False, int(len(inl))

            if np.linalg.norm(r_p) > 2.0 or np.linalg.norm(r_th) > np.deg2rad(40.0):
                self._last_vis_cam_pose = None
                self._last_vis_ts = frame_ts
                return False, int(len(inl))

            inlier_ratio = float(len(inl)) / float(obj_pts.shape[0])
            sigma_p = float(np.clip(0.02 + 0.10 * rmse + (1.0 - inlier_ratio) * 0.08, 0.01, 0.30))
            sigma_r = float(np.clip(np.deg2rad(0.8 + 8.0 * rmse + (1.0 - inlier_ratio) * 8.0),
                                    np.deg2rad(0.5), np.deg2rad(20.0)))

            Rm = np.diag([sigma_p * sigma_p] * 3 + [sigma_r * sigma_r] * 3).astype(np.float64)
            Rm *= self._vis_noise_scale
            H = np.zeros((6, 15), dtype=np.float64)
            H[:3, :3] = np.eye(3)
            H[3:, 6:9] = np.eye(3)

            r = np.hstack([r_p, r_th]).astype(np.float64)
            S = H @ self.P @ H.T + Rm
            try:
                S_inv_r = np.linalg.solve(S, r)
                nis = float(r @ S_inv_r)
                self._vis_last_nis = nis
                self._vis_nis_ema = 0.9 * self._vis_nis_ema + 0.1 * nis
                # Adaptive visual noise + consistency gating.
                if nis > VIS_NIS_CHI2_99 * max(1.0, self._vis_noise_scale):
                    self._vis_reject_streak += 1
                    self._vis_reject_count += 1
                    self._vis_noise_scale = min(12.0, self._vis_noise_scale * 1.25)
                    self.residual_log.append({
                        "tick": self._step_count,
                        "type": "visual_reject_nis",
                        "nis": nis,
                        "noise_scale": float(self._vis_noise_scale),
                        "inliers": int(len(inl)),
                    })
                    return False, int(len(inl))

                if nis > VIS_NIS_CHI2_95:
                    self._vis_noise_scale = min(12.0, self._vis_noise_scale * 1.10)
                else:
                    self._vis_noise_scale = max(1.0, self._vis_noise_scale * 0.985)

                K_gain = self.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return False, int(len(inl))

            dx = K_gain @ r
            self.p += dx[0:3]
            self.v += dx[3:6]
            dth = dx[6:9]
            dR, _ = cv2.Rodrigues(dth.astype(np.float64))
            self.R = self.R @ dR
            self.ba += dx[9:12]
            self.bg += dx[12:15]

            I = np.eye(15, dtype=np.float64)
            KH = K_gain @ H
            self.P = (I - KH) @ self.P @ (I - KH).T + K_gain @ Rm @ K_gain.T
            self.P = 0.5 * (self.P + self.P.T)

            if np.isnan(self.p).any() or np.isnan(self.R).any() or np.isnan(self.P).any():
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                self.P[:] = np.eye(15, dtype=np.float64) * 1e-3
                return False, int(len(inl))

            self._last_v_p = self.p.copy()
            self._last_v_R = self.R.copy()
            self._vis_reject_streak = 0
            self._vis_accept_count += 1
            R_wc_corr = self.R @ T_ic[:3, :3]
            p_wc_corr = self.p + self.R @ T_ic[:3, 3]
            self._last_vis_cam_pose = (R_wc_corr.copy(), p_wc_corr.copy())
            self._last_vis_ts = frame_ts

            self.residual_log.append({
                "tick": self._step_count,
                "type": "visual_pnp",
                "inliers": int(len(inl)),
                "tracks": int(obj_pts.shape[0]),
                "inlier_ratio": inlier_ratio,
                "reproj_rmse_px": rmse,
                "pos_residual_m": float(np.linalg.norm(r_p)),
                "rot_residual_deg": float(np.rad2deg(np.linalg.norm(r_th))),
                "nis": float(self._vis_last_nis),
                "noise_scale": float(self._vis_noise_scale),
            })
            return True, int(len(inl))

    def add_visual_tracks(self, frame_id, features_dict, K, T_ic):
        """
        Replaces solvePnPRansac with Tightly-Coupled MSCKF feature tracking.
        Appends to the sliding window, triangulates lost features, and executes Null-Space projection.
        """
        with self._lock:
            if not self.gravity_ready: return False, 0
            
            # --- 1. STATE AUGMENTATION (MSCKF Window) ---
            R_cam = self.R @ T_ic[:3, :3]
            p_cam = self.p + self.R @ T_ic[:3, 3]
            
            self.window.append({
                "id": frame_id,
                "p": p_cam.copy(),
                "R": R_cam.copy()
            })
            
            if len(self.window) > MSCKF_WINDOW:
                self.window.pop(0) 
                
            # Reset Manifold pre-integration
            self.pre_dp[:] = 0.0; self.pre_dv[:] = 0.0
            self.pre_dR[:] = np.eye(3); self.pre_dt = 0.0
            
            # --- 2. TRACK MANAGEMENT ---
            current_ids = set(features_dict.keys())
            for fid, (u, v) in features_dict.items():
                if fid not in self.tracks:
                    self.tracks[fid] = {"obs": []}
                self.tracks[fid]["obs"].append((frame_id, u, v))
                
            # --- 3. MSCKF NULLSPACE UPDATE ---
            lost_ids = [fid for fid in self.tracks.keys() if fid not in current_ids]
            
            total_r_o_norm = 0.0
            processed_tracks = 0
            
            for fid in lost_ids:
                track = self.tracks[fid]
                if len(track["obs"]) >= MIN_TRACK:
                    r_o = self._process_msckf_feature(track, K)
                    if r_o is not None:
                        total_r_o_norm += np.linalg.norm(r_o)
                        processed_tracks += 1
                del self.tracks[fid]
                
            # --- 4. SAFETY NETS & LOGGING ---
            if processed_tracks > 0:
                self._starvation_ticks = 0
                self._last_v_p = self.p.copy()
                self._last_v_R = self.R.copy()
                self.residual_log.append({
                    "tick": self._step_count, 
                    "type": "msckf_nullspace",
                    "avg_error_norm": float(total_r_o_norm / processed_tracks)
                })
            else:
                self._starvation_ticks += 1
                
            if np.isnan(self.p).any() or np.isnan(self.R).any() or np.isnan(self.P).any():
                print("  [CRITICAL] NaN detected in Visual Update! Reverting state.")
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                self.P[:] = np.eye(15) * 1e-3
                return False, processed_tracks

            return True, processed_tracks

    def _process_msckf_feature(self, track, K):
        obs = track["obs"]
        first_obs, last_obs = obs[0], obs[-1]
        
        pose1 = next((p for p in self.window if p["id"] == first_obs[0]), None)
        pose2 = next((p for p in self.window if p["id"] == last_obs[0]), None)
        
        if not pose1 or not pose2: return None
        
        # Build Projection Matrices
        P1 = K @ np.hstack((pose1["R"].T, -pose1["R"].T @ pose1["p"].reshape(3,1)))
        P2 = K @ np.hstack((pose2["R"].T, -pose2["R"].T @ pose2["p"].reshape(3,1)))
        
        pt1 = np.array([[first_obs[1]], [first_obs[2]]], dtype=np.float32)
        pt2 = np.array([[last_obs[1]], [last_obs[2]]], dtype=np.float32)
        
        # Triangulate historical 3D Point
        p4d = cv2.triangulatePoints(P1.astype(np.float32), P2.astype(np.float32), pt1, pt2)
        p3d_w = (p4d[:3] / (p4d[3] + 1e-6)).flatten()
        
        r_stack, Hf_stack = [], []
        
        for frame_id, u, v in obs:
            c_pose = next((p for p in self.window if p["id"] == frame_id), None)
            if not c_pose: continue
            
            p_c = c_pose["R"].T @ (p3d_w - c_pose["p"])
            pc_x, pc_y, z_depth = p_c[0], p_c[1], p_c[2]
            
            if z_depth < 0.1: continue # Reject points behind camera
            
            u_hat = K[0,0] * (pc_x / z_depth) + K[0,2]
            v_hat = K[1,1] * (pc_y / z_depth) + K[1,2]
            
            r_stack.append([u - u_hat, v - v_hat])
            
            dz_dp = np.array([
                [1/z_depth, 0, -pc_x/(z_depth**2)],
                [0, 1/z_depth, -pc_y/(z_depth**2)]
            ])
            Hf = dz_dp @ c_pose["R"].T
            Hf_stack.append(Hf)
            
        if len(r_stack) < MIN_TRACK: return None
        
        r_vec = np.vstack(r_stack).flatten()
        Hf_mat = np.vstack(Hf_stack)
        
        # Left Null-Space Projection (Hardware Accelerated)
        r_o, A = project_nullspace(Hf_mat, r_vec)
        return r_o

    def get_pose(self):
        with self._lock:
            T = np.eye(4)
            T[:3,:3] = self.R.copy()
            T[:3,3]  = self.p.copy()
            
            idx = [0,1,2,6,7,8]
            c6 = self.P[np.ix_(idx,idx)].copy()
        return T, c6

    def is_ready(self):
        with self._lock: return self.gravity_ready