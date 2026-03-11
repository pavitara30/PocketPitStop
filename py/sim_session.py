#!/usr/bin/env python3
"""
sim_session.py - fake a Sensor Logger kart recording with known ground truth, to test
kart_telemetry.py before any track time.

  python sim_session.py fake.zip [--seed 1] [--ios] [--total] [--gps-lag 0.4]

--ios   : flip accel sign (gravity reads 'down')    --total: write TotalAcceleration.csv (g)
default : Accelerometer.csv + Gravity.csv in m/s²
Also writes <name>_ir_laps.csv (true lap times + 5 ms noise) and <name>_truth.json.
"""
import argparse, json, zipfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

G, FS, R_E = 9.80665, 100.0, 6371000.0
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--ios", action="store_true"); ap.add_argument("--total", action="store_true")
ap.add_argument("--gps-lag", type=float, default=0.4); ap.add_argument("--laps", type=int, default=8)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# --- track: closed loop ~680 m with fast sweepers and tight corners
th = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
r = 1 + 0.28 * np.cos(3 * th) + 0.12 * np.sin(2 * th)
x, y = 120 * r * np.cos(th), 60 * r * np.sin(th)
seg = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
s_raw = np.r_[0, np.cumsum(seg)][:-1]; L = s_raw[-1] + seg[-1]
ds = 0.5; S = np.arange(0, L, ds)
X, Y = np.interp(S, s_raw, x, period=L), np.interp(S, s_raw, y, period=L)
psi = np.unwrap(np.arctan2(np.gradient(np.r_[Y, Y[:2]])[:len(S)], np.gradient(np.r_[X, X[:2]])[:len(S)]))
kappa = np.gradient(psi, ds)
kappa = np.convolve(np.r_[kappa[-20:], kappa, kappa[:20]], np.ones(21) / 21, "same")[20:-20]
Z = 1.5 * np.sin(2 * np.pi * S / L)           # gentle hill

# --- flying-lap speed profile (lateral limit, then accel/brake limits, periodic)
a_lat, a_brk, v_top = 1.3 * G, 1.0 * G, 22.0
v_lim = np.minimum(v_top, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)))
def limit(vcap, v0=None, vend=None):
    v = vcap.copy()
    if v0 is not None: v[0] = v0
    for i in range(1, len(v)):           # forward: engine-limited accel
        acc = 0.5 * G * max(1 - v[i - 1] / 25.0, 0.05)
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * acc * ds))
    if vend is not None: v[-1] = vend
    for i in range(len(v) - 2, -1, -1):  # backward: braking limit
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_brk * ds))
    return v

# session: start 60 m before the start/finish line, N laps, stop after
k_sf = int(np.argmax(v_lim * (np.abs(kappa) < 0.004)))     # S/F somewhere on a straight
s_sf = S[k_sf]
start = (s_sf - 60) % L
N = int(((args.laps + 0.4) * L) / ds)
Ssess = start + np.arange(N) * ds
idx = ((Ssess % L) / ds).astype(int) % len(S)
lapno = np.floor((Ssess - s_sf) / L).astype(int)
scale = 0.96 + 0.04 * rng.random(args.laps + 3)             # driver inconsistency per lap
vcap = v_lim[idx] * scale[np.clip(lapno + 1, 0, len(scale) - 1)]
v_s = limit(vcap, v0=0.0, vend=0.0)
dt_nodes = ds / np.maximum(0.5 * (v_s[1:] + v_s[:-1]), 1e-3)
T_nodes = np.r_[0, np.cumsum(dt_nodes)]
PARK0, PARK1 = 12.0, 6.0
T_nodes += PARK0
T_end = T_nodes[-1] + PARK1

t = np.arange(0, T_end, 1 / FS)
Sd = np.interp(t, T_nodes, Ssess - start)                # distance travelled
v = np.interp(t, T_nodes, v_s, left=0, right=0)
a_long = np.convolve(np.gradient(v, 1 / FS), np.ones(9) / 9, "same")
pos_idx = ((start + Sd) % L)
e_true, n_true = np.interp(pos_idx, S, X, period=L), np.interp(pos_idx, S, Y, period=L)
# heading: integrate curvature along distance (simple & continuous)
psi_s = np.cumsum(np.interp(pos_idx, S, kappa, period=L) * np.gradient(Sd))
psi_s += np.interp(pos_idx[0], S, psi)
yaw_rate = np.gradient(psi_s, 1 / FS)
a_lat_t = v * yaw_rate
grade = np.interp(pos_idx, S, np.gradient(Z, ds), period=L)

# true S/F crossings (lap times)
k = np.arange(0, args.laps + 2)
targets = (s_sf - start) % L + k * L
targets = targets[targets < Sd[-1]]
t_cross = np.interp(targets, Sd, t)
true_laps = np.diff(t_cross)

# --- body attitude + specific force (kart axes x fwd, y left, z up)
roll = np.radians(2.0) * a_lat_t / G           # chassis leans out of the corner
pitch = np.radians(0.8) * (-a_long / G) - np.arctan(grade)
f_grav = np.column_stack([-np.sin(pitch), np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch)])
f_kin = np.column_stack([a_long / G, a_lat_t / G, np.zeros_like(t)])
w_body = np.column_stack([np.gradient(roll, 1 / FS), np.gradient(pitch, 1 / FS), yaw_rate])

# vibration: engine order rises with speed + bumps
f_eng = 24 + 1.0 * v
phase = 2 * np.pi * np.cumsum(f_eng) / FS
amp = 0.25 + 0.03 * v
vib = np.column_stack([0.3 * amp * np.sin(phase + 1), 0.4 * amp * np.sin(phase + 2), amp * np.sin(phase)])
vib += rng.normal(0, 0.08, (len(t), 3)) * np.c_[0.5, 0.5, 1] * (0.3 + v[:, None] / 22)

M = Rotation.random(random_state=args.seed).as_matrix()     # phone <- kart (random mount)
sign = -1 if args.ios else 1
acc_user = sign * (f_kin + vib) @ M.T + rng.normal(0, 0.01, (len(t), 3)) + rng.normal(0, 0.02, 3)
acc_grav = sign * f_grav @ M.T
gyro = w_body @ M.T + rng.normal(0, 0.005, (len(t), 3)) + rng.normal(0, 0.008, 3) \
       + 0.05 * np.sin(phase)[:, None] * rng.normal(0, 1, 3)

# --- timestamps (ns epoch), jitter
T0 = 1764000000_000_000_000
def ts(tt): return (T0 + np.round(tt * 1e9)).astype(np.int64)
t_acc = t + rng.normal(0, 0.0008, len(t)); t_acc.sort()
t_gyr = t + 0.003 + rng.normal(0, 0.0008, len(t)); t_gyr.sort()

def imu_csv(tt, V, scale=1.0):
    return pd.DataFrame({"time": ts(tt), "seconds_elapsed": tt, "z": V[:, 2] * scale, "y": V[:, 1] * scale, "x": V[:, 0] * scale})

files = {}
if args.total:
    files["TotalAcceleration.csv"] = imu_csv(t_acc, acc_user + acc_grav)        # in g
else:
    files["Accelerometer.csv"] = imu_csv(t_acc, acc_user, G)                    # m/s²
    files["Gravity.csv"] = imu_csv(t_acc, acc_grav, G)
files["Gyroscope.csv"] = imu_csv(t_gyr, gyro)

# --- GPS 1 Hz, lagged, with correlated position error
tg = np.arange(rng.random(), T_end, 1.0)
src = tg - args.gps_lag
ar = np.zeros((len(tg), 2))
for i in range(1, len(tg)): ar[i] = 0.82 * ar[i - 1] + rng.normal(0, 0.7, 2)
eg = np.interp(src, t, e_true) + ar[:, 0] + rng.normal(0, 0.3, len(tg))
ng = np.interp(src, t, n_true) + ar[:, 1] + rng.normal(0, 0.3, len(tg))
vg = np.maximum(np.interp(src, t, v) + rng.normal(0, 0.15, len(tg)), 0)
lat0, lon0 = 12.9, 80.1
lat = lat0 + np.degrees(ng / R_E); lon = lon0 + np.degrees(eg / (R_E * np.cos(np.radians(lat0))))
hacc = np.r_[np.full(3, 35.0), rng.uniform(3, 6, len(tg) - 3)]       # first fixes poor
files["Location.csv"] = pd.DataFrame({
    "time": ts(tg), "seconds_elapsed": tg, "bearingAccuracy": 10.0, "speedAccuracy": rng.uniform(0.3, 0.6, len(tg)),
    "verticalAccuracy": 5.0, "horizontalAccuracy": hacc, "speed": np.where(vg > 0.3, vg, -1),
    "bearing": (90 - np.degrees(np.interp(src, t, psi_s))) % 360, "altitude": 50 + np.interp(src, t, np.interp(pos_idx, S, Z, period=L)),
    "longitude": lon, "latitude": lat})

with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, df in files.items(): z.writestr(name, df.to_csv(index=False))

base = args.out.rsplit(".", 1)[0]
pd.DataFrame({"lap": np.arange(1, len(true_laps) + 1), "time_s": np.round(true_laps + rng.normal(0, 0.005, len(true_laps)), 3)}
             ).to_csv(base + "_ir_laps.csv", index=False)
# true S/F line (+-10 m perpendicular at s_sf)
d = np.array([np.cos(psi[k_sf]), np.sin(psi[k_sf])]); nrm = np.array([-d[1], d[0]]) * 10
pts = [np.array([X[k_sf], Y[k_sf]]) - nrm, np.array([X[k_sf], Y[k_sf]]) + nrm]
line = []
for p in pts: line += [lat0 + np.degrees(p[1] / R_E), lon0 + np.degrees(p[0] / (R_E * np.cos(np.radians(lat0))))]
mv = v > 3
truth = dict(track_m=round(L, 1), true_laps=np.round(true_laps, 3).tolist(), M_phone_from_kart=M.round(4).tolist(),
             sign=sign, gps_lag=args.gps_lag, line=",".join(f"{q:.7f}" for q in line),
             max_lat_g=round(float(np.abs(a_lat_t[mv]).max() / G), 2), max_brake_g=round(float(-a_long.min() / G), 2),
             top_kmh=round(float(v.max() * 3.6), 1))
json.dump(truth, open(base + "_truth.json", "w"), indent=1)
np.savez(base + "_truth.npz", t=t, v=v, roll=np.degrees(roll), pitch=np.degrees(pitch), a_long=a_long / G, a_lat=a_lat_t / G)
print(json.dumps(truth, indent=1))
