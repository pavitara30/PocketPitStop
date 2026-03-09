#!/usr/bin/env python3
"""
kart_telemetry.py - turn a Sensor Logger recording into race-engineer telemetry.

  python kart_telemetry.py session.zip
  python kart_telemetry.py session.zip --line LAT1,LON1,LAT2,LON2 --ir ir_laps.csv

Record with Accelerometer, Gravity, Gyroscope and Location ON (highest sample rate).
Mount the phone rigidly, start recording, then sit still ~10 s before driving off.
Kart axes used everywhere: x forward, y left, z up (ISO 8855). Accel in g.
Outputs (in --out): dashboard.png, laps.csv, events.csv, telemetry.csv, summary.json
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

G, FS, R_EARTH = 9.80665, 100.0, 6371000.0     # m/s² per g, shared timeline (Hz), metres
TH = dict(brake_g=-0.30, brake_s=0.3,          # event thresholds - tune per kart/track
          corner_g=0.40, corner_s=0.5,
          flat_ax=-0.05, flat_ay=0.25, flat_s=1.0,   # flat-out: not slowing, not turning
          min_lap_s=15.0)


# ------------------------------------------------------------------ helpers
def xyz(d): return d[["x", "y", "z"]].to_numpy(float)
def secs(d, t0): return (d["time"].to_numpy(np.int64) - t0) / 1e9     # ns epoch -> s
def unit(v): return v / np.linalg.norm(v)
def interp(t_new, t, X): return np.column_stack([np.interp(t_new, t, c) for c in np.atleast_2d(X.T)])
def lowpass(X, fc, order=4): return sosfiltfilt(butter(order, fc, fs=FS, output="sos"), X, axis=0)


def to_en(lat, lon, lat0, lon0):          # lat/lon -> local east/north metres
    return (np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(np.radians(lat0)),
            np.radians(np.asarray(lat) - lat0) * R_EARTH)


def to_latlon(e, n, lat0, lon0):
    return lat0 + np.degrees(n / R_EARTH), lon0 + np.degrees(e / (R_EARTH * np.cos(np.radians(lat0))))


def runs(mask, min_s):                    # (start, end) of True stretches lasting >= min_s
    d = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(s, e) for s, e in zip(d[::2], d[1::2]) if (e - s) / FS >= min_s]


# ------------------------------------------------------------------ 1. load
def load(path):
    p, fr = Path(path), {}
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv") and "__MACOSX" not in n:
                    fr[Path(n).stem] = pd.read_csv(z.open(n))
    else:
        fr = {f.stem: pd.read_csv(f) for f in p.glob("*.csv")}
    return {k: d.sort_values("time").drop_duplicates("time") for k, d in fr.items() if "time" in d}


def total_accel(fr):
    """Accelerometer INCLUDING gravity, in g, phone axes (handles both Sensor Logger layouts)."""
    if "TotalAcceleration" in fr:
        d = fr["TotalAcceleration"]; a = xyz(d)
    elif "Accelerometer" in fr and "Gravity" in fr:       # user accel + gravity, split
        d, g = fr["Accelerometer"], fr["Gravity"]
        a = xyz(d) + interp(d["time"].to_numpy(float), g["time"].to_numpy(float), xyz(g))
    else:
        d = fr["Accelerometer"]; a = xyz(d)
    m = np.linalg.norm(a.mean(0))           # gravity survives averaging, vibration doesn't
    if m < 0.5:
        raise SystemExit("No gravity in the accel data - turn on Gravity in Sensor Logger.")
    return d, (a / G if m > 3 else a)       # m/s² -> g


def gps_track(fr, t0):
    d = fr.get("Location")
    if d is None: return None
    if "horizontalAccuracy" in d: d = d[(d.horizontalAccuracy > 0) & (d.horizontalAccuracy < 20)]
    if len(d) < 10: return None
    t = secs(d, t0)
    lat0, lon0 = d.latitude.mean(), d.longitude.mean()
    e, n = to_en(d.latitude.to_numpy(), d.longitude.to_numpy(), lat0, lon0)
    v_pos = np.hypot(np.gradient(e, t), np.gradient(n, t))
    v = d["speed"].to_numpy(float) if "speed" in d else v_pos.copy()
    v = np.where(v >= 0, v, v_pos)                         # iOS writes -1 for "no speed"
    sa = d["speedAccuracy"].to_numpy(float) if "speedAccuracy" in d else np.full(len(t), 0.5)
    return dict(t=t, e=e, n=n, v=v, sa=np.where(sa > 0, sa, 0.5), lat0=lat0, lon0=lon0,
                hdg=np.unwrap(np.arctan2(np.gradient(n), np.gradient(e))))


# ------------------------------------------------------------------ 2. align axes
def find_rest(t, a, gps):
    """Stillest 2 s while parked, within the 30 s before the kart first moves."""
    n = int(2 * FS)
    sd = pd.Series(np.linalg.norm(a, axis=1)).rolling(n).std().to_numpy()
    if gps is not None:
        go = gps["t"][gps["v"] > 3]
        t_go = go[0] if len(go) else t[-1]
        ok = (np.interp(t, gps["t"], gps["v"]) < 1.0) & (t < t_go) & (t > t_go - 30)
    else:
        ok = t < t[0] + 30
    sd_ok = np.where(ok, sd, np.nan)
    i = int(np.nanargmin(sd if np.all(np.isnan(sd_ok)) else sd_ok))
    return slice(max(i - n + 1, 0), i + 1)


def align(t, a, w, gps, rest):
    """R: rows = kart forward/left/up in phone axes, so a_kart = a_phone @ R.T.
    'down' comes from the parked reading; 'forward' from the horizontal direction whose
    accel best explains GPS dv/dt (or the first launch if there's no GPS). iOS and Android
    disagree on the accel sign, so we check it: gyro yaw must turn the same way as the GPS track."""
    for sign in (1, -1):
        sf = sign * a
        up = unit(sf[rest].mean(0))
        e1 = unit(np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0])); e2 = np.cross(up, e1)
        H = lowpass(sf, 0.5) @ np.column_stack([e1, e2])          # horizontal accel (2-D)
        if gps is not None:
            m = (gps["t"] > t[0]) & (gps["t"] < t[-1])
            dv = np.gradient(gps["v"], gps["t"]) / G
            c = np.linalg.lstsq(interp(gps["t"][m], t, H), dv[m], rcond=None)[0]
        else:
            go = np.flatnonzero((np.linalg.norm(H, axis=1) > 0.15) & (t > t[rest.stop - 1]))
            i = go[0] if len(go) else int(np.argmax(np.linalg.norm(H, axis=1)))
            c = H[i:i + int(FS)].mean(0)
        fwd = unit(c[0] * e1 + c[1] * e2)
        R = np.vstack([fwd, np.cross(up, fwd), up])
        if gps is None:
            print("! no GPS: accel sign convention assumed, left/right may be swapped")
            return R, sign
        m = (gps["v"] > 3) & (gps["t"] > t[0]) & (gps["t"] < t[-1])
        yaw = np.interp(gps["t"][m], t, lowpass(w, 0.5) @ up)
        if np.corrcoef(yaw, np.gradient(gps["hdg"], gps["t"])[m])[0, 1] > 0:
            return R, sign
    raise SystemExit("Gyro and GPS disagree on turn direction - is the phone mounted rigidly?")


# ------------------------------------------------------------------ 3. sync
def gps_lag(t, a_long, gps, max_lag=2.0):
    """Phone GPS fixes arrive late. Find the time shift that best lines up GPS dv/dt
    with the accelerometer (cross-correlation), so the two clocks agree."""
    m = (gps["t"] > t[0] + max_lag + 2) & (gps["t"] < t[-1] - max_lag - 2)
    dv = np.gradient(gps["v"], gps["t"])[m] / G
    n = int(2 * np.median(np.diff(gps["t"])) * FS)            # GPS dv/dt spans 2 fixes,
    a_box = np.convolve(a_long, np.ones(n) / n, "same")         # so average accel the same way
    lags = np.arange(-max_lag, max_lag + 1e-9, 0.02)
    score = [np.corrcoef(np.interp(gps["t"][m] - L, t, a_box), dv)[0, 1] for L in lags]
    return lags[int(np.argmax(score))]


# ------------------------------------------------------------------ 5. fuse
def complementary(A, W, v=None, tau=1.0):
    """Roll/pitch (deg), zero = parked attitude. Gyro for fast changes, gravity for the slow
    truth. Motion accel (dv/dt forward, speed x yaw rate sideways) is removed first so a corner
    or braking zone isn't read as tilt; what's left is gravity (low-passed: GPS corrections jump)."""
    dt, k = 1 / FS, tau / (tau + 1 / FS)
    g = A.copy()
    if v is not None:
        g[:, 0] -= np.gradient(v, dt) / G
        g[:, 1] -= v * W[:, 2] / G
        g[:, :2] = lowpass(g[:, :2], 0.5)
    roll_a = np.arctan2(g[:, 1], g[:, 2])
    pitch_a = np.arctan2(-g[:, 0], np.hypot(g[:, 1], g[:, 2]))
    roll, pitch = np.empty(len(A)), np.empty(len(A))
    roll[0], pitch[0] = roll_a[0], pitch_a[0]
    for i in range(1, len(A)):
        roll[i] = k * (roll[i - 1] + W[i, 0] * dt) + (1 - k) * roll_a[i]
        pitch[i] = k * (pitch[i - 1] + W[i, 1] * dt) + (1 - k) * pitch_a[i]
    return np.degrees(roll), np.degrees(pitch)


def kalman_speed(t, a_long, gps, q_acc=0.5, q_bias=0.02):
    """Kalman filter. State = [speed m/s, accel bias m/s²]. Predict at 100 Hz from the
    accelerometer (fast but drifts), correct at ~1 Hz with GPS speed (slow but no drift)."""
    dt = 1 / FS
    F = np.array([[1.0, -dt], [0.0, 1.0]]); B = np.array([dt, 0.0])
    Q = np.diag([(q_acc * dt) ** 2, q_bias ** 2 * dt])
    j = int(np.searchsorted(gps["t"], t[0]))
    x = np.array([gps["v"][min(j, len(gps["v"]) - 1)], 0.0]); P = np.diag([1.0, 0.1])
    out, bias = np.empty(len(t)), np.empty(len(t))
    for i in range(len(t)):
        x = F @ x + B * a_long[i] * G
        P = F @ P @ F.T + Q
        while j < len(gps["t"]) and gps["t"][j] <= t[i]:      # GPS fix arrived: correct
            S = P[0, 0] + gps["sa"][j] ** 2
            K = P[:, 0] / S
            x = x + K * (gps["v"][j] - x[0])
            P = P - np.outer(K, P[0])
            j += 1
        out[i], bias[i] = max(x[0], 0.0), x[1]
    return out, bias


# ------------------------------------------------------------------ 6. detect
def auto_line(gps, half=10.0):
    """Start/finish line across the track where the kart first nears top speed (main straight)."""
    k = int(np.flatnonzero(gps["v"] > 0.8 * np.percentile(gps["v"], 98))[0])
    k = min(max(k, 1), len(gps["t"]) - 2)
    p = np.array([gps["e"][k], gps["n"][k]])
    d = unit(np.array([gps["e"][k + 1] - gps["e"][k - 1], gps["n"][k + 1] - gps["n"][k - 1]]))
    return p - half * np.array([-d[1], d[0]]), p + half * np.array([-d[1], d[0]])


def crossings(gps, line, t, dist):
    """Times the GPS track crosses the line. Between two fixes, the exact moment is found by
    splitting the step using fused (100 Hz) distance rather than assuming constant speed."""
    a, b = line; s = b - a; P = np.column_stack([gps["e"], gps["n"]]); hits = []
    for k in range(len(P) - 1):
        r, q = P[k + 1] - P[k], a - P[k]
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) < 1e-9 or gps["v"][k] < 2: continue
        u = (q[0] * s[1] - q[1] * s[0]) / den          # fraction along the GPS step
        w = (q[0] * r[1] - q[1] * r[0]) / den          # fraction along the line
        if 0 <= u < 1 and 0 <= w <= 1: hits.append((k, u, np.sign(den)))
    main_dir, tc = np.sign(sum(h[2] for h in hits)), []
    for k, u, sg in hits:
        if sg != main_dir: continue
        t1, t2 = gps["t"][k], gps["t"][k + 1]
        if dist is not None and t[0] <= t1 and t2 <= t[-1]:
            d1, d2 = np.interp([t1, t2], t, dist)
            tx = float(np.interp(d1 + u * (d2 - d1), dist, t))
        else:
            tx = t1 + u * (t2 - t1)
        if not tc or tx - tc[-1] > TH["min_lap_s"]: tc.append(tx)
    return np.array(tc)


def lap_table(tc, t, A, v, dist):
    rows = []
    for i in range(len(tc) - 1):
        m = (t >= tc[i]) & (t < tc[i + 1])
        rows.append(dict(lap=i + 1, start_s=round(tc[i], 3), lap_s=round(tc[i + 1] - tc[i], 3),
                         top_kmh=v[m].max() * 3.6, min_kmh=v[m].min() * 3.6,
                         max_lat_g=np.abs(A[m, 1]).max(), max_brake_g=-A[m, 0].min(),
                         dist_m=dist[m][-1] - dist[m][0]))
    df = pd.DataFrame(rows)
    if len(df): df["gap_to_best_s"] = df.lap_s - df.lap_s.min()
    return df


def compare_ir(laps, path):
    """Line telemetry laps up with the IR lap counter's laps (best offset) and diff them."""
    ir = pd.to_numeric(pd.read_csv(path, header=None).iloc[:, -1], errors="coerce").dropna().to_numpy()
    tel, best = laps.lap_s.to_numpy(), (np.inf, 0)
    for k in range(-len(tel) + 1, len(ir)):
        j = np.arange(len(tel)) + k; m = (j >= 0) & (j < len(ir))
        if m.sum() >= min(3, len(tel)):
            best = min(best, (np.mean(np.abs(tel[m] - ir[j[m]])), k))
    j = np.arange(len(tel)) + best[1]; m = (j >= 0) & (j < len(ir))
    laps["ir_s"] = np.nan; laps.loc[m, "ir_s"] = ir[j[m]]
    laps["tel_minus_ir_s"] = laps.lap_s - laps.ir_s


def detect_events(t, A, v, tc):
    lap_of = lambda x: int(np.searchsorted(tc, x, side="right"))   # 0 = out-lap
    vv = v if v is not None else np.full(len(t), np.nan)
    ev = []
    def add(kind, mask, min_s, peak):
        for s, e in runs(mask, min_s):
            ev.append(dict(type=kind, lap=lap_of(t[s]), t_start=round(t[s], 2), t_end=round(t[e - 1], 2),
                           dur_s=(e - s) / FS, peak_g=peak[s:e][np.argmax(np.abs(peak[s:e]))],
                           v_in_kmh=vv[s] * 3.6, v_min_kmh=vv[s:e].min() * 3.6, v_out_kmh=vv[e - 1] * 3.6))
    add("brake", A[:, 0] < TH["brake_g"], TH["brake_s"], A[:, 0])
    add("corner_left", A[:, 1] > TH["corner_g"], TH["corner_s"], A[:, 1])
    add("corner_right", A[:, 1] < -TH["corner_g"], TH["corner_s"], A[:, 1])
    if v is not None:   # needs speed, or a parked kart counts as "flat-out"
        flat = (A[:, 0] > TH["flat_ax"]) & (np.abs(A[:, 1]) < TH["flat_ay"]) & (v > 5)
        add("flat_out", flat, TH["flat_s"], A[:, 0])
    return pd.DataFrame(ev).sort_values("t_start").reset_index(drop=True) if ev else pd.DataFrame()


# ------------------------------------------------------------------ 7. visualise
def dashboard(path, c):
    t, A, v, laps, tc, ev = c["t"], c["A"], c["v"], c["laps"], c["tc"], c["events"]
    fig, ax = plt.subplots(5, 2, figsize=(15, 23)); ax = ax.ravel()
    have_laps = len(laps) > 0
    if have_laps:
        b = laps.loc[laps.lap_s.idxmin()]
        bm = (t >= b.start_s) & (t < b.start_s + b.lap_s)
    else:
        bm = np.ones(len(t), bool)
    tb = t[bm] - t[bm][0]

    gps = c["gps"]
    if gps is not None:   # 1. track map coloured by speed
        ts = t[::10]
        e, n = np.interp(ts, gps["t"], gps["e"]), np.interp(ts, gps["t"], gps["n"])
        sc = ax[0].scatter(e, n, c=v[::10] * 3.6, s=3, cmap="turbo")
        fig.colorbar(sc, ax=ax[0], label="km/h")
        (a1, a2) = c["line"]; ax[0].plot([a1[0], a2[0]], [a1[1], a2[1]], "k-", lw=3, label="start/finish")
        ax[0].set_aspect("equal"); ax[0].legend(loc="upper right")
    ax[0].set(title="Track map (speed)", xlabel="east (m)", ylabel="north (m)")

    mv = v > 3 if v is not None else np.ones(len(t), bool)      # 2. friction circle
    ax[1].scatter(A[mv, 1][::5], A[mv, 0][::5], s=2, alpha=0.3)
    for r in (0.5, 1.0, 1.5):
        th = np.linspace(0, 2 * np.pi, 200); ax[1].plot(r * np.cos(th), r * np.sin(th), "k:", lw=0.8)
    ax[1].set(title="Friction circle (grip used)", xlabel="lateral g (+ left)", ylabel="longitudinal g (+ accel)")
    ax[1].set_aspect("equal"); ax[1].axhline(0, c="k", lw=0.5); ax[1].axvline(0, c="k", lw=0.5)

    ax[2].plot(tb, A[bm, 0], label="longitudinal"); ax[2].plot(tb, A[bm, 1], label="lateral")   # 3. g vs time
    if len(ev) and have_laps:
        for _, r in ev[ev.lap == int(b.lap)].iterrows():
            col = {"brake": "red", "flat_out": "green"}.get(r.type, "blue")
            ax[2].axvspan(r.t_start - b.start_s, r.t_end - b.start_s, color=col, alpha=0.12)
    ax[2].set(title="G-forces, best lap (red=brake, blue=corner, green=flat-out)", xlabel="s into lap", ylabel="g")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    if v is not None:   # 4. speed: GPS vs accel-only vs Kalman
        ax[3].plot(gps["t"], gps["v"] * 3.6, ".", ms=4, c="k", label="GPS (~1 Hz)")
        ax[3].plot(t, c["v_acc"] * 3.6, "--", lw=0.8, label="accel only (drifts)")
        ax[3].plot(t, v * 3.6, lw=1, label="Kalman fused (100 Hz)")
        ax[3].set_ylim(-5, max(gps["v"]) * 3.6 * 1.3); ax[3].legend()
    ax[3].set(title="Speed", xlabel="s", ylabel="km/h"); ax[3].grid(alpha=0.3)

    if have_laps:   # 5 & 6. laps on a distance axis, and time delta to best lap
        d_best = c["dist"][bm] - c["dist"][bm][0]
        for _, L in laps.iterrows():
            m = (t >= L.start_s) & (t < L.start_s + L.lap_s)
            d = c["dist"][m] - c["dist"][m][0]; d = d / d[-1] * d_best[-1]
            best = L.lap == b.lap
            ax[4].plot(d, v[m] * 3.6, lw=2.2 if best else 0.8, label=f"lap {int(L.lap)}" if best else None)
            ax[5].plot(d_best, np.interp(d_best, d, t[m] - t[m][0]) - (t[bm] - t[bm][0]),
                       lw=2.2 if best else 0.9, label=f"lap {int(L.lap)}")
        ax[5].axhline(0, c="k", lw=0.5); ax[5].legend(fontsize=7, ncol=3); ax[4].legend()
    ax[4].set(title="Speed vs distance, every lap (bold = best)", xlabel="m", ylabel="km/h"); ax[4].grid(alpha=0.3)
    ax[5].set(title="Time delta to best lap (rising = losing time)", xlabel="m", ylabel="s"); ax[5].grid(alpha=0.3)

    if have_laps:   # 7. lap times vs IR
        ax[6].bar(laps.lap, laps.lap_s, color="tab:blue", label="telemetry")
        if "ir_s" in laps: ax[6].plot(laps.lap, laps.ir_s, "k_", ms=25, mew=2, label="IR lap counter")
        lo = laps.lap_s.min(); ax[6].set_ylim(lo * 0.97, laps.lap_s.max() * 1.02); ax[6].legend()
    ax[6].set(title="Lap times", xlabel="lap", ylabel="s")

    ax[7].plot(tb, c["roll"][bm], label="roll"); ax[7].plot(tb, c["pitch"][bm], label="pitch")   # 8. orientation
    ax[7].set(title="Orientation, best lap (complementary filter)", xlabel="s into lap", ylabel="deg")
    ax[7].legend(); ax[7].grid(alpha=0.3)

    f, p = c["psd"]   # 9. vibration
    for i, lab in enumerate(("longitudinal", "lateral", "vertical")): ax[8].semilogy(f, p[:, i], label=lab)
    ax[8].axvline(c["cutoff"], c="k", ls="--", label=f"low-pass cut-off {c['cutoff']} Hz")
    ax[8].set(title=f"Chassis vibration while moving (peak {c['vib_peak_hz']:.0f} Hz)", xlabel="Hz", ylabel="g²/Hz")
    ax[8].legend(); ax[8].grid(alpha=0.3)

    ax[9].axis("off")
    ax[9].text(0, 1, "\n".join(f"{k}: {val}" for k, val in c["summary"].items() if not isinstance(val, (list, dict))),
               va="top", family="monospace", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="Sensor Logger zip export or unzipped folder")
    ap.add_argument("--out", default="telemetry_out")
    ap.add_argument("--cutoff", type=float, default=3.0, help="g-force low-pass cut-off, Hz (default 3)")
    ap.add_argument("--tau", type=float, default=1.0, help="complementary filter time constant, s")
    ap.add_argument("--line", help="start/finish as LAT1,LON1,LAT2,LON2 (default: auto on main straight)")
    ap.add_argument("--ir", help="CSV of IR lap-counter lap times in seconds (last column)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. record -> load
    fr = load(args.session)
    da, a_raw = total_accel(fr); dw = fr["Gyroscope"]
    t0 = min(da["time"].iloc[0], dw["time"].iloc[0])
    ta, tw = secs(da, t0), secs(dw, t0)
    gps = gps_track(fr, t0)

    # 3. sync: resample everything onto one shared 100 Hz timeline
    t = np.arange(max(ta[0], tw[0]), min(ta[-1], tw[-1]), 1 / FS)
    a, w = interp(t, ta, a_raw), interp(t, tw, xyz(dw))

    # 2. align phone axes to kart axes
    rest = find_rest(t, a, gps)
    w = w - w[rest].mean(0)                 # gyro bias: whatever it reads while parked
    R, conv = align(t, a, w, gps, rest)
    A_raw, W = (conv * a) @ R.T, w @ R.T

    # 4. filter out vibration
    A = lowpass(A_raw, args.cutoff)

    # 3b. GPS latency, then 5. fuse
    lag, v, v_acc, dist, tc, laps, line = 0.0, None, None, None, np.array([]), pd.DataFrame(), None
    if gps is not None:
        lag = gps_lag(t, A[:, 0], gps); gps["t"] = gps["t"] - lag
        v, _ = kalman_speed(t, A[:, 0], gps)
        v_acc = np.cumsum(A[:, 0] * G) / FS; v_acc += np.interp(t[0], gps["t"], gps["v"]) - v_acc[0]
        dist = np.cumsum(v) / FS
        # 6. laps
        if args.line:
            la1, lo1, la2, lo2 = map(float, args.line.split(","))
            e, n = to_en([la1, la2], [lo1, lo2], gps["lat0"], gps["lon0"])
            line = (np.array([e[0], n[0]]), np.array([e[1], n[1]]))
        else:
            line = auto_line(gps)
        tc = crossings(gps, line, t, dist)
        laps = lap_table(tc, t, A, v, dist)
        if args.ir and len(laps): compare_ir(laps, args.ir)
    roll, pitch = complementary(A, W, v, args.tau)
    events = detect_events(t, A, v, tc)

    # vibration spectrum (unfiltered, while moving)
    mv = v > 3 if v is not None else np.ones(len(t), bool)
    f, p = welch(A_raw[mv], fs=FS, nperseg=512, axis=0)
    vib_peak = f[f > 5][np.argmax(p[f > 5].sum(1))]

    # validation numbers
    s = {"duration_s": round(t[-1] - t[0], 1), "imu_rate_hz": round(1 / np.median(np.diff(ta)), 1),
         "parked_g_should_be_1": round(float(np.linalg.norm(a[rest].mean(0))), 3),
         "accel_sign_convention": "+1 (gravity reads up)" if conv == 1 else "-1 (gravity reads down)",
         "cutoff_hz": args.cutoff, "vibration_peak_hz": round(float(vib_peak), 1),
         "max_lat_g": round(float(np.abs(A[mv, 1]).max()), 2),
         "max_brake_g": round(float(-A[mv, 0].min()), 2), "max_accel_g": round(float(A[mv, 0].max()), 2)}
    if gps is not None:
        gm = (gps["t"] > t[0]) & (gps["t"] < t[-1])
        s["gps_rate_hz"] = round(1 / np.median(np.diff(gps["t"])), 2)
        s["gps_lag_s"] = round(float(lag), 2)
        s["top_speed_kmh"] = round(float(v.max() * 3.6), 1)
        s["kalman_vs_gps_rmse_kmh"] = round(float(np.sqrt(np.mean((np.interp(gps["t"][gm], t, v) - gps["v"][gm]) ** 2)) * 3.6), 2)
        s["accel_only_drift_kmh"] = round(float(np.abs(v_acc - v).max() * 3.6), 1)
        moving = gm & (gps["v"] > 3)
        yaw_err = np.interp(gps["t"][moving], t, np.cumsum(W[:, 2]) / FS) - gps["hdg"][moving]
        s["gyro_heading_drift_deg_per_min"] = round(float(np.degrees(np.polyfit(gps["t"][moving], yaw_err, 1)[0]) * 60), 2)
        s["line_latlon"] = [round(float(x), 7) for pt in line for x in to_latlon(pt[0], pt[1], gps["lat0"], gps["lon0"])]
    if len(laps):
        s["laps"] = len(laps); s["best_lap_s"] = float(laps.lap_s.min())
        s["lap_std_s"] = round(float(laps.lap_s.std()), 3)
        if "ir_s" in laps:
            s["tel_vs_ir_mean_abs_s"] = round(float(laps.tel_minus_ir_s.abs().mean()), 3)
            s["tel_vs_ir_max_abs_s"] = round(float(laps.tel_minus_ir_s.abs().max()), 3)
    if len(events):
        s.update({f"n_{k}": int(n) for k, n in events.type.value_counts().items()})

    # outputs
    pd.DataFrame({"t_s": t, "ax_g": A[:, 0], "ay_g": A[:, 1], "az_g": A[:, 2], "yaw_rate_dps": np.degrees(W[:, 2]),
                  "roll_deg": roll, "pitch_deg": pitch, "speed_kmh": v * 3.6 if v is not None else np.nan,
                  "dist_m": dist if dist is not None else np.nan,
                  "lap": np.searchsorted(tc, t, side="right")}).round(4).to_csv(out / "telemetry.csv", index=False)
    laps.round(3).to_csv(out / "laps.csv", index=False)
    events.round(3).to_csv(out / "events.csv", index=False)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    dashboard(out / "dashboard.png", dict(t=t, A=A, v=v, v_acc=v_acc, gps=gps, laps=laps, tc=tc, events=events,
                                          line=line, dist=dist, roll=roll, pitch=pitch, psd=(f, p),
                                          cutoff=args.cutoff, vib_peak_hz=vib_peak, summary=s))
    print(json.dumps(s, indent=2))
    if len(laps): print(laps.round(3).to_string(index=False))
    print(f"-> {out}/dashboard.png")


if __name__ == "__main__":
    main()
