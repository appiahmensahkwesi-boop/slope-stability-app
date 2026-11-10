"""
SlopeLab - Slope Stability Analysis Studio (dark studio interface)
------------------------------------------
Limit-equilibrium slope stability (Fellenius and Bishop's Simplified) with
  * layered soils (horizontal layers, own c', phi', unit weights)
  * water table (sloping line, hydrostatic pore pressure at slice bases)
  * automatic critical-circle search (grid + local refinement)
  * interactive 2D section, safety-factor map, rotatable 3D view
  * parametric study charts
  * site locations map

Geometry: toe at (0, 0). Flat ground at y = 0 left of the toe, slope rises to
the right to height H at x = H / tan(beta), then flat at y = H.
Elevations are measured from the toe level.

Run locally:  streamlit run slope_app.py
"""

import copy
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # lets the engine be tested without Streamlit
    st = None

GAMMA_W = 9.81  # unit weight of water, kN/m3
LAYER_COLORS = ["#d8b98a", "#b98d5b", "#9a7549", "#7a5b3a"]
STATUS_COLORS = {
    "Unstable": "#e5484d",
    "Marginal": "#f5a623",
    "Acceptable": "#3b82f6",
    "Stable": "#2e9e5b",
    "Current (unsaved)": "#8b5cf6",
}


# =====================================================================
# 1. MODEL PREPARATION
# =====================================================================
def prepare(M):
    """Turn the user model (dict) into arrays used by the engine."""
    H = float(M["H"])
    beta = float(M["beta"])
    tanb = np.tan(np.radians(beta))
    layers = M["layers"]

    gam = np.array([l["gam"] for l in layers], float)
    gsat = np.array([l["gsat"] for l in layers], float)
    c = np.array([l["c"] for l in layers], float)
    tphi = np.tan(np.radians([l["phi"] for l in layers]))

    ifaces, top = [], H
    for l in layers[:-1]:
        top -= float(l["t"])
        ifaces.append(top)
    ib = np.array(ifaces, float)  # interface elevations, descending
    hi = np.concatenate([[1e9], ib])
    lo = np.concatenate([ib, [-1e9]])

    return dict(
        H=H, beta=beta, tanb=tanb, xcrest=H / tanb,
        Dmax=float(M["Dfac"]) * H,
        gam=gam, gsat=gsat, c=c, tphi=tphi, ib=ib, hi=hi, lo=lo,
        water_on=bool(M["water_on"]),
        hw_toe=float(M["hw_toe"]), hw_crest=float(M["hw_crest"]),
        nlayers=len(layers),
    )


def ground_y(x, P):
    return np.clip(np.asarray(x, float) * P["tanb"], 0.0, P["H"])


def water_y(x, P):
    """Water-table elevation at x (capped at ground level). -1e9 if no water."""
    x = np.asarray(x, float)
    if not P["water_on"]:
        return np.full_like(x, -1e9)
    t = np.clip(x / P["xcrest"], 0.0, 1.0)
    hw = P["hw_toe"] + (P["hw_crest"] - P["hw_toe"]) * t
    return np.minimum(hw, ground_y(x, P))


# =====================================================================
# 2. ENGINE (vectorised over many trial circles at once)
# =====================================================================
def circle_from(xe, xi, R, P):
    """Circle through ground points at xe (exit) and xi (entry), radius R.
    Centre lies above the chord so the slip surface is the lower arc."""
    ge, gi = ground_y(xe, P), ground_y(xi, P)
    dx, dy = xi - xe, gi - ge
    L = np.hypot(dx, dy)
    s = np.sqrt(np.maximum(R ** 2 - (L / 2) ** 2, 0.0))
    xc = (xe + xi) / 2 - dy / L * s
    yc = (ge + gi) / 2 + dx / L * s
    return xc, yc, ge, gi, L


def analyse_circles(xe, xi, R, P, nsl=30, want_slices=False):
    xe = np.atleast_1d(np.asarray(xe, float))
    xi = np.atleast_1d(np.asarray(xi, float))
    R = np.atleast_1d(np.asarray(R, float))

    xc, yc, ge, gi, L = circle_from(xe, xi, R, P)
    ok = (L > 1e-6) & (R >= L / 2 - 1e-9) & (yc >= np.maximum(ge, gi) - 1e-9)
    inside = (xc >= xe) & (xc <= xi)
    ylow = np.where(inside, yc - R, np.minimum(ge, gi))
    ok &= ylow >= -P["Dmax"]

    k = (np.arange(nsl) + 0.5) / nsl
    b = ((xi - xe) / nsl)[:, None]
    xm = xe[:, None] + (xi - xe)[:, None] * k[None, :]
    dxc = xm - xc[:, None]
    Rr = R[:, None]
    arcm = yc[:, None] - np.sqrt(np.maximum(Rr ** 2 - dxc ** 2, 0.0))
    top = ground_y(xm, P)
    h = top - arcm
    ok &= h.min(axis=1) > -1e-3
    h = np.maximum(h, 0.0)

    sina = np.clip(dxc / Rr, -0.999, 0.999)
    cosa = np.sqrt(1.0 - sina ** 2)
    tana = sina / cosa
    alpha = np.arcsin(sina)

    hwm = water_y(xm, P)

    # weight per metre of slope length (per unit width), layer by layer
    wb = np.zeros_like(h)
    for g, gs, lo, hi in zip(P["gam"], P["gsat"], P["lo"], P["hi"]):
        a = np.maximum(arcm, lo)
        bt = np.minimum(top, hi)
        t = np.maximum(bt - a, 0.0)
        wet = np.minimum(np.maximum(np.minimum(bt, hwm) - a, 0.0), t)
        wb += g * (t - wet) + gs * wet
    W = wb * b

    # strength at slice base, from the layer the base lies in
    idx = (arcm[..., None] < P["ib"][None, None, :]).sum(-1)
    cb = P["c"][idx]
    tphi = P["tphi"][idx]
    u = GAMMA_W * np.maximum(hwm - arcm, 0.0)  # pore pressure at base, kPa
    l = b / cosa

    drive = (W * sina).sum(axis=1)
    drive_ok = drive > 1e-9
    safe_drive = np.where(drive_ok, drive, 1.0)

    # Ordinary method of slices (Fellenius)
    resF = (cb * l + (W * cosa - u * l) * tphi).sum(axis=1)
    Ff = np.where(drive_ok, resF / safe_drive, np.inf)
    Ff = np.where(Ff > 0, Ff, np.inf)

    # Bishop's simplified method (iterative)
    F = np.where(np.isfinite(Ff) & (Ff > 0.05), Ff, 1.0)
    for _ in range(80):
        m = np.maximum(cosa * (1.0 + tana * tphi / F[:, None]), 1e-3)
        num = ((cb * b + (W - u * b) * tphi) / m).sum(axis=1)
        Fn = np.clip(num / safe_drive, 0.05, 50.0)
        done = np.max(np.abs(Fn - F)) < 1e-6
        F = Fn
        if done:
            break
    m = cosa * (1.0 + tana * tphi / F[:, None])
    bad = (m.min(axis=1) < 0.2) | (~drive_ok)
    Fb = np.where(bad, np.inf, F)

    Ff = np.where(ok, Ff, np.inf)
    Fb = np.where(ok, Fb, np.inf)

    out = dict(Ff=Ff, Fb=Fb, xc=xc, yc=yc, ok=ok)
    if want_slices:
        out.update(xm=xm, b=b[:, 0], h=h, arcm=arcm, top=top, W=W, u=u,
                   alpha=alpha, hwm=hwm, cb=cb, tphi=tphi, xe=xe, xi=xi, R=R)
    return out


# =====================================================================
# 3. CRITICAL CIRCLE SEARCH (grid + random local refinement)
# =====================================================================
DENSITY = {"Fast": (14, 16, 9), "Standard": (24, 28, 13), "Fine": (36, 40, 17)}


def _eval_chunks(xe, xi, R, P, nsl, chunk=4000):
    n = len(xe)
    Ff, Fb = np.empty(n), np.empty(n)
    xc, yc = np.empty(n), np.empty(n)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        r = analyse_circles(xe[sl], xi[sl], R[sl], P, nsl)
        Ff[sl], Fb[sl], xc[sl], yc[sl] = r["Ff"], r["Fb"], r["xc"], r["yc"]
    return Ff, Fb, xc, yc


def search(P, nsl=30, method="Bishop", density="Standard", seed=7):
    H, xcr = P["H"], P["xcrest"]
    na, nb, nr = DENSITY[density]
    xe_v = np.linspace(-1.5 * H, 0.85 * xcr, na)
    xi_v = np.linspace(0.15 * xcr, xcr + 3.0 * H, nb)
    rf_v = np.geomspace(1.0, 12.0, nr)
    XE, XI, RF = [a.ravel() for a in np.meshgrid(xe_v, xi_v, rf_v, indexing="ij")]
    keep = XI > XE + 0.25 * H
    XE, XI, RF = XE[keep], XI[keep], RF[keep]
    chord = np.hypot(XI - XE, ground_y(XI, P) - ground_y(XE, P))
    R = RF * chord / 2

    Ff, Fb, xc, yc = _eval_chunks(XE, XI, R, P, nsl)
    Fm = Fb if method == "Bishop" else Ff
    if not np.isfinite(Fm).any():
        return None

    min_f, min_b = np.min(Ff), np.min(Fb)
    j = int(np.argmin(Fm))
    best = (XE[j], XI[j], RF[j])
    best_F = Fm[j]

    rng = np.random.default_rng(seed)
    sx_e = xe_v[1] - xe_v[0]
    sx_i = xi_v[1] - xi_v[0]
    for r in range(8):
        sc = 0.7 ** r
        n = 500
        e = best[0] + rng.normal(0, sx_e * sc, n)
        i = best[1] + rng.normal(0, sx_i * sc, n)
        rf = np.maximum(1.0, best[2] * (1 + rng.normal(0, 0.25 * sc, n)))
        good = i > e + 0.25 * H
        e, i, rf = e[good], i[good], rf[good]
        if len(e) == 0:
            continue
        ch = np.hypot(i - e, ground_y(i, P) - ground_y(e, P))
        Rn = rf * ch / 2
        f2, b2, _, _ = _eval_chunks(e, i, Rn, P, nsl)
        min_f, min_b = min(min_f, np.min(f2)), min(min_b, np.min(b2))
        m2 = b2 if method == "Bishop" else f2
        kk = int(np.argmin(m2))
        if m2[kk] < best_F:
            best_F = m2[kk]
            best = (e[kk], i[kk], rf[kk])

    chord_b = np.hypot(best[1] - best[0],
                       ground_y(best[1], P) - ground_y(best[0], P))
    Rb = best[2] * chord_b / 2
    final = analyse_circles([best[0]], [best[1]], [Rb], P, nsl, want_slices=True)

    valid = np.where(np.isfinite(Fm))[0]
    if len(valid) > 3500:
        valid = rng.choice(valid, 3500, replace=False)
    cloud = dict(xc=xc[valid], yc=yc[valid], F=Fm[valid])

    order = np.argsort(Fm)
    finite_order = [o for o in order[:200] if np.isfinite(Fm[o])][::5]
    trials = dict(xe=XE[finite_order], xi=XI[finite_order],
                  R=R[finite_order], F=Fm[finite_order])

    return dict(
        method=method, nsl=nsl,
        fos_bishop=float(min_b) if np.isfinite(min_b) else np.nan,
        fos_fellenius=float(min_f) if np.isfinite(min_f) else np.nan,
        best=final, cloud=cloud, trials=trials,
    )


def status_of(F):
    if not np.isfinite(F):
        return "No failure surface found", "info"
    if F < 1.0:
        return "Unstable", "error"
    if F < 1.3:
        return "Marginal", "warning"
    if F < 1.5:
        return "Acceptable", "info"
    return "Stable", "success"


# =====================================================================
# 4. FIGURES (plotly) - every figure works in light or dark mode
# =====================================================================
def arc_xy(xe, xi, xc, yc, R, n=90):
    x = np.linspace(xe, xi, n)
    y = yc - np.sqrt(np.maximum(R ** 2 - (x - xc) ** 2, 0.0))
    return x, y


def _view_range(P):
    return -1.7 * P["H"], P["xcrest"] + 3.2 * P["H"]


def _theme(dark):
    if dark:
        return dict(text="#cbd5e1", ground="#e2e8f0", grid="rgba(148,163,184,0.18)",
                    slice="rgba(203,213,225,0.45)", trial="rgba(148,163,184,0.40)")
    return dict(text="#1f2328", ground="#222222", grid="rgba(128,128,128,0.20)",
                slice="rgba(80,80,80,0.55)", trial="rgba(120,120,120,0.45)")


def _style(fig, T, dark, scene=False):
    """Transparent background and readable text for the dark interface."""
    if not dark:
        return fig
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=T["text"]))
    if scene:
        ax = dict(gridcolor=T["grid"], zerolinecolor=T["grid"],
                  showbackground=False)
        fig.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)", xaxis=dict(**ax),
                                     yaxis=dict(**ax), zaxis=dict(**ax)))
    else:
        fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
        fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    return fig


def _base_section(go, P, T):
    """Layers, water table and ground line. Returns (fig, x, ground_y)."""
    D = P["Dmax"]
    xmin, xmax = _view_range(P)
    x = np.linspace(xmin, xmax, 500)
    gy = ground_y(x, P)
    fig = go.Figure()

    for i in range(P["nlayers"]):
        upper = np.minimum(gy, P["hi"][i])
        lower = np.full_like(x, max(P["lo"][i], -D))
        upper = np.maximum(upper, lower)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=LAYER_COLORS[i % len(LAYER_COLORS)],
            name=f"Layer {i + 1}", hoverinfo="skip"))

    if P["water_on"]:
        wy = water_y(x, P)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([wy, np.full_like(x, -D)[::-1]]),
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor="rgba(30,144,255,0.22)", name="Saturated zone",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=wy, mode="lines",
            line=dict(color="#38a3ff", width=3, dash="dash"),
            name="Water table"))

    fig.add_trace(go.Scatter(x=x, y=gy, mode="lines",
                             line=dict(color=T["ground"], width=3),
                             name="Ground surface"))
    return fig, x, gy


def fig_preview(P, dark=False, height=380):
    """Slope geometry before any analysis has been run."""
    import plotly.graph_objects as go

    T = _theme(dark)
    H, D = P["H"], P["Dmax"]
    xmin, xmax = _view_range(P)
    fig, _, _ = _base_section(go, P, T)
    fig.update_layout(
        xaxis=dict(title="Distance (m)", range=[xmin, xmax]),
        yaxis=dict(title="Elevation (m)", range=[-D * 1.05, H * 2.2],
                   scaleanchor="x", scaleratio=1),
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.22))
    return _style(fig, T, dark)


def fig_section(P, res, show_slices=True, show_trials=True, dark=False,
                title=True, height=520):
    import plotly.graph_objects as go

    T = _theme(dark)
    H, D = P["H"], P["Dmax"]
    xmin, xmax = _view_range(P)
    fig, x, gy = _base_section(go, P, T)

    b = res["best"]
    xe, xi, R = float(b["xe"][0]), float(b["xi"][0]), float(b["R"][0])
    xc, yc = float(b["xc"][0]), float(b["yc"][0])

    if show_trials and len(res["trials"]["xe"]):
        tx, ty = [], []
        t = res["trials"]
        for e, i_, r in zip(t["xe"], t["xi"], t["R"]):
            cx, cy, _, _, _ = circle_from(e, i_, r, P)
            ax_, ay_ = arc_xy(e, i_, cx, cy, r, 40)
            tx += list(ax_) + [None]
            ty += list(ay_) + [None]
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines", line=dict(color=T["trial"], width=1),
            name="Other trial circles", hoverinfo="skip"))

    ax_, ay_ = arc_xy(xe, xi, xc, yc, R)
    top_ = ground_y(ax_, P)
    fig.add_trace(go.Scatter(
        x=np.concatenate([ax_, ax_[::-1]]),
        y=np.concatenate([ay_, top_[::-1]]),
        fill="toself", mode="lines", line=dict(width=0),
        fillcolor="rgba(239,68,68,0.28)", name="Sliding mass",
        hoverinfo="skip"))

    if show_slices:
        n = res["nsl"]
        ex = np.linspace(xe, xi, n + 1)
        ey = yc - np.sqrt(np.maximum(R ** 2 - (ex - xc) ** 2, 0.0))
        eg = ground_y(ex, P)
        sx, sy = [], []
        for a, bb, c_ in zip(ex, ey, eg):
            sx += [a, a, None]
            sy += [bb, c_, None]
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                                 line=dict(color=T["slice"], width=1),
                                 name="Slices", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines",
                             line=dict(color="#ef4444", width=4),
                             name="Critical slip surface"))
    fig.add_trace(go.Scatter(
        x=[xc, xe, xc, xi], y=[yc, ground_y(xe, P), yc, ground_y(xi, P)],
        mode="lines", line=dict(color="#ef4444", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[xc], y=[yc], mode="markers",
        marker=dict(color="#ef4444", size=11, symbol="cross"),
        name="Circle centre",
        hovertext=[f"centre ({xc:.2f}, {yc:.2f}) m, R = {R:.2f} m"],
        hoverinfo="text"))

    F = res["fos_bishop"] if res["method"] == "Bishop" else res["fos_fellenius"]
    layout = dict(
        xaxis=dict(title="Distance (m)", range=[xmin, xmax]),
        yaxis=dict(title="Elevation above toe (m)",
                   range=[-D * 1.05, H * 2.2], scaleanchor="x", scaleratio=1),
        height=height, margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        legend=dict(orientation="h", y=-0.2 if dark else -0.18),
        hovermode="closest")
    if title:
        layout["title"] = f"Critical slip surface - FoS = {F:.3f} ({res['method']})"
    fig.update_layout(**layout)
    return _style(fig, T, dark)


def fig_centres(P, res, dark=False):
    import plotly.graph_objects as go

    T = _theme(dark)
    c = res["cloud"]
    xmin, xmax = _view_range(P)
    x = np.linspace(xmin, xmax, 300)
    Fmin = float(np.min(c["F"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ground_y(x, P), mode="lines",
                             line=dict(color=T["ground"], width=2),
                             name="Ground surface"))
    fig.add_trace(go.Scatter(
        x=c["xc"], y=c["yc"], mode="markers",
        marker=dict(size=6, color=np.clip(c["F"], Fmin, Fmin + 1.5),
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="FoS")),
        name="Trial circle centres",
        hovertext=[f"FoS {v:.2f}" for v in c["F"]], hoverinfo="text"))
    b = res["best"]
    fig.add_trace(go.Scatter(
        x=[float(b["xc"][0])], y=[float(b["yc"][0])], mode="markers",
        marker=dict(size=16, color="#facc15" if dark else "#000000",
                    symbol="star"),
        name="Critical centre"))
    fig.update_layout(
        title="Safety-factor map of trial circle centres",
        xaxis=dict(title="Distance (m)"),
        yaxis=dict(title="Elevation (m)", scaleanchor="x", scaleratio=1),
        height=460, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2))
    return _style(fig, T, dark)


def _strip_mesh(go, xs, zu, zl, y, color, name, opacity=1.0, showlegend=False):
    n = len(xs)
    X = np.concatenate([xs, xs])
    Z = np.concatenate([zu, zl])
    Y = np.full(2 * n, float(y))
    i = np.arange(n - 1)
    return go.Mesh3d(
        x=X, y=Y, z=Z,
        i=np.concatenate([i, i + 1]),
        j=np.concatenate([i + 1, i + n + 1]),
        k=np.concatenate([i + n, i + n]),
        color=color, opacity=opacity, name=name, flatshading=True,
        showlegend=showlegend, hoverinfo="name")


def fig_3d(P, res, width=20.0, dark=False, height=620, title=True):
    import plotly.graph_objects as go

    T = _theme(dark)
    D = P["Dmax"]
    xmin, xmax = _view_range(P)
    x = np.linspace(xmin, xmax, 220)
    gy = ground_y(x, P)
    yv = np.array([0.0, float(width)])
    X, Y = np.meshgrid(x, yv)
    fig = go.Figure()

    for face_y in (0.0, float(width)):
        for i in range(P["nlayers"]):
            upper = np.minimum(gy, P["hi"][i])
            lower = np.full_like(x, max(P["lo"][i], -D))
            upper = np.maximum(upper, lower)
            fig.add_trace(_strip_mesh(
                go, x, upper, lower, face_y, LAYER_COLORS[i % len(LAYER_COLORS)],
                f"Layer {i + 1}", 1.0, showlegend=(face_y == 0.0)))

    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.tile(gy, (2, 1)), surfacecolor=np.tile(gy, (2, 1)),
        colorscale=[[0, "#cdb48a"], [1, "#8fb36b"]], showscale=False,
        opacity=0.55, name="Ground surface", showlegend=True))

    if P["water_on"]:
        wy = water_y(x, P)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=np.tile(wy, (2, 1)),
            colorscale=[[0, "#1e90ff"], [1, "#1e90ff"]], showscale=False,
            opacity=0.35, name="Water table", showlegend=True))

    b = res["best"]
    xe, xi, R = float(b["xe"][0]), float(b["xi"][0]), float(b["R"][0])
    xc, yc = float(b["xc"][0]), float(b["yc"][0])
    sx, sz = arc_xy(xe, xi, xc, yc, R, 80)
    SX, SY = np.meshgrid(sx, yv)
    fig.add_trace(go.Surface(
        x=SX, y=SY, z=np.tile(sz, (2, 1)),
        colorscale=[[0, "#ef4444"], [1, "#ef4444"]], showscale=False,
        opacity=0.85, name="Critical slip surface", showlegend=True))

    fig.add_trace(_strip_mesh(go, sx, ground_y(sx, P), sz,
                              -0.01 * max(width, 1.0), "#ef4444",
                              "Sliding mass", 0.6, showlegend=True))

    for face_y in (0.0, float(width)):
        fig.add_trace(go.Scatter3d(
            x=x, y=np.full_like(x, face_y), z=gy, mode="lines",
            line=dict(color=T["ground"], width=5), showlegend=False,
            hoverinfo="skip"))

    F = res["fos_bishop"] if res["method"] == "Bishop" else res["fos_fellenius"]
    layout = dict(
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="Distance (m)"),
            yaxis=dict(title="Width (m)"),
            zaxis=dict(title="Elevation (m)"),
            camera=dict(eye=dict(x=1.5, y=-1.9, z=0.9))),
        height=height, margin=dict(l=0, r=0, t=50 if title else 0, b=0),
        legend=dict(orientation="h", y=-0.05))
    if title:
        layout["title"] = (f"3D view (2D analysis extruded over {width:g} m)"
                           f" - FoS = {F:.3f}")
    fig.update_layout(**layout)
    return _style(fig, T, dark, scene=True)


def fig_parametric(df, label, current, dark=False):
    import plotly.graph_objects as go

    T = _theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["value"], y=df["Bishop"], mode="lines+markers",
                             name="Bishop", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["value"], y=df["Fellenius"], mode="lines+markers",
                             name="Fellenius", line=dict(width=2, dash="dash")))
    fig.add_hline(y=1.0, line=dict(color="#ef4444", dash="dot"),
                  annotation_text="FoS = 1.0 (failure)")
    fig.add_hline(y=1.5, line=dict(color="#22c55e", dash="dot"),
                  annotation_text="FoS = 1.5 (target)")
    if current is not None:
        fig.add_vline(x=current, line=dict(color="#94a3b8", dash="dash"),
                      annotation_text="current")
    fig.update_layout(
        title=f"Factor of safety vs {label}",
        xaxis=dict(title=label), yaxis=dict(title="Factor of safety"),
        height=440, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2))
    return _style(fig, T, dark)


def fig_sites_map(df, dark=False):
    import plotly.express as px

    d = df.copy()
    d["size"] = 14
    center, zoom = auto_view(d)
    style = "carto-darkmatter" if dark else "open-street-map"
    kwargs = dict(lat="Lat", lon="Lon", color="Status",
                  color_discrete_map=STATUS_COLORS, hover_name="Site",
                  hover_data={"FoS": ":.2f", "Lat": ":.5f", "Lon": ":.5f",
                              "size": False, "Status": True},
                  size="size", size_max=16, zoom=zoom, center=center,
                  height=460)
    try:
        fig = px.scatter_map(d, map_style=style, **kwargs)
    except AttributeError:  # older plotly
        fig = px.scatter_mapbox(d, mapbox_style=style, **kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if dark:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#cbd5e1"))
    return fig


# =====================================================================
# 5. PARAMETRIC STUDY
# =====================================================================
PARAMS = {
    "Slope angle (deg)": ("beta", 10.0, 80.0),
    "Slope height H (m)": ("H", 2.0, 40.0),
    "Layer 1 cohesion c' (kPa)": ("L1c", 0.0, 100.0),
    "Layer 1 friction angle (deg)": ("L1phi", 0.0, 45.0),
    "Water level at crest side (m above toe)": ("hw_crest", 0.0, 40.0),
    "Water level at toe (m above toe)": ("hw_toe", 0.0, 40.0),
}


def apply_param(M, key, v):
    M2 = copy.deepcopy(M)
    if key == "beta":
        M2["beta"] = v
    elif key == "H":
        M2["H"] = v
    elif key == "L1c":
        M2["layers"][0]["c"] = v
    elif key == "L1phi":
        M2["layers"][0]["phi"] = v
    elif key == "hw_crest":
        M2["water_on"], M2["hw_crest"] = True, v
    elif key == "hw_toe":
        M2["water_on"], M2["hw_toe"] = True, v
    return M2


def current_value(M, key):
    if key in ("beta", "H", "hw_crest", "hw_toe"):
        return float(M[key])
    return float(M["layers"][0]["c" if key == "L1c" else "phi"])


def run_parametric(M, key, values, nsl=24):
    rows = []
    for v in values:
        P = prepare(apply_param(M, key, float(v)))
        r = search(P, nsl=nsl, method="Bishop", density="Fast")
        if r is None:
            rows.append((float(v), np.nan, np.nan))
        else:
            rows.append((float(v), r["fos_bishop"], r["fos_fellenius"]))
    return pd.DataFrame(rows, columns=["value", "Bishop", "Fellenius"])



# =====================================================================
# 6b. SITES: sample data, place search, map view
# =====================================================================
SAMPLE_SITES = [
    # Demo values only (not real analyses): they show how the map works.
    ("Ho, Ghana (sample)", 6.6101, 0.4703, 10.0, 30.0, 1, False, 1.53, "Stable"),
    ("Accra, Ghana (sample)", 5.6037, -0.1870, 6.0, 35.0, 2, True, 1.12, "Marginal"),
    ("Kumasi, Ghana (sample)", 6.6885, -1.6244, 8.0, 40.0, 2, True, 0.94, "Unstable"),
    ("Hong Kong (sample)", 22.3193, 114.1694, 15.0, 45.0, 2, True, 1.38, "Acceptable"),
    ("Rio de Janeiro, Brazil (sample)", -22.9068, -43.1729, 12.0, 40.0, 2, True, 1.05, "Marginal"),
    ("Kathmandu, Nepal (sample)", 27.7172, 85.3240, 20.0, 38.0, 3, True, 0.89, "Unstable"),
    ("Seattle, USA (sample)", 47.6062, -122.3321, 9.0, 30.0, 2, True, 1.64, "Stable"),
    ("Oslo, Norway (sample)", 59.9139, 10.7522, 7.0, 25.0, 2, False, 1.71, "Stable"),
    ("Sydney, Australia (sample)", -33.8688, 151.2093, 11.0, 35.0, 1, False, 1.42, "Acceptable"),
]


def sample_sites_df():
    return pd.DataFrame([
        dict(Site=n, Lat=la, Lon=lo, H=h, Angle=a, Layers=nl, Water=w,
             Method="Bishop", FoS=f, Status=st_)
        for (n, la, lo, h, a, nl, w, f, st_) in SAMPLE_SITES])


def parse_geocode(data):
    """First hit of a Nominatim JSON answer -> (lat, lon, name) or None."""
    if not data:
        return None
    d = data[0]
    return float(d["lat"]), float(d["lon"]), d.get("display_name", "")


def geocode(query, timeout=8):
    """Look up any place name in the world with OpenStreetMap Nominatim."""
    url = ("https://nominatim.openstreetmap.org/search?format=json&limit=1&q="
           + urllib.parse.quote(query))
    req = urllib.request.Request(
        url, headers={"User-Agent": "SlopeLab-student-project/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_geocode(json.loads(r.read().decode("utf-8")))


def auto_view(df):
    """Map centre and zoom that fit all points (works anywhere on Earth)."""
    lat0, lon0 = float(df["Lat"].mean()), float(df["Lon"].mean())
    span = max(float(df["Lat"].max() - df["Lat"].min()),
               float(df["Lon"].max() - df["Lon"].min()))
    if span < 1e-4:
        zoom = 12.0
    else:
        zoom = float(np.clip(np.log2(360.0 / (span * 1.6)), 1.0, 12.0))
    return dict(lat=lat0, lon=lon0), zoom

# =====================================================================
# 6. PRESETS / DEFAULTS
# =====================================================================
def _layer_defaults():
    base = {}
    vals = [(4.0, 18.0, 20.0, 15.0, 20.0), (4.0, 18.5, 20.0, 20.0, 22.0),
            (4.0, 19.0, 20.5, 25.0, 25.0), (4.0, 19.5, 21.0, 30.0, 28.0)]
    for i, (t, g, gs, c, p) in enumerate(vals, start=1):
        base.update({f"L{i}_t": t, f"L{i}_gam": g, f"L{i}_gsat": gs,
                     f"L{i}_c": c, f"L{i}_phi": p})
    return base


DEFAULTS = dict(
    H=10.0, beta=30.0, Dfac=2.0, nlayers=1, water_on=False,
    hw_toe=0.0, hw_crest=6.0, method="Bishop", nsl=30, density="Standard",
    width=20.0, site_name="Site 1", lat=6.6101, lon=0.4703, preset="-",
    place_query="", place_msg="",
    **_layer_defaults())

PRESETS = {
    "Default homogeneous slope": dict(
        H=10.0, beta=30.0, nlayers=1, water_on=False,
        L1_gam=18.0, L1_gsat=20.0, L1_c=15.0, L1_phi=20.0),
    "Check: Taylor chart (phi = 0, 45 deg)": dict(
        H=10.0, beta=45.0, nlayers=1, water_on=False,
        L1_gam=18.0, L1_gsat=18.0, L1_c=25.0, L1_phi=0.0),
    "Laterite over clay with water table (illustrative)": dict(
        H=8.0, beta=35.0, nlayers=2, water_on=True, hw_toe=0.0, hw_crest=4.0,
        L1_t=3.0, L1_gam=19.0, L1_gsat=20.0, L1_c=10.0, L1_phi=30.0,
        L2_gam=17.5, L2_gsat=18.5, L2_c=12.0, L2_phi=15.0),
    "Wet clay slope (illustrative)": dict(
        H=6.0, beta=30.0, nlayers=1, water_on=True, hw_toe=0.0, hw_crest=5.0,
        L1_gam=17.0, L1_gsat=18.5, L1_c=10.0, L1_phi=10.0),
    "Road embankment on soft clay (illustrative)": dict(
        H=5.0, beta=34.0, nlayers=2, water_on=True, hw_toe=0.0, hw_crest=1.0,
        L1_t=5.0, L1_gam=19.0, L1_gsat=20.0, L1_c=8.0, L1_phi=30.0,
        L2_gam=16.0, L2_gsat=17.0, L2_c=12.0, L2_phi=12.0),
}


def model_from_state(S):
    layers = []
    for i in range(1, int(S["nlayers"]) + 1):
        layers.append(dict(t=float(S[f"L{i}_t"]), gam=float(S[f"L{i}_gam"]),
                           gsat=float(S[f"L{i}_gsat"]), c=float(S[f"L{i}_c"]),
                           phi=float(S[f"L{i}_phi"])))
    return dict(H=float(S["H"]), beta=float(S["beta"]), Dfac=float(S["Dfac"]),
                layers=layers, water_on=bool(S["water_on"]),
                hw_toe=float(S["hw_toe"]), hw_crest=float(S["hw_crest"]))


def results_table(P, res):
    b = res["best"]
    rows = [
        ("Factor of safety - Bishop (min)", f"{res['fos_bishop']:.3f}"),
        ("Factor of safety - Fellenius (min)", f"{res['fos_fellenius']:.3f}"),
        ("Exit point x (m)", f"{float(b['xe'][0]):.2f}"),
        ("Entry point x (m)", f"{float(b['xi'][0]):.2f}"),
        ("Circle centre (x, y) (m)",
         f"({float(b['xc'][0]):.2f}, {float(b['yc'][0]):.2f})"),
        ("Radius (m)", f"{float(b['R'][0]):.2f}"),
        ("Sliding mass weight (kN per m run)", f"{float(b['W'][0].sum()):.1f}"),
        ("Slices used", f"{res['nsl']}"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


# =====================================================================
# 7. STREAMLIT APP - SlopeLab studio interface
# =====================================================================
def _cache(fn):
    if st is None:
        return fn
    return st.cache_data(show_spinner=False)(fn)


@_cache
def cached_search(M_, nsl_, method_, density_):
    return search(prepare(M_), nsl=nsl_, method=method_, density=density_)


GREEN, AMBER, RED, BLUE = "#22c55e", "#f59e0b", "#ef4444", "#3b82f6"
PAGES = ["Model", "Geometry", "Soil Layers", "Water Table", "Analysis",
         "Settings", "Help"]
PAGE_ICONS = {"Model": "📄", "Geometry": "📐", "Soil Layers": "🧱",
              "Water Table": "💧", "Analysis": "📊", "Settings": "⚙️",
              "Help": "❓"}

# Keys that define a worksheet's model (used to detect changed inputs)
MODEL_KEYS = (["H", "beta", "Dfac", "nl", "wmode", "wd", "hwt", "hwc"]
              + [f"L{i}_{k}" for i in range(1, 5)
                 for k in ("t", "gam", "gsat", "c", "phi")])

THEME_CSS = """
<style>
:root{--bg:#0b1220;--card:#0f1a2d;--line:#1c3050;--text:#e6edf7;--muted:#93a4bd;}
.stApp{background:var(--bg);color:var(--text);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1600px;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"]{color:var(--text);}
[data-testid="stCaptionContainer"]{color:var(--muted)!important;}
[data-testid="stSidebar"]{background:#0a111e;border-right:1px solid var(--line);}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{display:none;}
/* inputs */
div[data-baseweb="input"],div[data-baseweb="base-input"],
div[data-baseweb="select"]>div{background:#0a1424!important;
 border-color:#1f3556!important;border-radius:8px!important;}
.stApp input,.stApp textarea{color:var(--text)!important;
 -webkit-text-fill-color:var(--text)!important;}
[data-testid="stNumberInput"] button{background:#0f1d33!important;color:#cbd5e1!important;}
[data-testid="stNumberInput"] label p,[data-testid="stTextInput"] label p,
[data-testid="stSelectbox"] label p,[data-testid="stSlider"] label p{
 color:#c7d2e4!important;font-size:.92rem;}
/* bordered containers = cards */
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;
 border-radius:12px!important;background:var(--card);}
/* primary button = Run Analysis */
[data-testid="stBaseButton-primary"],button[kind="primary"]{
 background:linear-gradient(90deg,#16c46a,#0e9f6e)!important;
 border:1px solid #22c55e!important;color:#fff!important;font-weight:700;
 font-size:1.15rem;min-height:3.2rem;border-radius:10px;width:100%;}
[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.1);}
[data-testid="stBaseButton-secondary"],button[kind="secondary"]{
 background:#0f1d33;border:1px solid #1f3556;color:#dbe6f6;border-radius:10px;}
.st-key-newws button{background:transparent!important;border:1.5px solid #3b82f6!important;
 color:#60a5fa!important;font-weight:600;min-height:3rem;width:100%;}
.st-key-delws button{background:transparent!important;border:1.5px solid #ef4444!important;
 color:#f87171!important;font-weight:600;min-height:3rem;width:100%;}
/* worksheet tabs + view toggle + sidebar navigation (radio as pills) */
.st-key-wstabs div[role="radiogroup"],.st-key-viewtoggle div[role="radiogroup"]{gap:6px;flex-wrap:wrap;}
.st-key-wstabs label,.st-key-viewtoggle label{background:#101c2f;border:1px solid #1f3556;
 border-radius:10px;padding:.65rem 1.3rem;cursor:pointer;}
.st-key-viewtoggle label{padding:.4rem .9rem;}
.st-key-wstabs label>div:first-child,.st-key-viewtoggle label>div:first-child,
[data-testid="stSidebar"] label>div:first-child:has(+ div){display:none;}
.st-key-wstabs label:has(input:checked),.st-key-viewtoggle label:has(input:checked){
 background:linear-gradient(180deg,#2563eb,#1d4ed8);border-color:#3b82f6;}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:2px;}
[data-testid="stSidebar"] div[role="radiogroup"] label{width:100%;padding:.7rem 1rem;
 border-radius:8px;border-left:3px solid transparent;cursor:pointer;}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
 background:linear-gradient(90deg,#1e3a8a,#12284a);border-left:3px solid #3b82f6;}
/* html cards */
.sl-card{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:14px 16px;margin-bottom:12px;}
.sl-title{font-weight:700;font-size:1.05rem;}
.sl-sub{color:var(--muted);font-size:.88rem;}
.sl-green{color:#22c55e;font-weight:700;font-size:1.05rem;}
.sl-fos{font-size:3.4rem;font-weight:800;line-height:1;}
.sl-pill{display:block;text-align:center;border-radius:999px;padding:.55rem 1rem;
 font-weight:700;margin-top:10px;}
.sl-row{display:flex;align-items:center;gap:10px;margin:6px 0;}
.sl-dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.sl-ok{display:inline-flex;width:20px;height:20px;border-radius:50%;
 align-items:center;justify-content:center;font-size:.8rem;color:#0b1220;}
.sl-empty{border:1px dashed #28446e;border-radius:12px;text-align:center;
 padding:3.2rem 1rem;color:var(--muted);}
</style>
"""


# ------------------------------------------------------------ state helpers
def blank_params(n):
    p = dict(H=0.0, beta=0.0, Dfac=2.0, nl=1, wmode="Depth below crest",
             wd=0.0, hwt=0.0, hwc=0.0, width=20.0, view="2D View",
             site_name=f"Site {n}", lat=6.6101, lon=0.4703,
             place_query="", place_msg="")
    p.update(_layer_defaults())
    p.update(L1_gam=0.0, L1_gsat=0.0, L1_c=0.0, L1_phi=0.0)
    return p


def apply_example(p, preset):
    """Write a PRESETS entry into a worksheet's parameters."""
    for k, v in preset.items():
        if k == "nlayers":
            p["nl"] = int(v)
        elif k == "water_on":
            if not v:
                p["wd"], p["wmode"] = 0.0, "Depth below crest"
            else:
                p["wmode"] = "Custom line"
        elif k == "hw_toe":
            p["hwt"] = float(v)
        elif k == "hw_crest":
            p["hwc"] = float(v)
        else:
            p[k] = float(v)


def example_params(n, preset_name="Default homogeneous slope"):
    p = blank_params(n)
    apply_example(p, PRESETS[preset_name])
    return p


def validate(p):
    """Message describing what is missing, or None if the model can be run."""
    nl = int(p["nl"])
    if p["H"] <= 0:
        return "Enter the slope height (H)."
    if not 10 <= p["beta"] <= 80:
        return "Slope angle (β) must be between 10° and 80°."
    for i in range(1, nl + 1):
        if p[f"L{i}_gam"] < 10:
            return f"Enter the unit weight (γ) for layer {i} (typically 16-22 kN/m³)."
        if p[f"L{i}_c"] <= 0 and p[f"L{i}_phi"] <= 0:
            return f"Enter the cohesion and/or friction angle for layer {i}."
    for i in range(1, nl):
        if p[f"L{i}_t"] <= 0:
            return f"Enter a thickness for layer {i}."
    return None


def model_from_params(p):
    nl = int(p["nl"])
    layers = []
    for i in range(1, nl + 1):
        gam = float(p[f"L{i}_gam"])
        layers.append(dict(
            t=float(p[f"L{i}_t"]), gam=gam,
            gsat=max(float(p[f"L{i}_gsat"]), gam),
            c=float(p[f"L{i}_c"]), phi=float(p[f"L{i}_phi"])))
    if p["wmode"] == "Custom line":
        water_on, hwt, hwc = True, float(p["hwt"]), float(p["hwc"])
    elif p["wd"] > 0:
        level = float(p["H"]) - float(p["wd"])
        water_on, hwt, hwc = True, level, level
    else:
        water_on, hwt, hwc = False, 0.0, 0.0
    return dict(H=float(p["H"]), beta=float(p["beta"]), Dfac=float(p["Dfac"]),
                layers=layers, water_on=water_on, hw_toe=hwt, hw_crest=hwc)


def _sig(p, stg):
    return (tuple(p[k] for k in MODEL_KEYS),
            stg["method"], stg["density"], int(stg["nsl"]))


def status3(F):
    """Three-band classification used by the interface."""
    if not np.isfinite(F):
        return "Unavailable", "#64748b"
    if F < 1.0:
        return "Unstable", RED
    if F < 1.5:
        return "Marginal", AMBER
    return "Stable", GREEN


def exit_text(xe):
    if abs(xe) < 0.05:
        return "at the toe"
    if xe < 0:
        return f"{abs(xe):.1f} m beyond the toe"
    return f"{xe:.1f} m up the slope from the toe"


# --------------------------------------------------------------- callbacks
def run_analysis(wid):
    S = st.session_state
    p, stg = S["store"][wid], S["store"]["settings"]
    err = validate(p)
    if err:
        S["flash"] = ("error", err)
        return
    M = model_from_params(p)
    res = cached_search(M, int(stg["nsl"]), stg["method"], stg["density"])
    if res is None:
        S["flash"] = ("error", "No valid slip surface was found. Try a larger "
                      "model depth (Geometry page) or different soil values.")
        return
    S["results"][wid] = dict(sig=_sig(p, stg), M=M, res=res)
    S["flash"] = None


def add_ws():
    S = st.session_state
    wid = S["next_id"]
    S["next_id"] += 1
    S["store"][wid] = blank_params(wid)
    S["ws"].append(dict(id=wid, kind="Blank"))
    S["store"]["ui"]["active"] = wid


def delete_ws():
    S = st.session_state
    wid = S["store"]["ui"]["active"]
    if len(S["ws"]) == 1:  # keep one worksheet: reset it to blank
        S["store"][wid] = blank_params(wid)
        S["ws"][0]["kind"] = "Blank"
        S["results"].pop(wid, None)
        return
    S["ws"] = [w for w in S["ws"] if w["id"] != wid]
    S["store"].pop(wid, None)
    S["results"].pop(wid, None)
    for k in [k for k in S.keys() if str(k).startswith(f"f_{wid}_")]:
        del S[k]
    S["store"]["ui"]["active"] = S["ws"][0]["id"]


def load_example():
    S = st.session_state
    name = S.get("ex_pick")
    wid = S["store"]["ui"]["active"]
    if name in PRESETS:
        p = blank_params(wid)
        apply_example(p, PRESETS[name])
        p["site_name"] = S["store"][wid]["site_name"]
        S["store"][wid] = p
        for w in S["ws"]:
            if w["id"] == wid:
                w["kind"] = "Example"
        S["results"].pop(wid, None)
    S["ex_pick"] = None


def find_place(wid):
    S = st.session_state
    p = S["store"][wid]
    q = str(p.get("place_query", "")).strip()
    if not q:
        p["place_msg"] = "Type a place name first."
        return
    try:
        hit = geocode(q)
    except Exception:  # noqa: BLE001
        p["place_msg"] = ("Could not reach the place-search service. "
                          "Type the coordinates instead.")
        return
    if hit is None:
        p["place_msg"] = f"No place found for '{q}'. Try a fuller name."
        return
    p["lat"], p["lon"] = round(hit[0], 5), round(hit[1], 5)
    p["place_msg"] = "Found: " + hit[2]


def reset_app():
    S = st.session_state
    for k in list(S.keys()):
        del S[k]


def init_state():
    S = st.session_state
    if "store" in S:
        return
    S["store"] = {
        "settings": dict(method="Bishop", density="Standard", nsl=30),
        "ui": dict(page="Model", active=1),
        1: example_params(1),
    }
    S["ws"] = [dict(id=1, kind="Example")]
    S["next_id"] = 2
    S["results"] = {}
    S["sites"] = []
    S["flash"] = None
    run_analysis(1)  # the app opens with the example already analysed


# ------------------------------------------------------------ widget helpers
def field(kind, label, owner, name, **kw):
    """A widget whose value lives in S['store'][owner][name].

    The value is re-seeded on every run, so it survives page changes (Streamlit
    forgets widgets that are not drawn on the current page)."""
    S = st.session_state
    key = f"f_{owner}_{name}"
    S[key] = S["store"][owner][name]

    def cb():
        S["store"][owner][name] = S[key]

    fn = dict(num=st.number_input, sel=st.selectbox, chk=st.checkbox,
              txt=st.text_input, rad=st.radio, sld=st.slider,
              ssl=st.select_slider)[kind]
    return fn(label, key=key, on_change=cb, **kw)


def box(key=None, border=False):
    try:
        return st.container(border=border, key=key)
    except TypeError:
        try:
            return st.container(border=border)
        except TypeError:
            return st.container()


def show_plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # noqa: BLE001
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:  # noqa: BLE001
        st.dataframe(df, use_container_width=True, hide_index=True)


def card_title(icon, text, sub=None, color=None):
    sub_html = f'<div class="sl-sub">{sub}</div>' if sub else ""
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="sl-card"><div class="sl-title"{style}>{icon}&nbsp; '
            f'{text}</div>{sub_html}</div>')


def shield_svg(color, symbol):
    marks = {
        "check": '<path d="M20 37 L29 46 L45 26" stroke="#0b1220" stroke-width="6" '
                 'fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
        "warn": '<path d="M32 22 V40 M32 48 V49" stroke="#0b1220" stroke-width="7" '
                'fill="none" stroke-linecap="round"/>',
        "cross": '<path d="M22 26 L42 46 M42 26 L22 46" stroke="#0b1220" '
                 'stroke-width="6" fill="none" stroke-linecap="round"/>',
    }
    return ('<svg width="66" height="74" viewBox="0 0 64 72"><path d="M32 4 L58 14 '
            'V36 C58 52 46 63 32 68 C18 63 6 52 6 36 V14 Z" fill="' + color + '"/>'
            + marks[symbol] + '</svg>')


def results_panel(entry, stale):
    """Right-hand panel: results, key findings, guidelines, recommendations."""
    if entry is None:
        st.markdown(
            '<div class="sl-card" style="border-color:#1f5a3a">'
            '<div class="sl-title" style="font-size:1.2rem">📊&nbsp; Analysis Results</div>'
            '<div class="sl-sub" style="margin-top:14px">No results yet. Enter your '
            'parameters and press <b>Run Analysis</b>.</div>'
            '<span class="sl-pill" style="background:#1b2a42;color:#93a4bd">'
            'Awaiting analysis</span></div>', unsafe_allow_html=True)
        _guidelines()
        return

    res = entry["res"]
    method = res["method"]
    F = res["fos_bishop"] if method == "Bishop" else res["fos_fellenius"]
    other = res["fos_fellenius"] if method == "Bishop" else res["fos_bishop"]
    label, color = status3(F)
    symbol = {"Stable": "check", "Marginal": "warn"}.get(label, "cross")
    b = res["best"]
    xe, R = float(b["xe"][0]), float(b["R"][0])
    fos_txt = f"{F:.2f}" if np.isfinite(F) else "-"

    st.markdown(
        f'<div class="sl-card" style="border-color:#1f5a3a">'
        f'<div class="sl-title" style="font-size:1.2rem">📊&nbsp; Analysis Results</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-top:14px;gap:12px">'
        f'<div style="display:flex;align-items:center;gap:14px">{shield_svg(color, symbol)}'
        f'<div><div class="sl-sub">Factor of Safety (FOS)</div>'
        f'<div class="sl-fos" style="color:{color}">{fos_txt}</div></div></div>'
        f'<div style="border-left:1px solid #1f3556;padding-left:14px">'
        f'<div class="sl-sub">Method</div>'
        f'<div style="font-size:1.4rem;font-weight:800">{method}</div></div></div>'
        f'<span class="sl-pill" style="border:1.5px solid {color};color:{color};'
        f'background:rgba(255,255,255,0.03)">Slope is {label.upper()}</span></div>',
        unsafe_allow_html=True)
    if stale:
        st.markdown('<div class="sl-card" style="border-color:#7a5a12;color:#f3c969">'
                    'Inputs changed since this analysis. Press <b>Run Analysis</b> '
                    'to update the results.</div>', unsafe_allow_html=True)

    result_line = {"Stable": "Safe (FOS ≥ 1.5)",
                   "Marginal": "Marginal (1.0 ≤ FOS < 1.5)",
                   "Unstable": "Unsafe (FOS < 1.0)"}.get(label, "Not available")
    rows = [
        ("Critical Slip Surface", f"exits {exit_text(xe)}, radius {R:.1f} m"),
        ("Minimum FOS", f"{fos_txt} ({method}); other method {other:.2f}"),
        ("Result", result_line),
    ]
    kf = "".join(
        f'<div class="sl-row"><span class="sl-ok" style="background:{color}">✔</span>'
        f'<span><b>{a}:</b> {v}</span></div>' for a, v in rows)
    st.markdown(f'<div class="sl-card"><div class="sl-title">💡&nbsp; Key Findings</div>'
                f'<div style="margin-top:8px">{kf}</div></div>',
                unsafe_allow_html=True)
    _guidelines()

    advice = {
        "Stable": "Your slope is stable. Continue with the current design or consider "
                  "additional checks for critical conditions (heavy rain, loading, "
                  "long-term strength).",
        "Marginal": "The slope is marginal. Consider a flatter angle, lower height, "
                    "better drainage to lower the water table, or ground improvement, "
                    "then re-run the analysis.",
        "Unstable": "The slope is unstable. A redesign is needed: reduce the angle or "
                    "height, drain the slope, or add retaining or stabilising measures.",
    }.get(label, "Check your inputs and run the analysis again.")
    st.markdown(
        f'<div class="sl-card" style="border-color:#8a6a14"><div class="sl-title" '
        f'style="color:#f3c969">'
        f'<span style="color:{color}">●</span>&nbsp; Recommendations</div>'
        f'<div style="margin-top:6px">{advice}</div>'
        f'<div class="sl-sub" style="margin-top:8px">Simplified 2D analysis. Verify '
        f'against hand calculations or design software before relying on it.</div></div>',
        unsafe_allow_html=True)


def _guidelines():
    st.markdown(
        '<div class="sl-card" style="border-color:#1f4a8a">'
        '<div class="sl-title" style="color:#60a5fa">ℹ️&nbsp; Safety Guidelines</div>'
        '<div style="margin-top:8px">'
        f'<div class="sl-row"><span class="sl-dot" style="background:{RED}"></span>'
        '<b>FOS &lt; 1.0</b>&nbsp;:&nbsp; Unstable (failure likely)</div>'
        f'<div class="sl-row"><span class="sl-dot" style="background:{AMBER}"></span>'
        '<b>1.0 – 1.5</b>&nbsp;:&nbsp; Marginal (use with caution)</div>'
        f'<div class="sl-row"><span class="sl-dot" style="background:{GREEN}"></span>'
        '<b>&gt; 1.5</b>&nbsp;:&nbsp; Stable (generally safe)</div></div></div>',
        unsafe_allow_html=True)


# ------------------------------------------------------------------- pages
def get_entry(wid, p):
    """(result entry or None, stale flag) for the active worksheet."""
    S = st.session_state
    entry = S["results"].get(wid)
    if entry is None:
        return None, False
    return entry, entry["sig"] != _sig(p, S["store"]["settings"])


def flash():
    S = st.session_state
    if S.get("flash"):
        kind, msg = S["flash"]
        getattr(st, kind)(msg)
        S["flash"] = None


def geometry_figure(wid, p, entry, stale, big=False):
    """2D/3D picture for the Slope Geometry card."""
    view = p["view"]
    h = 430 if big else 360
    err = validate(p)
    if err:
        st.markdown(
            '<div class="sl-empty"><div style="font-size:2.4rem">⛰️</div>'
            '<div style="font-size:1.2rem;font-weight:600;color:#e6edf7">'
            'No geometry defined yet</div>Enter your parameters and click '
            '“Run Analysis”</div>', unsafe_allow_html=True)
        return
    P = prepare(model_from_params(p))
    if entry is not None and not stale:
        res = entry["res"]
        if view == "3D View":
            show_plot(fig_3d(prepare(entry["M"]), res, float(p["width"]),
                             dark=True, height=h + 60, title=False),
                      key=f"g3d_{wid}")
        else:
            show_plot(fig_section(prepare(entry["M"]), res, True, False,
                                  dark=True, title=False, height=h),
                      key=f"g2d_{wid}")
        return
    if view == "3D View":
        st.info("Press **Run Analysis** to see the 3D view.")
        return
    show_plot(fig_preview(P, dark=True, height=h), key=f"gpre_{wid}")


def page_model(wid, p, ws_kind):
    S = st.session_state
    entry, stale = get_entry(wid, p)
    left, right = st.columns([2.15, 1], gap="medium")

    with left:
        h1, h2 = st.columns([3.2, 1.4])
        with h1:
            st.markdown(card_title(
                "📝", f"Worksheet {wid} – {ws_kind}",
                "Enter your slope geometry and soil parameters below."),
                unsafe_allow_html=True)
        with h2:
            st.selectbox("Load an example", list(PRESETS.keys()), index=None,
                         placeholder="Load an example…", key="ex_pick",
                         on_change=load_example, label_visibility="collapsed")

        with box(border=True):
            st.markdown('<div class="sl-green">🧊&nbsp; Model Parameters</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                field("num", "Slope Height (H), m", wid, "H",
                      min_value=0.0, max_value=100.0, step=0.5)
                field("num", "Slope Angle (β), °", wid, "beta",
                      min_value=0.0, max_value=89.0, step=1.0)
                field("num", "Cohesion (c'), kPa", wid, "L1_c",
                      min_value=0.0, max_value=500.0, step=1.0)
            with c2:
                field("num", "Friction Angle (φ'), °", wid, "L1_phi",
                      min_value=0.0, max_value=50.0, step=1.0)
                field("num", "Unit Weight (γ), kN/m³", wid, "L1_gam",
                      min_value=0.0, max_value=26.0, step=0.5)
                field("num", "Water Table Depth, m", wid, "wd",
                      min_value=0.0, max_value=100.0, step=0.5,
                      disabled=(p["wmode"] == "Custom line"),
                      help="Depth below the crest. Leave 0 for no water table. "
                           "More options are on the Water Table page.")
            if int(p["nl"]) > 1:
                st.caption(f"This model has {int(p['nl'])} soil layers. The "
                           "values above are for layer 1; edit the others on the "
                           "Soil Layers page.")

        with box(border=True):
            g1, g2 = st.columns([3, 2])
            g1.markdown('<div class="sl-green">📈&nbsp; Slope Geometry</div>',
                        unsafe_allow_html=True)
            with g2:
                with box("viewtoggle"):
                    field("rad", "View", wid, "view",
                          options=["2D View", "3D View"], horizontal=True,
                          label_visibility="collapsed")
            geometry_figure(wid, p, entry, stale)

        with box("runbtn"):
            st.button("▶  Run Analysis", type="primary", on_click=run_analysis,
                      args=(wid,), key="btn_run")
        flash()

    with right:
        results_panel(entry, stale)


def page_geometry(wid, p):
    entry, stale = get_entry(wid, p)
    S = st.session_state
    left, right = st.columns([1, 1.5], gap="medium")
    with left:
        with box(border=True):
            st.markdown('<div class="sl-green">📐&nbsp; Model extents</div>',
                        unsafe_allow_html=True)
            field("num", "Model depth below toe (× H)", wid, "Dfac",
                  min_value=0.5, max_value=6.0, step=0.5,
                  help="How deep the slip surface is allowed to go.")
            field("num", "3D block width (m)", wid, "width",
                  min_value=2.0, max_value=200.0, step=1.0)
        with box(border=True):
            st.markdown('<div class="sl-green">📍&nbsp; Site location</div>',
                        unsafe_allow_html=True)
            field("txt", "Site name", wid, "site_name")
            field("txt", "Search any place in the world", wid, "place_query",
                  placeholder="e.g. Ho Technical University, Ghana")
            st.button("Find place", on_click=find_place, args=(wid,),
                      key="btn_place")
            if p.get("place_msg"):
                st.caption(p["place_msg"])
            l1, l2 = st.columns(2)
            with l1:
                field("num", "Latitude", wid, "lat", min_value=-90.0,
                      max_value=90.0, step=0.0001, format="%.5f")
            with l2:
                field("num", "Longitude", wid, "lon", min_value=-180.0,
                      max_value=180.0, step=0.0001, format="%.5f")
    with right:
        with box(border=True):
            st.markdown('<div class="sl-green">🗺️&nbsp; Sites map</div>',
                        unsafe_allow_html=True)
            sites_map(wid, p, entry, stale)
        if validate(p) is None:
            with box(border=True):
                st.markdown('<div class="sl-green">⛰️&nbsp; Section</div>',
                            unsafe_allow_html=True)
                show_plot(fig_preview(prepare(model_from_params(p)), dark=True,
                                      height=300), key="geo_prev")


def sites_map(wid, p, entry, stale):
    S = st.session_state
    F = np.nan
    label = "Current (unsaved)"
    if entry is not None:
        res = entry["res"]
        F = res["fos_bishop"] if res["method"] == "Bishop" else res["fos_fellenius"]
    b1, b2, b3 = st.columns(3)
    if b1.button("Save this site", key="btn_save"):
        if entry is None or not np.isfinite(F):
            st.warning("Run the analysis first, then save the site.")
        else:
            S["sites"].append(dict(
                Site=p["site_name"], Lat=float(p["lat"]), Lon=float(p["lon"]),
                H=float(p["H"]), Angle=float(p["beta"]), Layers=int(p["nl"]),
                Water=bool(entry["M"]["water_on"]),
                Method=entry["res"]["method"], FoS=round(float(F), 3),
                Status=status3(F)[0]))
            st.success(f"Saved {p['site_name']}.")
    if b2.button("Load sample sites", key="btn_sample"):
        if not S.get("_sample_loaded"):
            S["sites"].extend(sample_sites_df().to_dict("records"))
            S["_sample_loaded"] = True
        st.info("Sample sites loaded (demo values, not real analyses).")
    if b3.button("Clear saved sites", key="btn_clear"):
        S["sites"] = []
        S["_sample_loaded"] = False

    up = st.file_uploader("Optional: load a sites CSV saved earlier", type="csv")
    if up is not None and not S.get("_loaded_" + up.name):
        try:
            extra = pd.read_csv(up)
            need = {"Site", "Lat", "Lon", "FoS", "Status"}
            if need.issubset(extra.columns):
                S["sites"].extend(extra.to_dict("records"))
                S["_loaded_" + up.name] = True
                st.success(f"Loaded {len(extra)} site(s).")
            else:
                st.error("CSV needs the columns: " + ", ".join(sorted(need)))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")

    cols = ["Site", "Lat", "Lon", "FoS", "Status"]
    saved = (pd.DataFrame(S["sites"]) if S["sites"]
             else pd.DataFrame(columns=cols))
    current = pd.DataFrame([dict(
        Site=f"{p['site_name']} (current)", Lat=float(p["lat"]),
        Lon=float(p["lon"]), FoS=float(F) if np.isfinite(F) else 0.0,
        Status=label)])
    both = pd.concat([saved[cols], current], ignore_index=True)
    try:
        show_plot(fig_sites_map(both, dark=True), key="figsites")
    except Exception:  # noqa: BLE001
        st.map(both.rename(columns={"Lat": "latitude", "Lon": "longitude"}))
    if len(saved):
        show_df(saved)
        st.download_button("Download sites (CSV)",
                           saved.to_csv(index=False).encode("utf-8"),
                           "slope_sites.csv", "text/csv")
    st.caption("Any latitude and longitude on Earth works. Saved sites last "
               "for this browser session; download the CSV to keep them.")


def page_layers(wid, p):
    nl = int(p["nl"])
    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        with box(border=True):
            st.markdown('<div class="sl-green">🧱&nbsp; Soil layers</div>',
                        unsafe_allow_html=True)
            field("sel", "Number of layers", wid, "nl", options=[1, 2, 3, 4])
            st.caption("Layers are horizontal, counted from the crest level "
                       "downward. The last layer continues to the model base.")
        nl = int(p["nl"])
        for i in range(1, nl + 1):
            with box(border=True):
                swatch = LAYER_COLORS[(i - 1) % len(LAYER_COLORS)]
                st.markdown(f'<div class="sl-title"><span style="color:{swatch}">'
                            f'■</span>&nbsp; Layer {i}</div>', unsafe_allow_html=True)
                if i < nl:
                    field("num", "Thickness (m)", wid, f"L{i}_t",
                          min_value=0.1, max_value=100.0, step=0.5)
                else:
                    st.caption("Extends down to the model base")
                a, b = st.columns(2)
                with a:
                    field("num", "Unit weight (kN/m³)", wid, f"L{i}_gam",
                          min_value=0.0, max_value=26.0, step=0.5)
                    field("num", "Cohesion c' (kPa)", wid, f"L{i}_c",
                          min_value=0.0, max_value=500.0, step=1.0)
                with b:
                    field("num", "Saturated weight (kN/m³)", wid, f"L{i}_gsat",
                          min_value=0.0, max_value=26.0, step=0.5,
                          help="Used below the water table. If lower than the "
                               "unit weight, the unit weight is used.")
                    field("num", "Friction angle φ' (°)", wid, f"L{i}_phi",
                          min_value=0.0, max_value=50.0, step=1.0)
    with right:
        if validate(p) is None:
            show_plot(fig_preview(prepare(model_from_params(p)), dark=True,
                                  height=420), key="lay_prev")
        else:
            st.markdown('<div class="sl-empty">Complete the model parameters to '
                        'preview the layers.</div>', unsafe_allow_html=True)


def page_water(wid, p):
    left, right = st.columns([1, 1.4], gap="medium")
    with left:
        with box(border=True):
            st.markdown('<div class="sl-green">💧&nbsp; Water table</div>',
                        unsafe_allow_html=True)
            field("rad", "Definition", wid, "wmode",
                  options=["Depth below crest", "Custom line"])
            if p["wmode"] == "Depth below crest":
                field("num", "Water table depth below crest (m)", wid, "wd",
                      min_value=0.0, max_value=100.0, step=0.5,
                      help="0 means no water table.")
                st.caption("A horizontal water level. 0 = no water table.")
            else:
                w1, w2 = st.columns(2)
                with w1:
                    field("num", "Level at toe (m)", wid, "hwt",
                          min_value=-50.0, max_value=100.0, step=0.5)
                with w2:
                    field("num", "Level at crest (m)", wid, "hwc",
                          min_value=-50.0, max_value=100.0, step=0.5)
                st.caption("Levels are measured up from the toe. The line is "
                           "straight between the toe and the crest.")
            st.caption("Pore pressure below the water line is taken as hydrostatic "
                       "(conservative). Set saturated unit weights on the Soil "
                       "Layers page.")
    with right:
        if validate(p) is None:
            show_plot(fig_preview(prepare(model_from_params(p)), dark=True,
                                  height=420), key="wat_prev")
        else:
            st.markdown('<div class="sl-empty">Complete the model parameters to '
                        'preview the water table.</div>', unsafe_allow_html=True)


def page_analysis(wid, p):
    S = st.session_state
    entry, stale = get_entry(wid, p)
    if entry is None:
        st.info("No analysis for this worksheet yet. Press the button to run it.")
        with box("runbtn"):
            st.button("▶  Run Analysis", type="primary", on_click=run_analysis,
                      args=(wid,), key="btn_run2")
        flash()
        return
    P = prepare(entry["M"])
    res = entry["res"]
    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        if stale:
            st.warning("Inputs changed since this analysis. Press Run Analysis "
                       "on the Model page to update.")
        t1, t2, t3, t4, t5 = st.tabs(["2D section", "Safety-factor map",
                                      "3D view", "Results table",
                                      "Parametric study"])
        with t1:
            o1, o2 = st.columns(2)
            sl = o1.checkbox("Show slices", value=True, key="an_sl")
            tr = o2.checkbox("Show other trial circles", value=True, key="an_tr")
            show_plot(fig_section(P, res, sl, tr, dark=True, title=True),
                      key="an_2d")
        with t2:
            show_plot(fig_centres(P, res, dark=True), key="an_map")
        with t3:
            show_plot(fig_3d(P, res, float(p["width"]), dark=True), key="an_3d")
            st.caption("A 3D visualisation of the 2D (plane-strain) analysis "
                       "extruded across the block width, not a full 3D analysis.")
        with t4:
            tbl = results_table(P, res)
            show_df(tbl)
            st.download_button("Download results (CSV)",
                               tbl.to_csv(index=False).encode("utf-8"),
                               "slope_results.csv", "text/csv")
        with t5:
            parametric_tab(entry["M"])
    with right:
        results_panel(entry, stale)


def parametric_tab(M):
    S = st.session_state
    label_p = st.selectbox("Parameter to vary", list(PARAMS.keys()),
                           key="pp_param")
    key_p, lim_lo, lim_hi = PARAMS[label_p]
    cur = current_value(M, key_p)
    if key_p in ("hw_crest", "hw_toe"):
        d_lo, d_hi = 0.0, float(M["H"])
    else:
        d_lo = max(lim_lo, round(cur * 0.5, 1))
        d_hi = min(lim_hi, round(cur * 1.6, 1) if cur > 0 else lim_lo + 30.0)
        if d_hi <= d_lo:
            d_lo, d_hi = lim_lo, min(lim_hi, lim_lo + 30.0)
    p1, p2, p3 = st.columns(3)
    v_lo = p1.number_input("From", value=float(d_lo), key=f"pmin_{key_p}")
    v_hi = p2.number_input("To", value=float(d_hi), key=f"pmax_{key_p}")
    n_pts = p3.slider("Points", 5, 20, 10, key=f"pn_{key_p}")
    if st.button("Run parametric study", key="btn_param"):
        if v_hi <= v_lo:
            st.error("'To' must be larger than 'From'.")
        else:
            with st.spinner("Running analyses..."):
                vals = np.linspace(v_lo, v_hi, int(n_pts))
                df = run_parametric(M, key_p, vals)
            S["param_result"] = dict(label=label_p, key=key_p, df=df, cur=cur)
    pr = S.get("param_result")
    if pr is not None:
        show_plot(fig_parametric(pr["df"], pr["label"], pr["cur"], dark=True),
                  key="figpar")
        show_df(pr["df"].rename(columns={"value": pr["label"]}).round(3))
        st.download_button("Download study (CSV)",
                           pr["df"].to_csv(index=False).encode("utf-8"),
                           "parametric_study.csv", "text/csv")
        st.caption("Uses the Fast search. Re-run after changing the model.")


def page_settings():
    S = st.session_state
    with box(border=True):
        st.markdown('<div class="sl-green">⚙️&nbsp; Analysis settings</div>',
                    unsafe_allow_html=True)
        field("rad", "Method", "settings", "method",
              options=["Bishop", "Fellenius"], horizontal=True)
        field("ssl", "Search density", "settings", "density",
              options=["Fast", "Standard", "Fine"])
        field("sld", "Number of slices", "settings", "nsl",
              min_value=15, max_value=60)
        st.caption("Press Run Analysis again after changing these settings.")
    with box(border=True):
        st.markdown('<div class="sl-green">🧹&nbsp; Reset</div>',
                    unsafe_allow_html=True)
        st.button("Reset the whole application", on_click=reset_app,
                  key="btn_reset")


def page_help():
    with box(border=True):
        st.markdown(
            """
### How to use SlopeLab
1. Pick a worksheet tab (use **New Worksheet** for another slope).
2. On **Model**, enter the height, angle and soil strength, or load an example.
3. Optional: add layers (**Soil Layers**), a water table (**Water Table**),
   the site location and sites map (**Geometry**).
4. Press **Run Analysis**. Results appear on the right; **Analysis** has the
   safety-factor map, 3D view, results table and parametric study.

### Method
Limit-equilibrium method of slices on circular slip surfaces. The critical
circle is found by a grid search over exit point, entry point and radius,
followed by local refinement.
            """)
        st.latex(r"F_{Bishop}=\frac{\sum\frac{c'b+(W-ub)\tan\phi'}{m_\alpha}}"
                 r"{\sum W\sin\alpha},\quad "
                 r"m_\alpha=\cos\alpha\left(1+\frac{\tan\alpha\tan\phi'}{F}\right)")
        st.latex(r"F_{Fellenius}=\frac{\sum\left[c'l+(W\cos\alpha-ul)"
                 r"\tan\phi'\right]}{\sum W\sin\alpha}")
        st.markdown(
            """
### Limits
- Horizontal layers, 2D plane strain, circular failure only.
- Water table is a straight line; pore pressure is hydrostatic below it
  (conservative compared with a seepage analysis).
- Use effective-stress values (c', φ') when a water table is present.
- No earthquake, surcharge or reinforcement loads yet.
- Verify results with hand calculations or design software (GEO5, Slide,
  GeoStudio) before using them for design.

### Quick check
Load **Check: Taylor chart (phi = 0, 45 deg)**. Taylor's chart gives
FoS = c' / (0.181 × γ × H) = 0.77.
            """)


# -------------------------------------------------------------------- main
def sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 6px 0">'
            '<svg width="92" height="54" viewBox="0 0 92 54"><path d="M2 52 L30 12 '
            'L44 30 L56 16 L90 52 Z" fill="none" stroke="#3b82f6" stroke-width="4" '
            'stroke-linejoin="round"/><path d="M30 12 L36 22 L28 20 Z" fill="#60a5fa"/>'
            '<path d="M56 16 L66 30 L50 28 Z" fill="#22c55e"/></svg>'
            '<div style="font-size:2rem;font-weight:800;letter-spacing:.5px">'
            'Slope<span style="color:#22c55e">Lab</span></div>'
            '<div class="sl-sub">Slope Stability Analysis Tool</div></div>',
            unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        field("rad", "Navigation", "ui", "page", options=PAGES,
              format_func=lambda x: f"{PAGE_ICONS[x]}  {x}",
              label_visibility="collapsed")
        st.markdown(
            '<div class="sl-card" style="margin-top:22px;border-color:#1f3a63">'
            '<div class="sl-title" style="color:#60a5fa">About SlopeLab</div>'
            '<div class="sl-sub" style="margin-top:4px">Professional tool for slope '
            'stability analysis using Bishop and Fellenius methods.</div></div>'
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:6px 4px"><span class="sl-sub">Version 1.0.0</span>'
            '<span><span class="sl-dot" style="background:#22c55e"></span>&nbsp;'
            '<b>Ready</b></span></div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="SlopeLab - Slope Stability Analysis Tool",
                       page_icon="⛰️", layout="wide",
                       initial_sidebar_state="expanded")
    init_state()
    S = st.session_state
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    sidebar()

    st.markdown('<div style="font-size:1.25rem;font-weight:700;margin-bottom:8px">'
                'SlopeLab – Slope Stability Analysis Tool</div>',
                unsafe_allow_html=True)

    ids = [w["id"] for w in S["ws"]]
    if S["store"]["ui"]["active"] not in ids:
        S["store"]["ui"]["active"] = ids[0]

    t_col, n_col, d_col = st.columns([6, 1.7, 1.7])
    with t_col:
        with box("wstabs"):
            field("rad", "Worksheets", "ui", "active", options=ids,
                  format_func=lambda i: f"📄  Worksheet {i}", horizontal=True,
                  label_visibility="collapsed")
    with n_col:
        with box("newws"):
            st.button("＋  New Worksheet", on_click=add_ws, key="btn_new")
    with d_col:
        with box("delws"):
            st.button("🗑  Delete Worksheet", on_click=delete_ws, key="btn_del")

    wid = S["store"]["ui"]["active"]
    if wid not in S["store"]:  # a worksheet was just removed
        wid = ids[0]
    p = S["store"][wid]
    ws_kind = next(w["kind"] for w in S["ws"] if w["id"] == wid)
    page = S["store"]["ui"]["page"]

    if page == "Model":
        page_model(wid, p, ws_kind)
    elif page == "Geometry":
        page_geometry(wid, p)
    elif page == "Soil Layers":
        page_layers(wid, p)
    elif page == "Water Table":
        page_water(wid, p)
    elif page == "Analysis":
        page_analysis(wid, p)
    elif page == "Settings":
        page_settings()
    else:
        page_help()


if __name__ == "__main__":
    main()
