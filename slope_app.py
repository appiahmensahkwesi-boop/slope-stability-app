"""
SlopeLab - Slope Stability Analysis Studio
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
# 4. FIGURES (plotly)
# =====================================================================
def arc_xy(xe, xi, xc, yc, R, n=90):
    x = np.linspace(xe, xi, n)
    y = yc - np.sqrt(np.maximum(R ** 2 - (x - xc) ** 2, 0.0))
    return x, y


def _view_range(P):
    return -1.7 * P["H"], P["xcrest"] + 3.2 * P["H"]


def fig_section(P, res, show_slices=True, show_trials=True):
    import plotly.graph_objects as go

    H, D = P["H"], P["Dmax"]
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
            fillcolor="rgba(30,144,255,0.20)", name="Saturated zone",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=wy, mode="lines",
            line=dict(color="#1e90ff", width=3, dash="dash"),
            name="Water table"))

    fig.add_trace(go.Scatter(x=x, y=gy, mode="lines",
                             line=dict(color="#222222", width=3),
                             name="Ground surface"))

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
            x=tx, y=ty, mode="lines",
            line=dict(color="rgba(120,120,120,0.45)", width=1),
            name="Other trial circles", hoverinfo="skip"))

    ax_, ay_ = arc_xy(xe, xi, xc, yc, R)
    top_ = ground_y(ax_, P)
    fig.add_trace(go.Scatter(
        x=np.concatenate([ax_, ax_[::-1]]),
        y=np.concatenate([ay_, top_[::-1]]),
        fill="toself", mode="lines", line=dict(width=0),
        fillcolor="rgba(229,72,77,0.25)", name="Sliding mass",
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
                                 line=dict(color="rgba(80,80,80,0.55)", width=1),
                                 name="Slices", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines",
                             line=dict(color="#e5484d", width=4),
                             name="Critical slip surface"))
    fig.add_trace(go.Scatter(
        x=[xc, xe, xc, xi], y=[yc, ground_y(xe, P), yc, ground_y(xi, P)],
        mode="lines", line=dict(color="#e5484d", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[xc], y=[yc], mode="markers",
        marker=dict(color="#e5484d", size=11, symbol="cross"),
        name="Circle centre",
        hovertext=[f"centre ({xc:.2f}, {yc:.2f}) m, R = {R:.2f} m"],
        hoverinfo="text"))

    F = res["fos_bishop"] if res["method"] == "Bishop" else res["fos_fellenius"]
    fig.update_layout(
        title=f"Critical slip surface - FoS = {F:.3f} ({res['method']})",
        xaxis=dict(title="Distance (m)", range=[xmin, xmax]),
        yaxis=dict(title="Elevation above toe (m)",
                   range=[-D * 1.05, H * 2.2], scaleanchor="x", scaleratio=1),
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.18),
        hovermode="closest")
    return fig


def fig_centres(P, res):
    import plotly.graph_objects as go

    c = res["cloud"]
    xmin, xmax = _view_range(P)
    x = np.linspace(xmin, xmax, 300)
    Fmin = float(np.min(c["F"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ground_y(x, P), mode="lines",
                             line=dict(color="#222222", width=2),
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
        marker=dict(size=16, color="#000000", symbol="star"),
        name="Critical centre"))
    fig.update_layout(
        title="Safety-factor map of trial circle centres",
        xaxis=dict(title="Distance (m)"),
        yaxis=dict(title="Elevation (m)", scaleanchor="x", scaleratio=1),
        height=460, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2))
    return fig


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


def fig_3d(P, res, width=20.0):
    import plotly.graph_objects as go

    D = P["Dmax"]
    xmin, xmax = _view_range(P)
    x = np.linspace(xmin, xmax, 220)
    gy = ground_y(x, P)
    yv = np.array([0.0, float(width)])
    X, Y = np.meshgrid(x, yv)
    fig = go.Figure()

    # layered front and back faces
    for face_y in (0.0, float(width)):
        for i in range(P["nlayers"]):
            upper = np.minimum(gy, P["hi"][i])
            lower = np.full_like(x, max(P["lo"][i], -D))
            upper = np.maximum(upper, lower)
            fig.add_trace(_strip_mesh(
                go, x, upper, lower, face_y, LAYER_COLORS[i % len(LAYER_COLORS)],
                f"Layer {i + 1}", 1.0, showlegend=(face_y == 0.0)))

    # translucent ground surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.tile(gy, (2, 1)), surfacecolor=np.tile(gy, (2, 1)),
        colorscale=[[0, "#cdb48a"], [1, "#8fb36b"]], showscale=False,
        opacity=0.55, name="Ground surface", showlegend=True))

    # water table plane
    if P["water_on"]:
        wy = water_y(x, P)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=np.tile(wy, (2, 1)),
            colorscale=[[0, "#1e90ff"], [1, "#1e90ff"]], showscale=False,
            opacity=0.35, name="Water table", showlegend=True))

    # slip surface
    b = res["best"]
    xe, xi, R = float(b["xe"][0]), float(b["xi"][0]), float(b["R"][0])
    xc, yc = float(b["xc"][0]), float(b["yc"][0])
    sx, sz = arc_xy(xe, xi, xc, yc, R, 80)
    SX, SY = np.meshgrid(sx, yv)
    fig.add_trace(go.Surface(
        x=SX, y=SY, z=np.tile(sz, (2, 1)),
        colorscale=[[0, "#e5484d"], [1, "#e5484d"]], showscale=False,
        opacity=0.85, name="Critical slip surface", showlegend=True))

    # sliding mass highlighted on the front face
    fig.add_trace(_strip_mesh(go, sx, ground_y(sx, P), sz,
                              -0.01 * max(width, 1.0), "#e5484d",
                              "Sliding mass", 0.6, showlegend=True))

    # outlines
    for face_y in (0.0, float(width)):
        fig.add_trace(go.Scatter3d(
            x=x, y=np.full_like(x, face_y), z=gy, mode="lines",
            line=dict(color="#222222", width=5), showlegend=False,
            hoverinfo="skip"))

    F = res["fos_bishop"] if res["method"] == "Bishop" else res["fos_fellenius"]
    fig.update_layout(
        title=f"3D view (2D analysis extruded over {width:g} m) - FoS = {F:.3f}",
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="Distance (m)"),
            yaxis=dict(title="Width (m)"),
            zaxis=dict(title="Elevation (m)"),
            camera=dict(eye=dict(x=1.5, y=-1.9, z=0.9))),
        height=620, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=-0.05))
    return fig


def fig_parametric(df, label, current, ):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["value"], y=df["Bishop"], mode="lines+markers",
                             name="Bishop", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["value"], y=df["Fellenius"], mode="lines+markers",
                             name="Fellenius", line=dict(width=2, dash="dash")))
    fig.add_hline(y=1.0, line=dict(color="#e5484d", dash="dot"),
                  annotation_text="FoS = 1.0 (failure)")
    fig.add_hline(y=1.5, line=dict(color="#2e9e5b", dash="dot"),
                  annotation_text="FoS = 1.5 (target)")
    if current is not None:
        fig.add_vline(x=current, line=dict(color="#888888", dash="dash"),
                      annotation_text="current")
    fig.update_layout(
        title=f"Factor of safety vs {label}",
        xaxis=dict(title=label), yaxis=dict(title="Factor of safety"),
        height=440, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2))
    return fig


def fig_sites_map(df, zoom=11):
    import plotly.express as px

    d = df.copy()
    d["size"] = 14
    kwargs = dict(lat="Lat", lon="Lon", color="Status",
                  color_discrete_map=STATUS_COLORS, hover_name="Site",
                  hover_data={"FoS": ":.2f", "Lat": ":.5f", "Lon": ":.5f",
                              "size": False, "Status": True},
                  size="size", size_max=16, zoom=zoom, height=460)
    try:
        fig = px.scatter_map(d, map_style="open-street-map", **kwargs)
    except AttributeError:  # older plotly
        fig = px.scatter_mapbox(d, mapbox_style="open-street-map", **kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
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
# 7. STREAMLIT APP
# =====================================================================
def _cache(fn):
    if st is None:
        return fn
    return st.cache_data(show_spinner=False)(fn)


@_cache
def cached_search(M_, nsl_, method_, density_):
    return search(prepare(M_), nsl=nsl_, method=method_, density=density_)


def main():
    st.set_page_config(page_title="SlopeLab - Slope Stability Studio",
                       page_icon="⛰️", layout="wide")

    def show_plot(fig, key=None):
        try:
            st.plotly_chart(fig, width="stretch", key=key)
        except TypeError:
            st.plotly_chart(fig, use_container_width=True, key=key)

    def show_df(df):
        try:
            st.dataframe(df, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(df, use_container_width=True, hide_index=True)

    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("sites", [])
    S = st.session_state

    def apply_preset():
        name = S["preset"]
        if name in PRESETS:
            for k, v in PRESETS[name].items():
                S[k] = v

    st.markdown(
        """
        <div style="padding:14px 18px;border-radius:12px;
        background:linear-gradient(90deg,#1f6feb 0%,#2e9e5b 100%);color:white;">
        <div style="font-size:1.6rem;font-weight:700;">⛰️ SlopeLab</div>
        <div style="opacity:.9;">Slope stability studio - layered soils, water table,
        critical-circle search, 3D view and site map</div></div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Model")
        st.selectbox("Load an example", ["-"] + list(PRESETS.keys()),
                     key="preset", on_change=apply_preset)

        st.subheader("Geometry")
        c1, c2 = st.columns(2)
        c1.number_input("Height H (m)", min_value=1.0, max_value=100.0,
                        step=0.5, key="H")
        c2.number_input("Angle (deg)", min_value=10.0, max_value=80.0,
                        step=1.0, key="beta")
        st.number_input("Model depth below toe (x H)", min_value=0.5,
                        max_value=6.0, step=0.5, key="Dfac")

        st.subheader("Soil layers")
        st.selectbox("Number of layers", [1, 2, 3, 4], key="nlayers")
        for i in range(1, int(S["nlayers"]) + 1):
            last = i == int(S["nlayers"])
            with st.expander(f"Layer {i}", expanded=(i == 1)):
                if not last:
                    st.number_input("Thickness (m)", min_value=0.1,
                                    max_value=100.0, step=0.5, key=f"L{i}_t")
                else:
                    st.caption("Extends down to the model base")
                a, b_ = st.columns(2)
                a.number_input("Unit weight (kN/m3)", min_value=10.0,
                               max_value=26.0, step=0.5, key=f"L{i}_gam")
                b_.number_input("Saturated (kN/m3)", min_value=10.0,
                                max_value=26.0, step=0.5, key=f"L{i}_gsat")
                a2, b2 = st.columns(2)
                a2.number_input("c' (kPa)", min_value=0.0, max_value=500.0,
                                step=1.0, key=f"L{i}_c")
                b2.number_input("phi' (deg)", min_value=0.0, max_value=50.0,
                                step=1.0, key=f"L{i}_phi")

        st.subheader("Water table")
        st.checkbox("Include water table", key="water_on")
        if S["water_on"]:
            w1, w2 = st.columns(2)
            w1.number_input("At toe (m)", min_value=-50.0, max_value=100.0,
                            step=0.5, key="hw_toe")
            w2.number_input("At crest (m)", min_value=-50.0, max_value=100.0,
                            step=0.5, key="hw_crest")
            st.caption("Levels are measured up from the toe. Pore pressure is "
                       "taken as hydrostatic below this line (conservative).")

        st.subheader("Analysis")
        st.radio("Method", ["Bishop", "Fellenius"], key="method", horizontal=True)
        st.select_slider("Search density", ["Fast", "Standard", "Fine"],
                         key="density")
        st.slider("Number of slices", 15, 60, key="nsl")

        st.subheader("Site")
        st.text_input("Site name", key="site_name")
        l1, l2 = st.columns(2)
        l1.number_input("Latitude", min_value=-90.0, max_value=90.0,
                        step=0.0001, format="%.5f", key="lat")
        l2.number_input("Longitude", min_value=-180.0, max_value=180.0,
                        step=0.0001, format="%.5f", key="lon")
        st.number_input("3D block width (m)", min_value=2.0, max_value=200.0,
                        step=1.0, key="width")

    # --------------------------------------------------------- analysis
    for i in range(1, int(S["nlayers"]) + 1):
        if S[f"L{i}_gsat"] < S[f"L{i}_gam"]:
            st.warning(f"Layer {i}: saturated unit weight is below the "
                       "unit weight. Check your inputs.")

    M = model_from_state(S)

    with st.spinner("Searching for the critical slip surface..."):
        res = cached_search(M, int(S["nsl"]), S["method"], S["density"])

    if res is None:
        st.error("No valid slip surface was found for this model. Try a larger "
                 "model depth, a steeper slope, or weaker soil.")
        return

    P = prepare(M)
    F = res["fos_bishop"] if S["method"] == "Bishop" else res["fos_fellenius"]
    label, kind = status_of(F)

    m1, m2, m3 = st.columns(3)
    m1.metric("FoS - Bishop", f"{res['fos_bishop']:.3f}")
    m2.metric("FoS - Fellenius", f"{res['fos_fellenius']:.3f}")
    m3.metric("Status", label)
    getattr(st, kind)(f"{S['method']}: factor of safety {F:.3f} - {label}")

    tabs = st.tabs(["2D section", "3D view", "Parametric study",
                    "Sites map", "About"])

    # ------------------------------------------------------------ 2D tab
    with tabs[0]:
        o1, o2 = st.columns(2)
        show_sl = o1.checkbox("Show slices", value=True)
        show_tr = o2.checkbox("Show other trial circles", value=True)
        show_plot(fig_section(P, res, show_sl, show_tr), key="fig2d")
        with st.expander("Safety-factor map of trial circle centres"):
            show_plot(fig_centres(P, res), key="figmap")
        st.subheader("Results")
        tbl = results_table(P, res)
        show_df(tbl)
        st.download_button("Download results (CSV)",
                           tbl.to_csv(index=False).encode("utf-8"),
                           "slope_results.csv", "text/csv")

    # ------------------------------------------------------------ 3D tab
    with tabs[1]:
        show_plot(fig_3d(P, res, float(S["width"])), key="fig3d")
        st.caption("Drag to rotate, scroll to zoom. This is a 3D visualisation "
                   "of the 2D (plane-strain) analysis extruded across the "
                   "block width. It is not a full 3D stability analysis.")

    # -------------------------------------------------- parametric tab
    with tabs[2]:
        label_p = st.selectbox("Parameter to vary", list(PARAMS.keys()))
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
        if st.button("Run parametric study"):
            if v_hi <= v_lo:
                st.error("'To' must be larger than 'From'.")
            else:
                with st.spinner("Running analyses..."):
                    vals = np.linspace(v_lo, v_hi, int(n_pts))
                    df = run_parametric(M, key_p, vals)
                S["param_result"] = dict(label=label_p, key=key_p, df=df,
                                         cur=cur)
        pr = S.get("param_result")
        if pr is not None:
            show_plot(fig_parametric(pr["df"], pr["label"], pr["cur"]),
                      key="figpar")
            show_df(pr["df"].rename(columns={"value": pr["label"]}).round(3))
            st.download_button("Download study (CSV)",
                               pr["df"].to_csv(index=False).encode("utf-8"),
                               "parametric_study.csv", "text/csv")
            st.caption("Uses the Fast search. Re-run after changing the model.")

    # ---------------------------------------------------------- map tab
    with tabs[3]:
        st.write("Save this analysis against the site coordinates in the "
                 "sidebar. Tip: long-press a spot in Google Maps to read its "
                 "latitude and longitude.")
        if st.button("Save this site"):
            S["sites"].append(dict(
                Site=S["site_name"], Lat=float(S["lat"]), Lon=float(S["lon"]),
                H=float(S["H"]), Angle=float(S["beta"]),
                Layers=int(S["nlayers"]), Water=bool(S["water_on"]),
                Method=S["method"], FoS=round(float(F), 3), Status=label))
            st.success(f"Saved {S['site_name']}.")
        up = st.file_uploader("Load sites from a CSV file", type="csv")
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

        saved = pd.DataFrame(S["sites"]) if S["sites"] else pd.DataFrame(
            columns=["Site", "Lat", "Lon", "FoS", "Status"])
        current = pd.DataFrame([dict(
            Site=f"{S['site_name']} (current)", Lat=float(S["lat"]),
            Lon=float(S["lon"]), FoS=float(F), Status="Current (unsaved)")])
        both = pd.concat([saved[["Site", "Lat", "Lon", "FoS", "Status"]],
                          current], ignore_index=True)
        try:
            show_plot(fig_sites_map(both), key="figsites")
        except Exception:  # noqa: BLE001
            st.map(both.rename(columns={"Lat": "latitude", "Lon": "longitude"}))
        if len(saved):
            show_df(saved)
            st.download_button("Download sites (CSV)",
                               saved.to_csv(index=False).encode("utf-8"),
                               "slope_sites.csv", "text/csv")
            if st.button("Clear saved sites"):
                S["sites"] = []
                st.rerun()
        st.caption("Saved sites last for this browser session. Download the "
                   "CSV to keep them, and upload it later to restore them.")

    # ---------------------------------------------------------- about tab
    with tabs[4]:
        st.markdown(
            """
**Method.** Limit-equilibrium method of slices on circular slip surfaces.
The critical circle is found by a grid search over exit point, entry point and
radius, followed by local refinement.
            """)
        st.latex(r"F_{Bishop}=\frac{\sum\frac{c'b+(W-ub)\tan\phi'}{m_\alpha}}"
                 r"{\sum W\sin\alpha},\quad "
                 r"m_\alpha=\cos\alpha\left(1+\frac{\tan\alpha\tan\phi'}{F}\right)")
        st.latex(r"F_{Fellenius}=\frac{\sum\left[c'l+(W\cos\alpha-ul)"
                 r"\tan\phi'\right]}{\sum W\sin\alpha}")
        st.markdown(
            """
**Assumptions and limits**
- Homogeneous horizontal layers; 2D plane strain; circular failure only.
- Water table is a straight line; pore pressure at each slice base is
  hydrostatic from that line (conservative compared with a seepage analysis).
- Use effective-stress parameters (c', phi') when a water table is present. For
  a quick undrained check use phi = 0 with the water table off.
- No earthquake, surcharge or reinforcement loads yet.
- Always verify against hand calculations or commercial software (for example
  GEO5, Slide, GeoStudio) before using results for design.

**Validation example.** Choose "Check: Taylor chart" in the sidebar. Taylor's
chart for phi = 0 and a 45 degree slope gives FoS = c / (0.181 x gamma x H)
= 0.77.
            """)


if __name__ == "__main__":
    main()
