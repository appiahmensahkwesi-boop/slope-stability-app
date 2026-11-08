"""
Slope Stability Analysis Tool
Limit equilibrium: Ordinary (Fellenius) and Bishop's Simplified methods
Homogeneous soil, circular slip surface, 2D.

Geometry: toe at (0, 0). Ground is flat at y = 0 to the left of the toe,
slope rises to the right to a crest at height H, then flat at y = H.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# CORE ENGINEERING FUNCTIONS
# ------------------------------------------------------------------
def ground_y(x, H, beta_deg):
    """Ground surface elevation at x."""
    x_crest = H / np.tan(np.radians(beta_deg))
    return np.clip(x * np.tan(np.radians(beta_deg)), 0.0, H)


def build_slices(xc, yc, R, H, beta_deg, n_slices):
    """Return slice data for a trial circle, or None if circle is invalid."""
    xs = np.linspace(xc - R, xc + R, 2000)
    arc_y = yc - np.sqrt(np.maximum(R**2 - (xs - xc) ** 2, 0))
    gy = ground_y(xs, H, beta_deg)
    inside = gy > arc_y + 1e-9          # soil lies above the arc
    if inside.sum() < 5:
        return None
    x_left, x_right = xs[inside].min(), xs[inside].max()
    if x_right - x_left < 1e-6:
        return None

    b = (x_right - x_left) / n_slices
    xm = x_left + b * (np.arange(n_slices) + 0.5)
    arc_m = yc - np.sqrt(np.maximum(R**2 - (xm - xc) ** 2, 0))
    h = ground_y(xm, H, beta_deg) - arc_m
    h = np.maximum(h, 0)
    sin_a = np.clip((xm - xc) / R, -0.999, 0.999)
    alpha = np.arcsin(sin_a)
    return b, h, alpha


def fos_fellenius(b, h, alpha, gamma, c, phi_deg, ru):
    phi = np.radians(phi_deg)
    W = gamma * h * b
    l = b / np.cos(alpha)
    u = ru * gamma * h
    resisting = c * l + (W * np.cos(alpha) - u * l) * np.tan(phi)
    driving = np.sum(W * np.sin(alpha))
    if driving <= 0:
        return np.inf
    return np.sum(resisting) / driving


def fos_bishop(b, h, alpha, gamma, c, phi_deg, ru, tol=1e-4, max_iter=100):
    phi = np.radians(phi_deg)
    W = gamma * h * b
    u = ru * gamma * h
    driving = np.sum(W * np.sin(alpha))
    if driving <= 0:
        return np.inf
    F = fos_fellenius(b, h, alpha, gamma, c, phi_deg, ru)  # starting guess
    if not np.isfinite(F):
        F = 1.0
    for _ in range(max_iter):
        m_alpha = np.cos(alpha) * (1 + np.tan(alpha) * np.tan(phi) / F)
        num = np.sum((c * b + (W - u * b) * np.tan(phi)) / m_alpha)
        F_new = num / driving
        if abs(F_new - F) < tol:
            return F_new
        F = F_new
    return F


def find_critical_circle(H, beta_deg, gamma, c, phi_deg, ru, method="Bishop",
                         n_slices=30, n_grid=14):
    """Grid search for the circle with the lowest FoS."""
    x_crest = H / np.tan(np.radians(beta_deg))
    best = {"fos": np.inf}
    xc_vals = np.linspace(-0.5 * H, x_crest + 0.5 * H, n_grid)
    yc_vals = np.linspace(1.0 * H, 3.0 * H, n_grid)
    r_factors = np.linspace(1.0, 1.8, 9)

    func = fos_bishop if method == "Bishop" else fos_fellenius
    for xc in xc_vals:
        for yc in yc_vals:
            r_toe = np.hypot(xc - 0.0, yc - 0.0)   # circle through the toe
            for rf in r_factors:
                R = r_toe * rf
                data = build_slices(xc, yc, R, H, beta_deg, n_slices)
                if data is None:
                    continue
                b, h, alpha = data
                F = func(b, h, alpha, gamma, c, phi_deg, ru)
                if np.isfinite(F) and 0 < F < best["fos"]:
                    best = {"fos": F, "xc": xc, "yc": yc, "R": R,
                            "b": b, "h": h, "alpha": alpha}
    return best


def plot_result(H, beta_deg, best):
    x_crest = H / np.tan(np.radians(beta_deg))
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.linspace(-0.8 * H, x_crest + 1.2 * H, 400)
    ax.fill_between(xs, ground_y(xs, H, beta_deg), -0.6 * H,
                    color="#d9c3a0", label="Soil")
    ax.plot(xs, ground_y(xs, H, beta_deg), "k", lw=2)
    if "xc" in best:
        xc, yc, R = best["xc"], best["yc"], best["R"]
        th = np.linspace(np.pi, 2 * np.pi, 300)
        ax.plot(xc + R * np.cos(th), yc + R * np.sin(th), "r--", lw=2,
                label="Critical slip circle")
        ax.plot(xc, yc, "r+", ms=10)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(-0.6 * H, 1.6 * H)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig


# ------------------------------------------------------------------
# APP (Streamlit)  -- run with:  streamlit run slope_app.py
# ------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="Slope Stability Tool", layout="centered")
    st.title("Slope Stability Analysis Tool")
    st.caption("Method of slices: Fellenius and Bishop's Simplified | "
               "homogeneous soil, circular failure")

    st.subheader("Inputs")
    col1, col2 = st.columns(2)
    H = col1.number_input("Slope height H (m)", 1.0, 100.0, 10.0, 0.5)
    beta = col2.number_input("Slope angle (degrees)", 10.0, 80.0, 30.0, 1.0)
    gamma = col1.number_input("Unit weight (kN/m3)", 10.0, 25.0, 18.0, 0.5)
    c = col2.number_input("Cohesion c' (kPa)", 0.0, 200.0, 15.0, 1.0)
    phi = col1.number_input("Friction angle (degrees)", 0.0, 45.0, 20.0, 1.0)
    ru = col2.number_input("Pore pressure ratio ru", 0.0, 0.6, 0.0, 0.05)
    method = st.radio("Method", ["Bishop", "Fellenius"], horizontal=True)

    if st.button("Analyse"):
        with st.spinner("Searching for critical slip circle..."):
            best = find_critical_circle(H, beta, gamma, c, phi, ru, method)
        if not np.isfinite(best["fos"]):
            st.error("No valid slip circle found. Check your inputs.")
            return

        F = best["fos"]
        st.subheader("Result")
        st.metric(f"Minimum factor of safety ({method})", f"{F:.3f}")
        if F < 1.0:
            st.error("UNSTABLE (FoS < 1.0)")
        elif F < 1.3:
            st.warning("MARGINAL (FoS between 1.0 and 1.3)")
        elif F < 1.5:
            st.info("ACCEPTABLE for temporary works (FoS 1.3 to 1.5)")
        else:
            st.success("STABLE (FoS >= 1.5)")

        st.write(f"Critical circle: centre ({best['xc']:.2f}, {best['yc']:.2f}) m, "
                 f"radius {best['R']:.2f} m")
        st.pyplot(plot_result(H, beta, best))


if __name__ == "__main__":
    main()
