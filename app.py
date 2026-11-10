"""SlopeLab — dark multi-worksheet slope stability dashboard.

Run with: streamlit run app.py
Requires slope_app.py from the original SlopeLab project for real calculations.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# The calculation engine belongs to the original SlopeLab project.
try:
    from slope_app import cached_search, fig_section, prepare, status_of
    ENGINE_AVAILABLE = True
    ENGINE_ERROR = ""
except Exception as exc:  # Keep the dashboard usable, but never fake results.
    ENGINE_AVAILABLE = False
    ENGINE_ERROR = str(exc)
    cached_search = fig_section = prepare = status_of = None

DB_PATH = Path(__file__).with_name("slopelab_workspaces.sqlite3")
DEFAULT_INPUTS = {
    # A new worksheet is deliberately blank: users must enter project data.
    "height": None,
    "angle": None,
    "depth_factor": 2.0,
    "unit_weight": None,
    "sat_weight": None,
    "cohesion": None,
    "friction": None,
    "water_enabled": False,
    "water_toe": None,
    "water_crest": None,
    "method": "Bishop",
    "slices": 30,
    "density": "Standard",
}

st.set_page_config(page_title="SlopeLab | Slope Stability", page_icon="⛰️", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(circle at 18% 0%, #102542 0, #071322 38%, #050d19 100%); color:#e8f1ff; }
[data-testid="stHeader"] { background:rgba(4,12,24,.92); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#08172b,#0b1d34); border-right:1px solid #203957; }
.block-container { padding-top:1.25rem; max-width:1600px; }
.hero { padding:20px 24px; border-radius:15px; color:white; background:linear-gradient(105deg,#0756bd,#087f9b 55%,#08a879); box-shadow:0 8px 28px #00152c88; margin-bottom:18px; }
.hero h1 { margin:0; font-size:1.8rem; font-weight:800; }
.hero p { margin:5px 0 0 0; opacity:.9; }
.panel { background:linear-gradient(145deg,#101f34,#0a1728); border:1px solid #29415f; border-radius:13px; padding:16px 18px; margin-bottom:14px; box-shadow:0 6px 18px #0003; }
.panel h3 { color:#eaf4ff; margin:0 0 12px 0; font-size:1.08rem; }
.metric { background:#0b1c2d; border:1px solid #28425f; border-radius:11px; padding:13px 15px; min-height:95px; }
.metric-label { color:#9eb5d0; font-size:.82rem; }
.metric-value { color:#f5f9ff; font-size:1.8rem; font-weight:800; margin-top:4px; }
.good { border-color:#08b981; background:linear-gradient(145deg,#062c2a,#0b1c2d); }
.good .metric-value { color:#20d99b; }
.bad { border-color:#ef5350; background:linear-gradient(145deg,#35191f,#0b1c2d); }
.bad .metric-value { color:#ff7777; }
.muted { color:#9eb5d0; font-size:.86rem; }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#08b96f,#00a9a5); border:1px solid #27e6b2; color:#041a1d; font-weight:800; min-height:3.2rem; font-size:1.1rem; border-radius:10px; }
div.stButton > button { border-radius:9px; }
.stTabs [data-baseweb="tab-list"] { gap:7px; }
.stTabs [data-baseweb="tab"] { background:#14253b; border:1px solid #29415f; border-radius:9px; padding:10px 18px; }
.stTabs [aria-selected="true"] { background:#075ac4 !important; }
[data-testid="stNumberInput"] input { background:#13243a; }
hr { border-color:#233b56; }
</style>
""", unsafe_allow_html=True)


def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS app_state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
    return con


def fresh_book():
    return {"Worksheet 1": {"inputs": DEFAULT_INPUTS.copy(), "result": None}}


def load_book():
    try:
        with db_connect() as con:
            row = con.execute("SELECT payload FROM app_state WHERE id=1").fetchone()
        if row:
            book = json.loads(row[0])
            if isinstance(book, dict) and book:
                return book
    except Exception:
        pass
    return fresh_book()


def save_book():
    def json_fallback(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)
    payload = json.dumps(st.session_state.book, default=json_fallback)
    with db_connect() as con:
        con.execute("INSERT INTO app_state(id,payload) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (payload,))


def new_name(book):
    i = 1
    while f"Worksheet {i}" in book:
        i += 1
    return f"Worksheet {i}"


def make_model(v):
    return {
        "H": float(v["height"]),
        "beta": float(v["angle"]),
        "Dfac": float(v["depth_factor"]),
        "layers": [{"t": 1.0, "gam": float(v["unit_weight"]), "gsat": float(v["sat_weight"]),
                    "c": float(v["cohesion"]), "phi": float(v["friction"])}],
        "water_on": bool(v["water_enabled"]),
        "hw_toe": float(v["water_toe"]),
        "hw_crest": float(v["water_crest"]),
    }


def geometry_figure(v):
    h = float(v["height"] or 0.0)
    beta = float(v["angle"] or 30.0)
    run = h / max(np.tan(np.radians(beta)), 0.01) if h > 0 else 0.0
    crest = run + h * 1.25 if h > 0 else 20.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, crest + 8, crest + 8, run, 0, 0], y=[0, 0, h, h, 0, 0],
                             fill="toself", fillcolor="rgba(241,190,94,.23)",
                             line=dict(color="#f2c36b", width=2), name="Slope mass"))
    fig.add_trace(go.Scatter(x=[0, run, crest + 8], y=[0, h, h], mode="lines",
                             line=dict(color="#f2c36b", width=4), name="Ground profile"))
    if v["water_enabled"] and v["water_toe"] is not None and v["water_crest"] is not None:
        fig.add_trace(go.Scatter(x=[0, crest + 8], y=[float(v["water_toe"]), float(v["water_crest"])],
                                 mode="lines", line=dict(color="#35b9ff", dash="dash", width=2), name="Water table"))
    fig.update_layout(height=330, margin=dict(l=10,r=10,t=12,b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#091729", font_color="#dbeaff", xaxis_title="Distance (m)",
                      yaxis_title="Elevation (m)", xaxis=dict(gridcolor="#263c56"),
                      yaxis=dict(gridcolor="#263c56", rangemode="tozero"), legend=dict(orientation="h", y=1.12))
    return fig


def render_results(ws):
    result = ws.get("result")
    if not result:
        st.markdown('<div class="panel"><h3>📊 Analysis Results</h3><div class="muted">No analysis has been run for this worksheet yet. Enter your soil and geometry data, then select <b>Run Analysis</b>.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><h3>💡 Key Findings</h3><div class="muted">The critical slip surface, factor of safety and recommendations will appear here after a successful calculation.</div></div>', unsafe_allow_html=True)
        return
    fos_b = float(result["fos_bishop"])
    fos_f = float(result["fos_fellenius"])
    selected = fos_b if result["method"] == "Bishop" else fos_f
    status_text = "STABLE" if selected >= 1.5 else ("MARGINAL" if selected >= 1.0 else "UNSTABLE")
    cls = "good" if selected >= 1.5 else "bad"
    st.markdown(f'<div class="panel {cls}"><h3>📊 Analysis Results</h3><div class="metric-label">Factor of Safety · {result["method"]}</div><div class="metric-value">{selected:.3f}</div><div style="font-weight:800;color:{"#20d99b" if selected>=1.5 else "#ff7777"};margin-top:5px">● Slope is {status_text}</div><hr><div class="muted">Bishop: {fos_b:.3f} &nbsp; | &nbsp; Fellenius: {fos_f:.3f}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3>💡 Key Findings</h3>'
                f'<div>Selected method: <b>{result["method"]}</b></div>'
                f'<div>Calculated factor of safety: <b>{selected:.3f}</b></div>'
                f'<div>Screening interpretation: <b>{status_text}</b></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3>🛡️ Safety Guidelines</h3><div class="muted">FOS below 1.0: calculated instability<br>1.0–1.5: review design criteria and conditions<br>Above 1.5: commonly treated as stable for preliminary screening; confirm project requirements.</div></div>', unsafe_allow_html=True)
    st.warning("Engineering screening only. Verify assumptions, groundwater, loading, stratigraphy and design criteria with a qualified geotechnical engineer.")


if "book" not in st.session_state:
    st.session_state.book = load_book()
if "current" not in st.session_state or st.session_state.current not in st.session_state.book:
    st.session_state.current = next(iter(st.session_state.book))

book = st.session_state.book
current = st.session_state.current

st.markdown('<div class="hero"><h1>⛰️ SlopeLab</h1><p>Slope Stability Studio · Layered soils · Groundwater · Bishop & Fellenius analysis</p></div>', unsafe_allow_html=True)

# Worksheet navigation and actions
nav_cols = st.columns([1,1,1,1,1.35,1.1])
ws_names = list(book.keys())
for idx in range(4):
    with nav_cols[idx]:
        name = f"Worksheet {idx+1}"
        if name in book:
            if st.button(f"▤ {name}", key=f"open_{idx}", use_container_width=True,
                         type="primary" if current == name else "secondary"):
                st.session_state.current = name
                st.rerun()
        else:
            st.button(f"＋ Worksheet {idx+1}", key=f"empty_{idx}", disabled=True, use_container_width=True)
with nav_cols[4]:
    if st.button("＋ New Worksheet", type="primary", use_container_width=True):
        name = new_name(book)
        book[name] = {"inputs": DEFAULT_INPUTS.copy(), "result": None}
        st.session_state.current = name
        save_book()
        st.rerun()
with nav_cols[5]:
    if st.button("🗑 Delete", disabled=len(book) <= 1, use_container_width=True):
        del book[current]
        st.session_state.current = next(iter(book))
        save_book()
        st.rerun()

ws = book[current]
values = ws.setdefault("inputs", DEFAULT_INPUTS.copy())
values = {**DEFAULT_INPUTS, **values}
left, right = st.columns([2.15, 1], gap="large")

with left:
    st.markdown(f'<div class="panel"><h3>✏️ {current} — Model Parameters</h3><div class="muted">Enter the slope geometry and soil properties. New worksheets start with editable default values; clear or replace them for your project.</div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Geometry")
        g1, g2, g3 = st.columns(3)
        values["height"] = g1.number_input("Slope height H (m)", min_value=1.0, max_value=100.0, value=values["height"], step=.5, key=f"h_{current}")
        values["angle"] = g2.number_input("Slope angle β (°)", min_value=10.0, max_value=80.0, value=values["angle"], step=1.0, key=f"a_{current}")
        values["depth_factor"] = g3.number_input("Search depth below toe (×H)", .5, 6.0, float(values["depth_factor"]), .5, key=f"d_{current}")
        st.markdown("#### Soil Properties")
        s1, s2, s3, s4 = st.columns(4)
        values["unit_weight"] = s1.number_input("Unit weight γ (kN/m³)", min_value=10.0, max_value=26.0, value=values["unit_weight"], step=.5, key=f"uw_{current}")
        values["sat_weight"] = s2.number_input("Saturated γsat (kN/m³)", min_value=10.0, max_value=26.0, value=values["sat_weight"], step=.5, key=f"sw_{current}")
        values["cohesion"] = s3.number_input("Cohesion c′ (kPa)", min_value=0.0, max_value=500.0, value=values["cohesion"], step=1.0, key=f"c_{current}")
        values["friction"] = s4.number_input("Friction angle φ′ (°)", min_value=0.0, max_value=50.0, value=values["friction"], step=1.0, key=f"phi_{current}")
        st.markdown("#### Groundwater")
        values["water_enabled"] = st.checkbox("Include water table", value=bool(values["water_enabled"]), key=f"water_{current}")
        w1, w2 = st.columns(2)
        values["water_toe"] = w1.number_input("Water level at toe (m)", min_value=0.0, max_value=100.0, value=values["water_toe"], step=.5, disabled=not values["water_enabled"], key=f"wt_{current}")
        values["water_crest"] = w2.number_input("Water level at crest (m)", min_value=0.0, max_value=100.0, value=values["water_crest"], step=.5, disabled=not values["water_enabled"], key=f"wc_{current}")
        st.markdown("#### Analysis Settings")
        a1, a2, a3 = st.columns(3)
        values["method"] = a1.selectbox("Primary method", ["Bishop", "Fellenius"], index=["Bishop", "Fellenius"].index(values["method"]), key=f"method_{current}")
        values["slices"] = a2.slider("Number of slices", 15, 60, int(values["slices"]), key=f"slices_{current}")
        values["density"] = a3.selectbox("Search density", ["Fast", "Standard", "Fine"], index=["Fast", "Standard", "Fine"].index(values["density"]), key=f"density_{current}")
    st.markdown('<div class="panel"><h3>📐 Slope Geometry</h3><div class="muted">Preview updates as geometry values change.</div></div>', unsafe_allow_html=True)
    st.plotly_chart(geometry_figure(values), use_container_width=True, key=f"geometry_{current}")
    if st.button("▶  Run Analysis", type="primary", use_container_width=True, key=f"run_{current}"):
        missing = [label for key, label in [("height", "slope height"), ("angle", "slope angle"), ("unit_weight", "unit weight"), ("sat_weight", "saturated unit weight"), ("cohesion", "cohesion"), ("friction", "friction angle")] if values.get(key) is None]
        if missing:
            st.error("Complete the required inputs before analysis: " + ", ".join(missing) + ".")
        elif not ENGINE_AVAILABLE:
            st.error("The calculation engine could not be loaded. Put the original slope_app.py file in the same folder as app.py. Details: " + ENGINE_ERROR)
        else:
            try:
                model = make_model(values)
                with st.spinner("Searching for the critical slip surface and calculating factors of safety…"):
                    result = cached_search(model, int(values["slices"]), "Bishop", values["density"])
                if result is None:
                    st.error("No valid slip surface was found. Review the model inputs and search settings.")
                else:
                    ws["result"] = {"fos_bishop": float(result["fos_bishop"]),
                                     "fos_fellenius": float(result["fos_fellenius"]),
                                     "method": values["method"], "model": model, "raw": result}
                    save_book()
                    st.success("Analysis completed. Results are shown in the panel.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
    # Persist all current input values on every rerun.
    ws["inputs"] = values
    save_book()

with right:
    render_results(ws)
    if ws.get("result") and ENGINE_AVAILABLE:
        try:
            st.markdown('<div class="panel"><h3>🔎 Critical Slip Surface</h3></div>', unsafe_allow_html=True)
            st.plotly_chart(fig_section(prepare(ws["result"]["model"]), ws["result"]["raw"], True, True),
                            use_container_width=True, key=f"critical_{current}")
        except Exception as exc:
            st.caption(f"Slip-surface plot unavailable: {exc}")

st.caption("Local SQLite storage keeps worksheets on this server between app restarts. For multi-user cloud deployment, configure a persistent database and per-user authentication. Engineering outputs must be independently verified before design use.")

if not ENGINE_AVAILABLE:
    st.info("Dashboard preview mode: the interface works, but real factor-of-safety calculations require the original slope_app.py calculation engine.")
