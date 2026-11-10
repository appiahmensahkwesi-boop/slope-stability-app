"""SlopeLab — dark multi-worksheet slope stability dashboard.

Run with: streamlit run app.py

Requires slope_app.py from the original SlopeLab project for real calculations.
The dashboard remains usable in preview mode if that calculation engine is absent.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# The calculation engine belongs to the original SlopeLab project.
try:
    from slope_app import cached_search, fig_section, prepare
    ENGINE_AVAILABLE = True
    ENGINE_ERROR = ""
except Exception as exc:
    ENGINE_AVAILABLE = False
    ENGINE_ERROR = str(exc)
    cached_search = fig_section = prepare = None


DB_PATH = Path(__file__).with_name("slopelab_workspaces.sqlite3")

# New worksheets intentionally start blank for project-specific soil data.
DEFAULT_INPUTS = {
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

st.set_page_config(
    page_title="SlopeLab | Slope Stability",
    page_icon="⛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(circle at 18% 0%, #102542 0, #071322 38%, #050d19 100%);
        color: #e8f1ff;
    }
    [data-testid="stHeader"] { background: rgba(4,12,24,.92); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#08172b,#0b1d34);
        border-right: 1px solid #203957;
    }
    .block-container { padding-top: 1.25rem; max-width: 1600px; }
    .hero {
        padding: 20px 24px; border-radius: 15px; color: white;
        background: linear-gradient(105deg,#0756bd,#087f9b 55%,#08a879);
        box-shadow: 0 8px 28px #00152c88; margin-bottom: 18px;
    }
    .hero h1 { margin: 0; font-size: 1.8rem; font-weight: 800; }
    .hero p { margin: 5px 0 0 0; opacity: .9; }
    .panel {
        background: linear-gradient(145deg,#101f34,#0a1728);
        border: 1px solid #29415f; border-radius: 13px;
        padding: 16px 18px; margin-bottom: 14px;
        box-shadow: 0 6px 18px #0003;
    }
    .panel h3 { color: #eaf4ff; margin: 0 0 12px 0; font-size: 1.08rem; }
    .metric {
        background: #0b1c2d; border: 1px solid #28425f;
        border-radius: 11px; padding: 13px 15px; min-height: 95px;
    }
    .metric-label { color: #9eb5d0; font-size: .82rem; }
    .metric-value { color: #f5f9ff; font-size: 1.8rem; font-weight: 800; margin-top: 4px; }
    .good { border-color: #08b981; background: linear-gradient(145deg,#062c2a,#0b1c2d); }
    .good .metric-value { color: #20d99b; }
    .bad { border-color: #ef5350; background: linear-gradient(145deg,#35191f,#0b1c2d); }
    .bad .metric-value { color: #ff7777; }
    .muted { color: #9eb5d0; font-size: .86rem; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg,#08b96f,#00a9a5);
        border: 1px solid #27e6b2; color: #041a1d;
        font-weight: 800; min-height: 3.2rem;
        font-size: 1.05rem; border-radius: 10px;
    }
    div.stButton > button { border-radius: 9px; }
    .stTabs [data-baseweb="tab-list"] { gap: 7px; }
    .stTabs [data-baseweb="tab"] {
        background: #14253b; border: 1px solid #29415f;
        border-radius: 9px; padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] { background: #075ac4 !important; }
    [data-testid="stNumberInput"] input { background: #13243a; }
    hr { border-color: #233b56; }
    </style>
    """,
    unsafe_allow_html=True,
)


def db_connect():
    """Open the local worksheet database and ensure its table exists."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            id INTEGER PRIMARY KEY CHECK(id=1),
            payload TEXT NOT NULL
        )
        """
    )
    return con


def fresh_book():
    return {
        "Worksheet 1": {
            "inputs": DEFAULT_INPUTS.copy(),
            "result": None,
        }
    }


def load_book():
    try:
        with db_connect() as con:
            row = con.execute(
                "SELECT payload FROM app_state WHERE id=1"
            ).fetchone()

        if row:
            loaded = json.loads(row[0])
            if isinstance(loaded, dict) and loaded:
                # Add any newly introduced input fields to older saved worksheets.
                for worksheet in loaded.values():
                    worksheet.setdefault("inputs", {})
                    worksheet["inputs"] = {
                        **DEFAULT_INPUTS,
                        **worksheet["inputs"],
                    }
                    worksheet.setdefault("result", None)
                return loaded
    except Exception:
        pass

    return fresh_book()


def save_book():
    """Persist worksheet inputs/results; convert NumPy values to JSON-safe values."""
    def json_fallback(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    payload = json.dumps(st.session_state.book, default=json_fallback)
    with db_connect() as con:
        con.execute(
            """
            INSERT INTO app_state(id, payload) VALUES(1, ?)
            ON CONFLICT(id) DO UPDATE SET payload=excluded.payload
            """,
            (payload,),
        )


def new_name(book):
    index = 1
    while f"Worksheet {index}" in book:
        index += 1
    return f"Worksheet {index}"


def make_model(v):
    """Convert worksheet inputs into the model format expected by slope_app.py."""
    water_on = bool(v["water_enabled"])

    # When groundwater is disabled, levels are not used by the calculation engine.
    # Use zero values rather than calling float(None), which causes a TypeError.
    water_toe = float(v["water_toe"]) if water_on and v["water_toe"] is not None else 0.0
    water_crest = (
        float(v["water_crest"])
        if water_on and v["water_crest"] is not None
        else 0.0
    )

    return {
        "H": float(v["height"]),
        "beta": float(v["angle"]),
        "Dfac": float(v["depth_factor"]),
        "layers": [
            {
                "t": 1.0,
                "gam": float(v["unit_weight"]),
                "gsat": float(v["sat_weight"]),
                "c": float(v["cohesion"]),
                "phi": float(v["friction"]),
            }
        ],
        "water_on": water_on,
        "hw_toe": water_toe,
        "hw_crest": water_crest,
    }


def geometry_figure(v):
    height = float(v["height"] or 0.0)
    angle = float(v["angle"] or 30.0)
    run = (
        height / max(np.tan(np.radians(angle)), 0.01)
        if height > 0
        else 0.0
    )
    crest = run + height * 1.25 if height > 0 else 20.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, crest + 8, crest + 8, run, 0, 0],
            y=[0, 0, height, height, 0, 0],
            fill="toself",
            fillcolor="rgba(241,190,94,.23)",
            line=dict(color="#f2c36b", width=2),
            name="Slope mass",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, run, crest + 8],
            y=[0, height, height],
            mode="lines",
            line=dict(color="#f2c36b", width=4),
            name="Ground profile",
        )
    )

    if (
        v["water_enabled"]
        and v["water_toe"] is not None
        and v["water_crest"] is not None
    ):
        fig.add_trace(
            go.Scatter(
                x=[0, crest + 8],
                y=[float(v["water_toe"]), float(v["water_crest"])],
                mode="lines",
                line=dict(color="#35b9ff", dash="dash", width=2),
                name="Water table",
            )
        )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=12, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#091729",
        font_color="#dbeaff",
        xaxis_title="Distance (m)",
        yaxis_title="Elevation (m)",
        xaxis=dict(gridcolor="#263c56"),
        yaxis=dict(gridcolor="#263c56", rangemode="tozero"),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def validate_inputs(values):
    """Return a list of missing or inconsistent required inputs."""
    required = [
        ("height", "slope height"),
        ("angle", "slope angle"),
        ("unit_weight", "unit weight"),
        ("sat_weight", "saturated unit weight"),
        ("cohesion", "cohesion"),
        ("friction", "friction angle"),
    ]
    missing = [
        label for key, label in required
        if values.get(key) is None
    ]

    if values.get("water_enabled"):
        if values.get("water_toe") is None:
            missing.append("water level at toe")
        if values.get("water_crest") is None:
            missing.append("water level at crest")

    return missing


def render_results(worksheet):
    result = worksheet.get("result")

    if not result:
        st.markdown(
            """
            <div class="panel">
              <h3>📊 Analysis Results</h3>
              <div class="muted">
                No analysis has been run for this worksheet yet.
                Enter your soil and geometry data, then select Run Analysis.
              </div>
            </div>
            <div class="panel">
              <h3>💡 Key Findings</h3>
              <div class="muted">
                The critical slip surface, factor of safety and recommendations
                will appear here after a successful calculation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    fos_bishop = float(result["fos_bishop"])
    fos_fellenius = float(result["fos_fellenius"])
    selected_method = result["method"]
    selected_fos = (
        fos_bishop if selected_method == "Bishop" else fos_fellenius
    )

    status_text = (
        "STABLE"
        if selected_fos >= 1.5
        else "MARGINAL"
        if selected_fos >= 1.0
        else "UNSTABLE"
    )
    css_class = "good" if selected_fos >= 1.5 else "bad"
    status_color = "#20d99b" if selected_fos >= 1.5 else "#ff7777"

    st.markdown(
        f"""
        <div class="panel {css_class}">
          <h3>📊 Analysis Results</h3>
          <div class="metric-label">Factor of Safety · {selected_method}</div>
          <div class="metric-value">{selected_fos:.3f}</div>
          <div style="font-weight:800;color:{status_color};margin-top:5px">
            ● Slope is {status_text}
          </div>
          <hr>
          <div class="muted">
            Bishop: {fos_bishop:.3f}
            &nbsp; | &nbsp;
            Fellenius: {fos_fellenius:.3f}
          </div>
        </div>
        <div class="panel">
          <h3>💡 Key Findings</h3>
          <div>Selected method: <b>{selected_method}</b></div>
          <div>Calculated factor of safety: <b>{selected_fos:.3f}</b></div>
          <div>Screening interpretation: <b>{status_text}</b></div>
        </div>
        <div class="panel">
          <h3>🛡️ Safety Guidelines</h3>
          <div class="muted">
            FOS below 1.0: calculated instability<br>
            1.0–1.5: review design criteria and conditions<br>
            Above 1.5: commonly treated as stable for preliminary screening;
            confirm project requirements.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Engineering screening only. Verify assumptions, groundwater, loading, "
        "stratigraphy and design criteria with a qualified geotechnical engineer."
    )


# --------------------------- Session and worksheet setup --------------------
if "book" not in st.session_state:
    st.session_state.book = load_book()

if not st.session_state.book:
    st.session_state.book = fresh_book()

if (
    "current" not in st.session_state
    or st.session_state.current not in st.session_state.book
):
    st.session_state.current = next(iter(st.session_state.book))

book = st.session_state.book

st.markdown(
    """
    <div class="hero">
      <h1>⛰️ SlopeLab</h1>
      <p>Slope Stability Studio · Layered soils · Groundwater · Bishop &amp; Fellenius analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dynamic worksheet selector: unlike a fixed four-button bar, this supports
# Worksheet 5, Worksheet 6, and any later worksheets too.
nav_col, new_col, delete_col = st.columns([2.2, 1, 1])

with nav_col:
    worksheet_names = list(book.keys())
    current_index = worksheet_names.index(st.session_state.current)
    selected_worksheet = st.selectbox(
        "Active worksheet",
        worksheet_names,
        index=current_index,
        key="worksheet_selector",
    )
    st.session_state.current = selected_worksheet

with new_col:
    st.markdown("<div style='height:29px'></div>", unsafe_allow_html=True)
    if st.button("＋ New Worksheet", type="primary", use_container_width=True):
        worksheet_name = new_name(book)
        book[worksheet_name] = {
            "inputs": DEFAULT_INPUTS.copy(),
            "result": None,
        }
        st.session_state.current = worksheet_name
        save_book()
        st.rerun()

with delete_col:
    st.markdown("<div style='height:29px'></div>", unsafe_allow_html=True)
    if st.button(
        "🗑 Delete",
        disabled=len(book) <= 1,
        use_container_width=True,
    ):
        del book[st.session_state.current]
        st.session_state.current = next(iter(book))
        save_book()
        st.rerun()


# ------------------------------- Active sheet -------------------------------
current = st.session_state.current
worksheet = book[current]
stored_inputs = worksheet.setdefault("inputs", {})
values = {**DEFAULT_INPUTS, **stored_inputs}

left, right = st.columns([2.15, 1], gap="large")

with left:
    st.markdown(
        f"""
        <div class="panel">
          <h3>✏️ {current} — Model Parameters</h3>
          <div class="muted">
            Enter the slope geometry and soil properties. New worksheets start
            blank for project-specific data.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Geometry")
        geometry_1, geometry_2, geometry_3 = st.columns(3)

        values["height"] = geometry_1.number_input(
            "Slope height H (m)",
            min_value=1.0,
            max_value=100.0,
            value=values["height"],
            step=0.5,
            key=f"height_{current}",
        )
        values["angle"] = geometry_2.number_input(
            "Slope angle β (°)",
            min_value=10.0,
            max_value=80.0,
            value=values["angle"],
            step=1.0,
            key=f"angle_{current}",
        )
        values["depth_factor"] = geometry_3.number_input(
            "Search depth below toe (×H)",
            min_value=0.5,
            max_value=6.0,
            value=float(values["depth_factor"]),
            step=0.5,
            key=f"depth_{current}",
        )

        st.markdown("#### Soil Properties")
        soil_1, soil_2, soil_3, soil_4 = st.columns(4)

        values["unit_weight"] = soil_1.number_input(
            "Unit weight γ (kN/m³)",
            min_value=10.0,
            max_value=26.0,
            value=values["unit_weight"],
            step=0.5,
            key=f"unit_weight_{current}",
        )
        values["sat_weight"] = soil_2.number_input(
            "Saturated γsat (kN/m³)",
            min_value=10.0,
            max_value=26.0,
            value=values["sat_weight"],
            step=0.5,
            key=f"sat_weight_{current}",
        )
        values["cohesion"] = soil_3.number_input(
            "Cohesion c′ (kPa)",
            min_value=0.0,
            max_value=500.0,
            value=values["cohesion"],
            step=1.0,
            key=f"cohesion_{current}",
        )
        values["friction"] = soil_4.number_input(
            "Friction angle φ′ (°)",
            min_value=0.0,
            max_value=50.0,
            value=values["friction"],
            step=1.0,
            key=f"friction_{current}",
        )

        st.markdown("#### Groundwater")
        values["water_enabled"] = st.checkbox(
            "Include water table",
            value=bool(values["water_enabled"]),
            key=f"water_enabled_{current}",
        )
        water_1, water_2 = st.columns(2)

        values["water_toe"] = water_1.number_input(
            "Water level at toe (m)",
            min_value=0.0,
            max_value=100.0,
            value=values["water_toe"],
            step=0.5,
            disabled=not values["water_enabled"],
            key=f"water_toe_{current}",
        )
        values["water_crest"] = water_2.number_input(
            "Water level at crest (m)",
            min_value=0.0,
            max_value=100.0,
            value=values["water_crest"],
            step=0.5,
            disabled=not values["water_enabled"],
            key=f"water_crest_{current}",
        )

        st.markdown("#### Analysis Settings")
        analysis_1, analysis_2, analysis_3 = st.columns(3)

        values["method"] = analysis_1.selectbox(
            "Primary method",
            ["Bishop", "Fellenius"],
            index=["Bishop", "Fellenius"].index(values["method"]),
            key=f"method_{current}",
        )
        values["slices"] = analysis_2.slider(
            "Number of slices",
            min_value=15,
            max_value=60,
            value=int(values["slices"]),
            key=f"slices_{current}",
        )
        values["density"] = analysis_3.selectbox(
            "Search density",
            ["Fast", "Standard", "Fine"],
            index=["Fast", "Standard", "Fine"].index(values["density"]),
            key=f"density_{current}",
        )

    st.markdown(
        """
        <div class="panel">
          <h3>📐 Slope Geometry</h3>
          <div class="muted">Preview updates as geometry values change.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        geometry_figure(values),
        use_container_width=True,
        key=f"geometry_plot_{current}",
    )

    if st.button(
        "▶  Run Analysis",
        type="primary",
        use_container_width=True,
        key=f"run_{current}",
    ):
        missing = validate_inputs(values)

        if missing:
            st.error(
                "Complete the required inputs before analysis: "
                + ", ".join(missing)
                + "."
            )
        elif not ENGINE_AVAILABLE:
            st.error(
                "The calculation engine could not be loaded. Put the original "
                "slope_app.py file in the same folder as app.py. Details: "
                + ENGINE_ERROR
            )
        else:
            try:
                model = make_model(values)

                with st.spinner(
                    "Searching for the critical slip surface and calculating "
                    "factors of safety…"
                ):
                    # Pass the worksheet's selected primary method to the engine.
                    raw_result = cached_search(
                        model,
                        int(values["slices"]),
                        values["method"],
                        values["density"],
                    )

                if raw_result is None:
                    st.error(
                        "No valid slip surface was found. Review the model "
                        "inputs and search settings."
                    )
                else:
                    worksheet["result"] = {
                        "fos_bishop": float(raw_result["fos_bishop"]),
                        "fos_fellenius": float(raw_result["fos_fellenius"]),
                        "method": values["method"],
                        "model": model,
                        "raw": raw_result,
                    }
                    save_book()
                    st.success(
                        "Analysis completed. Results are shown in the panel."
                    )
                    st.rerun()

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    # Persist current worksheet inputs after widget values have been collected.
    worksheet["inputs"] = values
    save_book()


with right:
    render_results(worksheet)

    if worksheet.get("result") and ENGINE_AVAILABLE:
        try:
            st.markdown(
                '<div class="panel"><h3>🔎 Critical Slip Surface</h3></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                fig_section(
                    prepare(worksheet["result"]["model"]),
                    worksheet["result"]["raw"],
                    True,
                    True,
                ),
                use_container_width=True,
                key=f"critical_plot_{current}",
            )
        except Exception as exc:
            st.caption(f"Slip-surface plot unavailable: {exc}")


st.caption(
    "Local SQLite storage keeps worksheets on this server between app restarts. "
    "For multi-user cloud deployment, configure a persistent database and "
    "per-user authentication. Engineering outputs must be independently "
    "verified before design use."
)

if not ENGINE_AVAILABLE:
    st.info(
        "Dashboard preview mode: the interface works, but real factor-of-safety "
        "calculations require the original slope_app.py calculation engine."
    )
