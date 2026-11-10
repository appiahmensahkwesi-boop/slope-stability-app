"""
SlopeLab Worksheet
------------------
A batch worksheet for slope stability: fill in 4 or 6 cases in a table, press
"Analyse worksheet", and get the factor of safety (Bishop and Fellenius) for
every case, a comparison chart, and a percentage difference against a
reference value (hand calculation, Taylor chart or other software).

The app opens with a ready-made example worksheet. It uses the same engine as
slope_app.py, so keep both files in the same repository.

Run locally:  streamlit run worksheet_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from slope_app import (STATUS_COLORS, cached_search, fig_section, prepare,
                       status_of)

COLS = ["Case", "H (m)", "Angle (deg)", "Unit wt (kN/m3)", "Sat. wt (kN/m3)",
        "c' (kPa)", "phi' (deg)", "Water at toe (m)", "Water at crest (m)",
        "Reference FoS"]

# Example worksheet (6 rows). The first 4 rows are used for the 4-case sheet.
NAN = np.nan
EXAMPLE_ROWS = [
    ["1. Taylor chart, 45 deg, phi = 0", 10.0, 45.0, 18.0, 18.0, 25.0, 0.0, NAN, NAN, 0.767],
    ["2. Taylor chart, 60 deg, phi = 0", 10.0, 60.0, 18.0, 18.0, 40.0, 0.0, NAN, NAN, 1.163],
    ["3. Dry sand (infinite slope check)", 10.0, 25.0, 17.0, 20.0, 0.0, 35.0, NAN, NAN, 1.50],
    ["4. Clay-sand slope with water table", 8.0, 35.0, 18.5, 19.5, 12.0, 22.0, 0.0, 4.0, NAN],
    ["5. Taylor chart, 75 deg, phi = 0", 10.0, 75.0, 18.0, 18.0, 50.0, 0.0, NAN, NAN, 1.268],
    ["6. Homogeneous slope (default)", 10.0, 30.0, 18.0, 20.0, 15.0, 20.0, NAN, NAN, NAN],
]


def example_df():
    return pd.DataFrame(EXAMPLE_ROWS, columns=COLS)


def row_to_model(r, dfac):
    """One worksheet row -> model dict for the engine (single soil layer)."""
    gam = float(r["Unit wt (kN/m3)"])
    gsat = r["Sat. wt (kN/m3)"]
    gsat = float(gsat) if pd.notna(gsat) else gam
    c = float(r["c' (kPa)"]) if pd.notna(r["c' (kPa)"]) else 0.0
    phi = float(r["phi' (deg)"]) if pd.notna(r["phi' (deg)"]) else 0.0
    toe, crest = r["Water at toe (m)"], r["Water at crest (m)"]
    has_toe, has_crest = pd.notna(toe), pd.notna(crest)
    water_on = bool(has_toe or has_crest)
    toe_v = float(toe) if has_toe else 0.0
    crest_v = float(crest) if has_crest else toe_v
    return dict(
        H=float(r["H (m)"]), beta=float(r["Angle (deg)"]), Dfac=float(dfac),
        layers=[dict(t=1.0, gam=gam, gsat=gsat, c=c, phi=phi)],
        water_on=water_on, hw_toe=toe_v, hw_crest=crest_v)


def check_row(r):
    """Return an error message for a bad row, or None if it is fine."""
    need = ["H (m)", "Angle (deg)", "Unit wt (kN/m3)"]
    if any(pd.isna(r[k]) for k in need):
        return "H, angle and unit weight are required"
    if r["H (m)"] <= 0:
        return "H must be positive"
    if not 10 <= r["Angle (deg)"] <= 80:
        return "angle must be between 10 and 80 degrees"
    if r["Unit wt (kN/m3)"] <= 0:
        return "unit weight must be positive"
    return None


def analyse(df, dfac, nsl, density):
    out = []
    for _, r in df.iterrows():
        name = str(r["Case"])
        err = check_row(r)
        if err:
            out.append(dict(name=name, error=err))
            continue
        M = row_to_model(r, dfac)
        res = cached_search(M, int(nsl), "Bishop", density)
        if res is None:
            out.append(dict(name=name, error="no valid slip surface found"))
            continue
        out.append(dict(name=name, M=M, res=res, ref=r["Reference FoS"]))
    return out


def results_frame(items, method):
    rows = []
    for it in items:
        if "error" in it:
            rows.append(dict(Case=it["name"], Bishop=np.nan, Fellenius=np.nan,
                             Status="Input error: " + it["error"],
                             Reference=np.nan, **{"Difference (%)": np.nan}))
            continue
        res = it["res"]
        F = res["fos_bishop"] if method == "Bishop" else res["fos_fellenius"]
        ref = it["ref"]
        diff = ((res["fos_bishop"] - ref) / ref * 100.0
                if pd.notna(ref) and ref else np.nan)
        rows.append(dict(Case=it["name"], Bishop=res["fos_bishop"],
                         Fellenius=res["fos_fellenius"],
                         Status=status_of(F)[0],
                         Reference=ref if pd.notna(ref) else np.nan,
                         **{"Difference (%)": diff}))
    return pd.DataFrame(rows)


def fig_bars(rf, method):
    import plotly.graph_objects as go

    col = "Bishop" if method == "Bishop" else "Fellenius"
    colors = [STATUS_COLORS.get(s, "#999999") for s in rf["Status"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rf["Case"], y=rf[col], marker_color=colors,
                         text=[f"{v:.2f}" if pd.notna(v) else "" for v in rf[col]],
                         textposition="outside", name=f"FoS ({method})"))
    if rf["Reference"].notna().any():
        fig.add_trace(go.Scatter(x=rf["Case"], y=rf["Reference"], mode="markers",
                                 marker=dict(symbol="diamond", size=12,
                                             color="#111111"),
                                 name="Reference FoS"))
    fig.add_hline(y=1.0, line=dict(color="#e5484d", dash="dot"),
                  annotation_text="FoS = 1.0")
    fig.add_hline(y=1.5, line=dict(color="#2e9e5b", dash="dot"),
                  annotation_text="FoS = 1.5")
    fig.update_layout(title=f"Factor of safety by case ({method})",
                      yaxis=dict(title="Factor of safety"),
                      height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.3))
    return fig


def main():
    st.set_page_config(page_title="SlopeLab Worksheet", page_icon="📋",
                       layout="wide")

    def show(obj_fn, *a, **k):
        """Call a Streamlit display function with full width on any version."""
        try:
            return obj_fn(*a, width="stretch", **k)
        except Exception:  # noqa: BLE001
            return obj_fn(*a, use_container_width=True, **k)

    S = st.session_state

    st.markdown(
        """
        <div style="padding:14px 18px;border-radius:12px;
        background:linear-gradient(90deg,#7c3aed 0%,#1f6feb 100%);color:white;">
        <div style="font-size:1.6rem;font-weight:700;">📋 SlopeLab Worksheet</div>
        <div style="opacity:.9;">Fill in 4 or 6 slope cases, press Analyse, and
        compare every case in one table</div></div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Worksheet settings")
        n = st.radio("Number of cases", [4, 6], index=1, horizontal=True)
        method = st.radio("Method shown in status and chart",
                          ["Bishop", "Fellenius"], horizontal=True)
        dfac = st.number_input("Model depth below toe (x H)", min_value=0.5,
                               max_value=6.0, value=2.0, step=0.5)
        density = st.select_slider("Search density",
                                   ["Fast", "Standard", "Fine"],
                                   value="Standard")
        nsl = st.slider("Number of slices", 15, 60, 30)
        if st.button("Reset worksheet to the example"):
            for k in [k for k in S.keys()
                      if str(k).startswith("ws_") or k == "results"]:
                del S[k]
            st.rerun()
        st.caption("Each case is one homogeneous slope. For layered soils, "
                   "3D and maps, use the main SlopeLab app.")

    st.subheader(f"Worksheet - {n} cases")
    st.caption("Edit any cell. Leave the water-level cells empty for no water "
               "table (levels are metres above the toe). 'Reference FoS' is "
               "optional: type a hand-calculation, chart or software value to "
               "see the percentage difference.")

    cfg = {
        "Case": st.column_config.TextColumn("Case", width="large"),
        "H (m)": st.column_config.NumberColumn("H (m)", min_value=1.0,
                                               max_value=100.0, step=0.5),
        "Angle (deg)": st.column_config.NumberColumn(
            "Angle (deg)", min_value=10.0, max_value=80.0, step=1.0),
        "Unit wt (kN/m3)": st.column_config.NumberColumn(
            "Unit wt (kN/m3)", min_value=10.0, max_value=26.0, step=0.5),
        "Sat. wt (kN/m3)": st.column_config.NumberColumn(
            "Sat. wt (kN/m3)", min_value=10.0, max_value=26.0, step=0.5),
        "c' (kPa)": st.column_config.NumberColumn(
            "c' (kPa)", min_value=0.0, max_value=500.0, step=1.0),
        "phi' (deg)": st.column_config.NumberColumn(
            "phi' (deg)", min_value=0.0, max_value=50.0, step=1.0),
        "Water at toe (m)": st.column_config.NumberColumn(
            "Water at toe (m)", step=0.5),
        "Water at crest (m)": st.column_config.NumberColumn(
            "Water at crest (m)", step=0.5),
        "Reference FoS": st.column_config.NumberColumn(
            "Reference FoS", min_value=0.0, step=0.01),
    }
    base = example_df().iloc[:n].reset_index(drop=True)
    edited = show(st.data_editor, base, key=f"ws_editor_{n}", num_rows="fixed",
                  hide_index=True, column_config=cfg)

    sig = (edited.to_json(), n, float(dfac), int(nsl), density)

    run = st.button("Analyse worksheet", type="primary")
    if run or "results" not in S:  # opens with the example already analysed
        with st.spinner("Analysing all cases..."):
            S["results"] = dict(items=analyse(edited, dfac, nsl, density),
                                sig=sig)
    R = S["results"]
    if R["sig"] != sig:
        st.info("The worksheet has changed since the last analysis. "
                "Press **Analyse worksheet** to update the results.")

    items = R["items"]
    rf = results_frame(items, method)

    st.subheader("Results")
    shown = rf.copy()
    for c in ("Bishop", "Fellenius", "Reference", "Difference (%)"):
        shown[c] = shown[c].round(3 if c != "Difference (%)" else 1)
    show(st.dataframe, shown, hide_index=True)
    if rf["Bishop"].notna().any():
        show(st.plotly_chart, fig_bars(rf, method), key="ws_bars")

    ok = [it for it in items if "error" not in it]
    if ok:
        names = [it["name"] for it in ok]
        pick = st.selectbox("Show the critical slip surface for a case", names)
        it = ok[names.index(pick)]
        show(st.plotly_chart,
             fig_section(prepare(it["M"]), it["res"], True, True),
             key="ws_section")

    full = pd.concat([edited.reset_index(drop=True),
                      rf.drop(columns=["Case", "Reference"])], axis=1)
    st.download_button("Download worksheet and results (CSV)",
                       full.to_csv(index=False).encode("utf-8"),
                       "slope_worksheet_results.csv", "text/csv")
    st.caption("Differences compare the Bishop result with your reference "
               "value. Differences of a few percent are normal for chart "
               "values read from a published chart.")


if __name__ == "__main__":
    main()
