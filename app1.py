# app1.py

import os
import streamlit as st
import pandas as pd
from io import BytesIO
from simulation import run_simulation

# 1. Page config
st.set_page_config(page_title="Grain Distribution Simulator", layout="wide")
st.title("🚛 Grain Distribution Simulator")

# 2. Helper: write multiple DataFrames into a single Excel in memory
def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()

# 3. File uploader (or fallback to a local template)
uploaded = st.file_uploader("Upload master workbook (.xlsx)", type="xlsx")
if uploaded is not None:
    master = uploaded
elif os.path.exists("grain_simulator_template.xlsx"):
    master = "grain_simulator_template.xlsx"
else:
    st.error("❌ Please upload an Excel template or place 'grain_simulator_template.xlsx' in the working directory.")
    st.stop()

# 4. Load input sheets from the workbook
@st.cache_data
def load_inputs(path):
    settings = pd.read_excel(path, sheet_name="Settings")
    lgs      = pd.read_excel(path, sheet_name="LGs")
    fps      = pd.read_excel(path, sheet_name="FPS")
    try:
        vehicles = pd.read_excel(path, sheet_name="Vehicles", dtype={"Mapped_LG_IDs": "string"})
    except ValueError:
        vehicles = pd.DataFrame(columns=["Vehicle_ID", "Capacity_tons", "Mapped_LG_IDs"])
    if "Mapped_LG_IDs" in vehicles.columns:
        vehicles["Mapped_LG_IDs"] = vehicles["Mapped_LG_IDs"].astype("string")
    return settings, lgs, fps, vehicles

settings, lgs, fps, vehicles = load_inputs(master)

# 5. Preview inputs
with st.expander("🔍 Preview Inputs"):
    st.subheader("Settings");      st.dataframe(settings)
    st.subheader("Local Godowns"); st.dataframe(lgs)
    st.subheader("Shops (FPS)");   st.dataframe(fps)
    st.subheader("Vehicles");      st.dataframe(vehicles)

# 6. Run simulation
if st.button("▶️ Run Simulation"):
    with st.spinner("Running…"):
        dispatch_cg, dispatch_lg, stock_levels = run_simulation(
            master, settings, lgs, fps, vehicles
        )
    st.success("✅ Simulation complete")

    # 7. Package outputs into an Excel workbook
    output_sheets = {
        "Settings":      settings,
        "LGs":           lgs,
        "FPS":           fps,
        "Vehicles":      vehicles,
        "CG_to_LG":      dispatch_cg,
        "LG_to_FPS":     dispatch_lg,
        "Stock_Levels":  stock_levels
    }
    excel_data = to_excel(output_sheets)

    # 8. Download button
    st.download_button(
        label="📥 Download simulation_output.xlsx",
        data=excel_data,
        file_name="simulation_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Upload your master workbook above and then click ▶️ to run the simulation.")