# simulation.py
import streamlit
import io
import pandas as pd
import math

def run_simulation(
    master_workbook,          # str path or file-like buffer
    settings: pd.DataFrame,
    lgs: pd.DataFrame,
    fps: pd.DataFrame,
    vehicles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs a two-phase simulation:

    1) LG → FPS dispatch (priority-based, with vehicle mapping + per-vehicle trip caps)
       - Produces `dispatch_lg` with columns: Day, Vehicle_ID, LG_ID, FPS_ID, Quantity_tons
       - Produces `stock_levels` (LG & FPS end-of-day stock)

    2) CG → LG pre-dispatch using derived LG daily requirement from phase (1)
       - Produces `dispatch_cg` with columns: Day, Vehicle_ID, LG_ID, Quantity_tons

    Returns:
        (dispatch_cg, dispatch_lg, stock_levels)
    """

    # -----------------------------
    # 0) Read key parameters safely
    # -----------------------------
    def _get_setting(param_name, default=None, cast=float):
        try:
            val = settings.loc[settings["Parameter"] == param_name, "Value"].iloc[0]
            return cast(val)
        except Exception:
            if default is None:
                raise ValueError(f"Missing required setting: {param_name}")
            return cast(default)

    DAYS       = _get_setting("Distribution_Days", cast=int)
    TRUCK_CAP  = _get_setting("Vehicle_Capacity_tons", cast=float)
    TOT_V      = _get_setting("Vehicles_Total", cast=int)
    MAX_TRIPS  = _get_setting("Max_Trips_Per_Vehicle_Per_Day", cast=int)
    DEFAULT_LEAD = _get_setting("Default_Lead_Time_days", cast=float)

    # -----------------------------
    # 1) Prepare LG & FPS mappings
    # -----------------------------
    # Normalize LG keys (support either ID or Name references in FPS.Linked_LG_ID)
    lgs = lgs.copy()
    if "LG_ID" not in lgs.columns or "LG_Name" not in lgs.columns:
        raise ValueError("LGs sheet must contain columns: LG_ID, LG_Name")

    # Build bi-directional maps
    lgid_by_name = {str(nm).strip().lower(): int(lg_id) for lg_id, nm in zip(lgs["LG_ID"], lgs["LG_Name"])}
    valid_lg_ids = set(int(x) for x in lgs["LG_ID"])

    def normalize_lg_ref(val):
        """Accepts either an int-like ID or a name; returns int LG_ID or None."""
        if pd.isna(val):
            return None
        s = str(val).strip()
        # Try as int ID
        try:
            i = int(float(s))  # handles "5" or "5.0"
            return i if i in valid_lg_ids else None
        except ValueError:
            pass
        # Try as name
        return lgid_by_name.get(s.lower())

    # Make sure FPS has core columns
    req_cols = {"FPS_ID", "Monthly_Demand_tons", "Max_Capacity_tons", "Linked_LG_ID"}
    missing = req_cols - set(fps.columns)
    if missing:
        raise ValueError(f"FPS sheet missing required columns: {missing}")

    fps = fps.copy()
    # Ensure Lead_Time_days exists and fill NaN with default
    if "Lead_Time_days" not in fps.columns:
        fps["Lead_Time_days"] = DEFAULT_LEAD
    else:
        fps["Lead_Time_days"] = fps["Lead_Time_days"].fillna(DEFAULT_LEAD)

    # Compute daily demand and thresholds
    fps["Daily_Demand_tons"] = fps["Monthly_Demand_tons"] / 30.0
    fps["Reorder_Threshold_tons"] = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]

    # Attach LG_ID (normalized) to each FPS
    fps["LG_ID"] = fps["Linked_LG_ID"].apply(normalize_lg_ref)
    if fps["LG_ID"].isna().any():
        bad_rows = fps[fps["LG_ID"].isna()][["FPS_ID", "Linked_LG_ID"]]
        raise ValueError(
            "Some FPS rows couldn't map Linked_LG_ID to a valid LG_ID. "
            f"Examples:\n{bad_rows.head(5).to_string(index=False)}\n"
            "Ensure Linked_LG_ID is either a valid LG_ID or a valid LG_Name."
        )

    fps["LG_ID"] = fps["LG_ID"].astype(int)

    # -----------------------------
    # 2) Prepare Vehicles mapping
    # -----------------------------
    vehicles = vehicles.copy()
    if vehicles.empty:
        # Fallback: create a basic pool of vehicles all mapped to all LGs
        vehicles = pd.DataFrame({
            "Vehicle_ID": list(range(1, TOT_V + 1)),
            "Capacity_tons": [TRUCK_CAP] * TOT_V,
            "Mapped_LG_IDs": [",".join(str(x) for x in sorted(valid_lg_ids))] * TOT_V
        })
    else:
        if "Vehicle_ID" not in vehicles.columns:
            raise ValueError("Vehicles sheet must contain 'Vehicle_ID'")
        if "Capacity_tons" not in vehicles.columns:
            vehicles["Capacity_tons"] = TRUCK_CAP
        if "Mapped_LG_IDs" not in vehicles.columns:
            # If not given, assume each vehicle can serve all LGs
            vehicles["Mapped_LG_IDs"] = ",".join(str(x) for x in sorted(valid_lg_ids))

    # Parse Mapped_LG_IDs into normalized lists for easy filtering
    def parse_lg_list(val):
        if pd.isna(val):
            return []
        out = []
        for token in str(val).split(","):
            token = token.strip()
            if not token:
                continue
            # try ID then name
            try:
                i = int(float(token))
                if i in valid_lg_ids:
                    out.append(i)
                    continue
            except ValueError:
                pass
            # maybe it is a name
            mapped = normalize_lg_ref(token)
            if mapped is not None:
                out.append(mapped)
        return sorted(set(out))

    vehicles["Mapped_LGs_List"] = vehicles["Mapped_LG_IDs"].apply(parse_lg_list)
    if vehicles["Mapped_LGs_List"].apply(len).eq(0).any():
        bad = vehicles[vehicles["Mapped_LGs_List"].apply(len).eq(0)][["Vehicle_ID", "Mapped_LG_IDs"]]
        raise ValueError(
            "Some vehicles couldn't map any LGs from 'Mapped_LG_IDs'. "
            f"Examples:\n{bad.head(5).to_string(index=False)}"
        )

    # -----------------------------
    # 3) LG → FPS SIMULATION
    # -----------------------------
    # Initialize stocks
    if "Initial_Allocation_tons" not in lgs.columns:
        lgs["Initial_Allocation_tons"] = 0.0

    lg_stock = {int(row["LG_ID"]): float(row["Initial_Allocation_tons"]) for _, row in lgs.iterrows()}
    fps_stock = {int(fid): 0.0 for fid in fps["FPS_ID"]}

    dispatch_lg_rows = []
    stock_rows = []

    for day in range(1, DAYS + 1):
        # 3a) FPS consumes daily demand
        for _, r in fps.iterrows():
            fid = int(r["FPS_ID"])
            fps_stock[fid] = max(0.0, fps_stock[fid] - float(r["Daily_Demand_tons"]))

        # 3b) Compute needs
        needs = []
        for _, r in fps.iterrows():
            fid  = int(r["FPS_ID"])
            lgid = int(r["LG_ID"])
            current = fps_stock[fid]
            threshold = float(r["Reorder_Threshold_tons"])
            max_cap  = float(r["Max_Capacity_tons"])
            if current <= threshold:
                available_at_lg = lg_stock.get(lgid, 0.0)
                need_qty = min(max_cap - current, available_at_lg)
                if need_qty > 0:
                    urgency = (threshold - current) / float(r["Daily_Demand_tons"]) if r["Daily_Demand_tons"] > 0 else 0
                    needs.append((urgency, fid, lgid, need_qty))
        needs.sort(reverse=True, key=lambda x: x[0])

        # 3c) Reset vehicle usage counters for the day
        vehicles["Trips_Used"] = 0

        # 3d) Dispatch loop
        for urgency, fid, lgid, need_qty in needs:
            # candidate vehicles that can serve this LG and have trips left
            cand = vehicles[vehicles["Mapped_LGs_List"].apply(lambda lst: lgid in lst)].copy()
            cand = cand[cand["Trips_Used"] < MAX_TRIPS]
            if cand.empty:
                continue

            # Prefer shared vehicles (mapped to >1 LG)
            cand["is_shared"] = cand["Mapped_LGs_List"].apply(lambda lst: len(lst) > 1)
            cand = cand.sort_values(["is_shared"], ascending=False)
            chosen = cand.iloc[0]

            vid = chosen["Vehicle_ID"]
            cap = float(chosen["Capacity_tons"])
            qty = min(cap, need_qty, lg_stock.get(lgid, 0.0))
            if qty <= 0:
                continue

            dispatch_lg_rows.append({
                "Day": int(day),
                "Vehicle_ID": vid,
                "LG_ID": int(lgid),           # <-- GUARANTEED LG_ID
                "FPS_ID": int(fid),
                "Quantity_tons": float(qty)
            })

            # update stocks & vehicle usage
            lg_stock[lgid] = lg_stock.get(lgid, 0.0) - qty
            fps_stock[fid] = fps_stock.get(fid, 0.0) + qty
            vehicles.loc[vehicles["Vehicle_ID"] == vid, "Trips_Used"] += 1

        # 3e) Record end-of-day stocks
        for lgid, st in lg_stock.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "LG",  "Entity_ID": int(lgid), "Stock_Level_tons": float(st)})
        for fid, st in fps_stock.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "FPS", "Entity_ID": int(fid),  "Stock_Level_tons": float(st)})

    # Build DataFrames with **expected schema**
    dispatch_lg = pd.DataFrame(dispatch_lg_rows, columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons"])
    stock_levels = pd.DataFrame(stock_rows, columns=["Day","Entity_Type","Entity_ID","Stock_Level_tons"])

    # Ensure required columns exist even if empty (prevents KeyError later)
    if dispatch_lg.empty:
        dispatch_lg = pd.DataFrame(columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons"])

    # -----------------------------------------------
    # 4) Derive LG daily requirement from dispatch_lg
    # -----------------------------------------------
    required_cols = {"LG_ID", "Day", "Quantity_tons"}
    missing = required_cols - set(dispatch_lg.columns)
    if missing:
        raise ValueError(f"dispatch_lg is missing required columns: {missing}")

    if dispatch_lg.empty:
        # If nothing was dispatched, there is no derived requirement.
        # To avoid crashing, create an all-zero requirement for the known LGs.
        lg_daily_req = (
            pd.MultiIndex.from_product([sorted(valid_lg_ids), range(1, DAYS + 1)], names=["LG_ID","Day"])
            .to_frame(index=False)
            .assign(Daily_Requirement_tons=0.0)
        )
    else:
        lg_daily_req = (
            dispatch_lg
            .groupby(["LG_ID", "Day"])["Quantity_tons"]
            .sum()
            .reset_index()
            .rename(columns={"Quantity_tons": "Daily_Requirement_tons"})
        )

    req_pivot = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                         values="Daily_Requirement_tons",
                                         aggfunc="sum", fill_value=0.0)

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=True, sheet_name='Req_Pivot')
        return output.getvalue()

    # Inside your Streamlit code:
    excel_data = to_excel(req_pivot)

    streamlit.download_button(
        label="📥 Download req_pivot.xlsx",
        data=excel_data,
        file_name="req_pivot.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    excel_raw_data = to_excel(lg_daily_req, sheet_name="LG_Daily_Req")
    st.download_button(
        label="📥 Download lg_daily_req.xlsx",
        data=excel_raw_data,
        file_name="lg_daily_req.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    # -----------------------------------------------
    # 5) CG → LG PRE-DISPATCH (same DAYS timeline)
    # -----------------------------------------------
    # Capacity
    try:
        cap_df = pd.read_excel(master_workbook, sheet_name="LG_Capacity")
        if {"LG_ID","Capacity_tons"} <= set(cap_df.columns):
            capacity = {int(r["LG_ID"]): float(r["Capacity_tons"]) for _, r in cap_df.iterrows()}
        else:
            raise ValueError
    except Exception:
        if "Storage_Capacity_tons" not in lgs.columns:
            raise ValueError("Provide LG_Capacity sheet or 'Storage_Capacity_tons' in LGs.")
        capacity = {int(r["LG_ID"]): float(r["Storage_Capacity_tons"]) for _, r in lgs.iterrows()}

    # start CG stocks at current LG stock (or 0 if you prefer fresh inflow)
    lg_stock_cg = {int(r["LG_ID"]): float(r.get("Initial_LG_stock", 0.0)) for _, r in lgs.iterrows()}

    dispatch_cg_rows = []

    def free_room(lg_id: int) -> float:
        return max(0.0, capacity.get(lg_id, 0.0) - lg_stock_cg.get(lg_id, 0.0))

    for day in range(1, DAYS + 1):
        trips_left = TOT_V

        # Serve today's requirement first for each LG
        for lgid in req_pivot.index:
            need_today = max(0.0, float(req_pivot.at[lgid, day]) - lg_stock_cg.get(lgid, 0.0))
            # ship in trips while we still need and have trips
            while trips_left > 0 and need_today > 1e-9:
                # assign a simple rotating vehicle id 1..TOT_V
                vid = TOT_V - trips_left + 1
                qty = min(TRUCK_CAP, need_today, free_room(lgid))
                if qty <= 1e-9:
                    break
                dispatch_cg_rows.append({
                    "Day": int(day),
                    "Vehicle_ID": int(vid),
                    "LG_ID": int(lgid),
                    "Quantity_tons": float(qty)
                })
                lg_stock_cg[lgid] = lg_stock_cg.get(lgid, 0.0) + qty
                trips_left -= 1
                need_today -= qty

        # Optional: If trips remain, you could pre-stock for future days (round-robin).
        # Skipped here to keep logic minimal and strictly "no backlog on the day" as per your earlier constraints.

    dispatch_cg = pd.DataFrame(dispatch_cg_rows, columns=["Day","Vehicle_ID","LG_ID","Quantity_tons"])

    return dispatch_cg, dispatch_lg, stock_levels
