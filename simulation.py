# simulation.py

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
    Two-phase simulation with commodity tagging (AYY / PHH) on every dispatch row.

    Phase 1: LG → FPS
      - Produces dispatch_lg with:
        Day, Vehicle_ID, LG_ID, FPS_ID, Quantity_tons, AYY_tons, PHH_tons
      - Produces stock_levels (total stock, no commodity column) for LG & FPS (EOD snapshots)

    Phase 2: CG → LG
      - Uses per-LG per-day requirement derived from Phase 1 (and its AYY/PHH split)
      - Produces dispatch_cg with:
        Day, Vehicle_ID, LG_ID, Quantity_tons, AYY_tons, PHH_tons

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
    lgs = lgs.copy()
    if "LG_ID" not in lgs.columns or "LG_Name" not in lgs.columns:
        raise ValueError("LGs sheet must contain columns: LG_ID, LG_Name")

    lgid_by_name = {str(nm).strip().lower(): int(lg_id) for lg_id, nm in zip(lgs["LG_ID"], lgs["LG_Name"])}
    valid_lg_ids = set(int(x) for x in lgs["LG_ID"])

    def normalize_lg_ref(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        try:
            i = int(float(s))
            return i if i in valid_lg_ids else None
        except ValueError:
            pass
        return lgid_by_name.get(s.lower())

    req_cols = {"FPS_ID", "Monthly_Demand_tons", "Max_Capacity_tons", "Linked_LG_ID"}
    missing = req_cols - set(fps.columns)
    # We allow Monthly_Demand_tons to be absent IF per-commodity monthly columns are present.
    if missing - {"Monthly_Demand_tons"}:
        raise ValueError(f"FPS sheet missing required columns: {missing - {'Monthly_Demand_tons'}}")

    fps = fps.copy()
    if "Lead_Time_days" not in fps.columns:
        fps["Lead_Time_days"] = DEFAULT_LEAD
    else:
        fps["Lead_Time_days"] = fps["Lead_Time_days"].fillna(DEFAULT_LEAD)

    # -----------------------------
    # 1a) Per-commodity daily demand & shares
    # Accept either explicit monthly per-commodity or AYY/PHH columns.
    # -----------------------------
    # Normalize potential monthly columns
    if "Monthly_Demand_AYY_tons" not in fps.columns and "AYY" in fps.columns:
        fps["Monthly_Demand_AYY_tons"] = fps["AYY"]
    if "Monthly_Demand_PHH_tons" not in fps.columns and "PHH" in fps.columns:
        fps["Monthly_Demand_PHH_tons"] = fps["PHH"]

    # Daily per-commodity
    fps["Daily_Demand_AYY_tons"] = pd.to_numeric(fps.get("Monthly_Demand_AYY_tons", 0), errors="coerce").fillna(0) / 30.0
    fps["Daily_Demand_PHH_tons"] = pd.to_numeric(fps.get("Monthly_Demand_PHH_tons", 0), errors="coerce").fillna(0) / 30.0

    # Total daily demand:
    # - Prefer explicit total if present; otherwise sum of commodities.
    explicit_total_daily = pd.to_numeric(fps.get("Monthly_Demand_tons", 0), errors="coerce").fillna(0) / 30.0
    derived_total_daily  = fps["Daily_Demand_AYY_tons"] + fps["Daily_Demand_PHH_tons"]
    fps["Daily_Demand_tons"] = derived_total_daily.where(derived_total_daily > 0, explicit_total_daily)

    # Reorder thresholds (total + per-commodity for reference)
    fps["Reorder_Threshold_tons"]      = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]
    fps["Reorder_Threshold_AYY_tons"]  = fps["Daily_Demand_AYY_tons"] * fps["Lead_Time_days"]
    fps["Reorder_Threshold_PHH_tons"]  = fps["Daily_Demand_PHH_tons"] * fps["Lead_Time_days"]

    # Attach normalized LG_ID
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
    # 2) Vehicles mapping
    # -----------------------------
    vehicles = vehicles.copy()
    if vehicles.empty:
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
            vehicles["Mapped_LG_IDs"] = ",".join(str(x) for x in sorted(valid_lg_ids))

    def parse_lg_list(val):
        if pd.isna(val):
            return []
        out = []
        for token in str(val).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                i = int(float(token))
                if i in valid_lg_ids:
                    out.append(i); continue
            except ValueError:
                pass
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
    # 3) LG → FPS SIMULATION  (adds AYY_tons / PHH_tons on each row)
    # -----------------------------
    # Starting stocks (total) for LG & FPS used for capacity and threshold logic
    if "Initial_Allocation_tons" not in lgs.columns:
        lgs["Initial_Allocation_tons"] = 0.0

    lg_stock_total  = {int(row["LG_ID"]): float(row["Initial_Allocation_tons"]) for _, row in lgs.iterrows()}
    fps_stock_total = {int(fid): 0.0 for fid in fps["FPS_ID"]}

    dispatch_lg_rows = []
    stock_rows = []

    for day in range(1, DAYS + 1):
        # 3a) FPS consumes daily TOTAL demand at start of day
        for _, r in fps.iterrows():
            fid = int(r["FPS_ID"])
            fps_stock_total[fid] = max(0.0, fps_stock_total[fid] - float(r["Daily_Demand_tons"]))

        # 3b) Compute "needs" based on total threshold and available at LG
        needs = []
        for _, r in fps.iterrows():
            fid  = int(r["FPS_ID"])
            lgid = int(r["LG_ID"])
            current   = fps_stock_total[fid]
            threshold = float(r["Reorder_Threshold_tons"])
            max_cap   = float(r["Max_Capacity_tons"])

            if current <= threshold:
                available_at_lg = lg_stock_total.get(lgid, 0.0)
                need_qty = min(max_cap - current, available_at_lg)
                if need_qty > 0:
                    urgency = (threshold - current) / float(r["Daily_Demand_tons"]) if r["Daily_Demand_tons"] > 0 else 0
                    needs.append((urgency, fid, lgid, need_qty))
        needs.sort(reverse=True, key=lambda x: x[0])

        # 3c) Reset vehicle usage counters
        vehicles["Trips_Used"] = 0

        # 3d) Dispatch
        for urgency, fid, lgid, need_qty in needs:
            cand = vehicles[vehicles["Mapped_LGs_List"].apply(lambda lst: lgid in lst)].copy()
            cand = cand[cand["Trips_Used"] < MAX_TRIPS]
            if cand.empty:
                continue

            cand["is_shared"] = cand["Mapped_LGs_List"].apply(lambda lst: len(lst) > 1)
            cand = cand.sort_values(["is_shared"], ascending=False)
            chosen = cand.iloc[0]

            vid = chosen["Vehicle_ID"]
            cap = float(chosen["Capacity_tons"])
            qty = min(cap, need_qty, lg_stock_total.get(lgid, 0.0))
            if qty <= 0:
                continue

            # ---- NEW: split qty into AYY/PHH using FPS daily share (fallback to monthly share; else all PHH) ----
            r = fps.loc[fps["FPS_ID"] == fid].iloc[0]
            total_daily = float(r["Daily_Demand_tons"]) if "Daily_Demand_tons" in r else 0.0
            if total_daily > 0:
                ayy_ratio = float(r.get("Daily_Demand_AYY_tons", 0.0)) / total_daily
            else:
                base_ayy = float(r.get("Monthly_Demand_AYY_tons", r.get("AYY", 0.0)) or 0.0)
                base_phh = float(r.get("Monthly_Demand_PHH_tons", r.get("PHH", 0.0)) or 0.0)
                denom = base_ayy + base_phh
                ayy_ratio = (base_ayy / denom) if denom > 0 else 0.0
            ayy_t = qty * max(0.0, min(1.0, ayy_ratio))
            phh_t = qty - ayy_t
            # ----------------------------------------------------------------------------------------------------

            dispatch_lg_rows.append({
                "Day": int(day),
                "Vehicle_ID": vid,
                "LG_ID": int(lgid),
                "FPS_ID": int(fid),
                "Quantity_tons": float(qty),
                "AYY_tons": float(ayy_t),
                "PHH_tons": float(phh_t),
            })

            # update total stocks & vehicle usage
            lg_stock_total[lgid] = lg_stock_total.get(lgid, 0.0) - qty
            fps_stock_total[fid] = fps_stock_total.get(fid, 0.0) + qty
            vehicles.loc[vehicles["Vehicle_ID"] == vid, "Trips_Used"] += 1

        # 3e) EOD stocks (TOTAL only to keep dashboard compatibility)
        for lgid, st in lg_stock_total.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "LG",  "Entity_ID": int(lgid), "Stock_Level_tons": float(st)})
        for fid, st in fps_stock_total.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "FPS", "Entity_ID": int(fid),  "Stock_Level_tons": float(st)})

    dispatch_lg = pd.DataFrame(dispatch_lg_rows,
                               columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AYY_tons","PHH_tons"])
    stock_levels = pd.DataFrame(stock_rows, columns=["Day","Entity_Type","Entity_ID","Stock_Level_tons"])
    if dispatch_lg.empty:
        dispatch_lg = pd.DataFrame(columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AYY_tons","PHH_tons"])

    # -----------------------------------------------
    # 4) Derive per-day LG requirement (and its AYY/PHH split) from LG→FPS
    # -----------------------------------------------
    required_cols = {"LG_ID", "Day", "Quantity_tons"}
    if not required_cols.issubset(dispatch_lg.columns):
        raise ValueError(f"dispatch_lg is missing required columns: {required_cols - set(dispatch_lg.columns)}")

    if dispatch_lg.empty:
        lg_daily_req = (
            pd.MultiIndex.from_product([sorted(valid_lg_ids), range(1, DAYS + 1)], names=["LG_ID","Day"])
            .to_frame(index=False)
            .assign(Daily_Requirement_tons=0.0,
                    Daily_Requirement_AYY_tons=0.0,
                    Daily_Requirement_PHH_tons=0.0)
        )
    else:
        base = (
            dispatch_lg.groupby(["LG_ID", "Day"])[["Quantity_tons","AYY_tons","PHH_tons"]]
            .sum().reset_index()
            .rename(columns={
                "Quantity_tons": "Daily_Requirement_tons",
                "AYY_tons": "Daily_Requirement_AYY_tons",
                "PHH_tons": "Daily_Requirement_PHH_tons",
            })
        )
        lg_daily_req = base

    # Quick day-wise pivot for total requirement (used to iterate)
    req_pivot_total = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                               values="Daily_Requirement_tons",
                                               aggfunc="sum", fill_value=0.0)

    # Also keep per-commodity lookups (for proportional CG→LG splits)
    req_pivot_ayy = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                             values="Daily_Requirement_AYY_tons",
                                             aggfunc="sum", fill_value=0.0)
    req_pivot_phh = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                             values="Daily_Requirement_PHH_tons",
                                             aggfunc="sum", fill_value=0.0)

    # -----------------------------------------------
    # 5) CG → LG PRE-DISPATCH (TOTAL trips, per-trip AYY/PHH split)
    # -----------------------------------------------
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

    # Start CG→LG with optional baseline (TOTAL); if absent, start at 0
    lg_stock_cg_total = {int(r["LG_ID"]): float(r.get("Initial_LG_stock", 0.0)) for _, r in lgs.iterrows()}

    def free_room(lg_id: int) -> float:
        return max(0.0, capacity.get(lg_id, 0.0) - lg_stock_cg_total.get(lg_id, 0.0))

    dispatch_cg_rows = []

    for day in range(1, DAYS + 1):
        trips_left = TOT_V

        # For each LG: satisfy today's TOTAL requirement first
        for lgid in req_pivot_total.index:
            req_total = float(req_pivot_total.at[lgid, day]) if day in req_pivot_total.columns else 0.0
            req_ayy   = float(req_pivot_ayy.at[lgid, day])   if day in req_pivot_ayy.columns   else 0.0
            req_phh   = float(req_pivot_phh.at[lgid, day])   if day in req_pivot_phh.columns   else 0.0

            # Track how much of each commodity we've delivered for this LG today
            delivered_ayy = 0.0
            delivered_phh = 0.0

            need_today = max(0.0, req_total - lg_stock_cg_total.get(lgid, 0.0))
            while trips_left > 0 and need_today > 1e-9:
                vid = TOT_V - trips_left + 1
                qty = min(TRUCK_CAP, need_today, free_room(lgid))
                if qty <= 1e-9:
                    break

                # Split this trip by remaining per-commodity requirement for TODAY
                rem_ayy = max(0.0, req_ayy - delivered_ayy)
                rem_phh = max(0.0, req_phh - delivered_phh)
                denom = rem_ayy + rem_phh
                if denom > 0:
                    ayy_ratio = rem_ayy / denom
                else:
                    # Fallback: no explicit split left → send all as PHH
                    ayy_ratio = 0.0
                ayy_t = qty * max(0.0, min(1.0, ayy_ratio))
                phh_t = qty - ayy_t

                dispatch_cg_rows.append({
                    "Day": int(day),
                    "Vehicle_ID": int(vid),
                    "LG_ID": int(lgid),
                    "Quantity_tons": float(qty),
                    "AYY_tons": float(ayy_t),
                    "PHH_tons": float(phh_t),
                })

                # Update totals and counters
                lg_stock_cg_total[lgid] = lg_stock_cg_total.get(lgid, 0.0) + qty
                delivered_ayy += ayy_t
                delivered_phh += phh_t
                trips_left -= 1
                need_today -= qty

        # (Optional) Pre-stocking for future days with remaining trips could be added here if needed.

    dispatch_cg = pd.DataFrame(dispatch_cg_rows,
                               columns=["Day","Vehicle_ID","LG_ID","Quantity_tons","AYY_tons","PHH_tons"])

    return dispatch_cg, dispatch_lg, stock_levels
