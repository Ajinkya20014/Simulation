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
    Two-phase simulation with per-dispatch commodity tagging (AYY / PHH).

    Phase 1: LG → FPS
      - Produces dispatch_lg with:
        Day, Vehicle_ID, LG_ID, FPS_ID, Quantity_tons, AYY_tons, PHH_tons
      - Produces stock_levels (TOTAL stock, no commodity column) for LG & FPS (end-of-day snapshots)

    Phase 2: CG → LG  (MIRROR MODE)
      - EXACTLY mirrors per-LG, per-day totals (and AYY/PHH split) that LG shipped to FPS.
      - Produces dispatch_cg with:
        Day, Vehicle_ID, LG_ID, Quantity_tons, AYY_tons, PHH_tons

    Returns:
        (dispatch_cg, dispatch_lg, stock_levels)
    """

    # -----------------------------
    # 0) Settings
    # -----------------------------
    def _get_setting(param_name, default=None, cast=float):
        try:
            val = settings.loc[settings["Parameter"] == param_name, "Value"].iloc[0]
            return cast(val)
        except Exception:
            if default is None:
                raise ValueError(f"Missing required setting: {param_name}")
            return cast(default)

    DAYS         = _get_setting("Distribution_Days", cast=int)
    TRUCK_CAP    = _get_setting("Vehicle_Capacity_tons", cast=float)
    TOT_V        = _get_setting("Vehicles_Total", cast=int)
    MAX_TRIPS    = _get_setting("Max_Trips_Per_Vehicle_Per_Day", cast=int)
    DEFAULT_LEAD = _get_setting("Default_Lead_Time_days", cast=float)

    # -----------------------------
    # 1) LG & FPS normalization
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

    # FPS minimal requirements (allow Monthly_Demand_tons to be missing if per-commodity exists)
    need_cols = {"FPS_ID", "Max_Capacity_tons", "Linked_LG_ID"}
    miss = need_cols - set(fps.columns)
    if miss:
        raise ValueError(f"FPS sheet missing required columns: {miss}")

    fps = fps.copy()
    # Lead time
    fps["Lead_Time_days"] = pd.to_numeric(
        fps["Lead_Time_days"] if "Lead_Time_days" in fps.columns else DEFAULT_LEAD,
        errors="coerce"
    ).fillna(DEFAULT_LEAD)

    # Accept short aliases for monthly per-commodity
    if "Monthly_Demand_AYY_tons" not in fps.columns and "AYY" in fps.columns:
        fps["Monthly_Demand_AYY_tons"] = fps["AYY"]
    if "Monthly_Demand_PHH_tons" not in fps.columns and "PHH" in fps.columns:
        fps["Monthly_Demand_PHH_tons"] = fps["PHH"]

    # Guarantee monthly columns exist as Series (avoid scalar fallbacks)
    for col in ["Monthly_Demand_AYY_tons", "Monthly_Demand_PHH_tons", "Monthly_Demand_tons"]:
        if col not in fps.columns:
            fps[col] = 0.0

    # Daily demand
    ayy_monthly = pd.to_numeric(fps["Monthly_Demand_AYY_tons"], errors="coerce").fillna(0.0)
    phh_monthly = pd.to_numeric(fps["Monthly_Demand_PHH_tons"], errors="coerce").fillna(0.0)
    tot_monthly = pd.to_numeric(fps["Monthly_Demand_tons"],     errors="coerce").fillna(0.0)

    fps["Daily_Demand_AYY_tons"] = ayy_monthly / 30.0
    fps["Daily_Demand_PHH_tons"] = phh_monthly / 30.0

    explicit_total_daily = tot_monthly / 30.0
    derived_total_daily  = fps["Daily_Demand_AYY_tons"] + fps["Daily_Demand_PHH_tons"]
    fps["Daily_Demand_tons"] = derived_total_daily.where(derived_total_daily > 0, explicit_total_daily)

    fps["Reorder_Threshold_tons"]     = fps["Daily_Demand_tons"]     * fps["Lead_Time_days"]
    fps["Reorder_Threshold_AYY_tons"] = fps["Daily_Demand_AYY_tons"] * fps["Lead_Time_days"]
    fps["Reorder_Threshold_PHH_tons"] = fps["Daily_Demand_PHH_tons"] * fps["Lead_Time_days"]

    # Link FPS -> LG_ID
    fps["LG_ID"] = fps["Linked_LG_ID"].apply(normalize_lg_ref)
    if fps["LG_ID"].isna().any():
        bad = fps[fps["LG_ID"].isna()][["FPS_ID", "Linked_LG_ID"]]
        raise ValueError(
            "Some FPS rows couldn't map Linked_LG_ID to a valid LG_ID. "
            f"Examples:\n{bad.head(5).to_string(index=False)}\n"
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
    # 3) LG → FPS (with AYY/PHH on each row)
    # -----------------------------
    if "Initial_Allocation_tons" not in lgs.columns:
        lgs["Initial_Allocation_tons"] = 0.0

    lg_stock_total  = {int(row["LG_ID"]): float(row["Initial_Allocation_tons"]) for _, row in lgs.iterrows()}
    fps_stock_total = {int(fid): 0.0 for fid in fps["FPS_ID"]}

    dispatch_lg_rows = []
    stock_rows = []

    for day in range(1, DAYS + 1):
        # consume at FPS
        for _, r in fps.iterrows():
            fid = int(r["FPS_ID"])
            fps_stock_total[fid] = max(0.0, fps_stock_total[fid] - float(r["Daily_Demand_tons"]))

        # needs list
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

        # reset daily usage
        vehicles["Trips_Used"] = 0

        # dispatch
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

            r_fps = fps.loc[fps["FPS_ID"] == fid].iloc[0]
            total_daily = float(r_fps["Daily_Demand_tons"])
            if total_daily > 0:
                ayy_ratio = float(r_fps.get("Daily_Demand_AYY_tons", 0.0)) / total_daily
            else:
                base_ayy = float(r_fps.get("Monthly_Demand_AYY_tons", r_fps.get("AYY", 0.0)) or 0.0)
                base_phh = float(r_fps.get("Monthly_Demand_PHH_tons", r_fps.get("PHH", 0.0)) or 0.0)
                denom = base_ayy + base_phh
                ayy_ratio = (base_ayy / denom) if denom > 0 else 0.0
            ayy_ratio = max(0.0, min(1.0, ayy_ratio))
            ayy_t = qty * ayy_ratio
            phh_t = qty - ayy_t

            dispatch_lg_rows.append({
                "Day": int(day),
                "Vehicle_ID": chosen["Vehicle_ID"],
                "LG_ID": int(lgid),
                "FPS_ID": int(fid),
                "Quantity_tons": float(qty),
                "AYY_tons": float(ayy_t),
                "PHH_tons": float(phh_t),
            })

            lg_stock_total[lgid] = lg_stock_total.get(lgid, 0.0) - qty
            fps_stock_total[fid] = fps_stock_total.get(fid, 0.0) + qty
            vehicles.loc[vehicles["Vehicle_ID"] == vid, "Trips_Used"] += 1

        # end-of-day stock snapshot
        for lgid, st in lg_stock_total.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "LG",  "Entity_ID": int(lgid), "Stock_Level_tons": float(st)})
        for fid, st in fps_stock_total.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "FPS", "Entity_ID": int(fid),  "Stock_Level_tons": float(st)})

    dispatch_lg = pd.DataFrame(
        dispatch_lg_rows,
        columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AYY_tons","PHH_tons"]
    )
    stock_levels = pd.DataFrame(stock_rows, columns=["Day","Entity_Type","Entity_ID","Stock_Level_tons"])
    if dispatch_lg.empty:
        dispatch_lg = pd.DataFrame(columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AYY_tons","PHH_tons"])

    # -----------------------------------------------
    # 4) Required LG inflow per day (mirror from LG→FPS)
    # -----------------------------------------------
    # If nothing shipped LG→FPS, requirements are zeros
    if dispatch_lg.empty:
        lg_daily_req = (
            pd.MultiIndex.from_product([sorted(valid_lg_ids), range(1, DAYS + 1)], names=["LG_ID","Day"])
              .to_frame(index=False)
              .assign(Daily_Requirement_tons=0.0,
                      Daily_Requirement_AYY_tons=0.0,
                      Daily_Requirement_PHH_tons=0.0)
        )
    else:
        lg_daily_req = (
            dispatch_lg
            .groupby(["LG_ID", "Day"])[["Quantity_tons","AYY_tons","PHH_tons"]]
            .sum().reset_index()
            .rename(columns={
                "Quantity_tons": "Daily_Requirement_tons",
                "AYY_tons": "Daily_Requirement_AYY_tons",
                "PHH_tons": "Daily_Requirement_PHH_tons",
            })
        )

    # Pivots for lookup
    req_pivot_total = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                               values="Daily_Requirement_tons", aggfunc="sum", fill_value=0.0)
    req_pivot_ayy   = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                               values="Daily_Requirement_AYY_tons", aggfunc="sum", fill_value=0.0)
    req_pivot_phh   = lg_daily_req.pivot_table(index="LG_ID", columns="Day",
                                               values="Daily_Requirement_PHH_tons", aggfunc="sum", fill_value=0.0)

    # -----------------------------------------------
    # 5) CG → LG (MIRROR MODE: exact per-day totals & AYY/PHH split)
    # -----------------------------------------------
    dispatch_cg_rows = []

    # Helper to emit trips that exactly sum to (req_total, req_ayy, req_phh)
    def emit_mirrored_trips(day: int, lgid: int, req_total: float, req_ayy: float, req_phh: float):
        # Cycle vehicle IDs 1..TOT_V as we create trips; ignore per-day trip cap intentionally to match totals
        trips_needed = 0 if req_total <= 0 else int(math.floor(req_total / TRUCK_CAP))
        rem_total = req_total
        rem_ayy   = req_ayy
        rem_phh   = req_phh

        vid_counter = 0
        # Full-capacity trips
        while rem_total > TRUCK_CAP + 1e-9:
            qty = TRUCK_CAP
            denom = rem_ayy + rem_phh
            ayy_ratio = (rem_ayy / denom) if denom > 0 else 0.0
            ayy_t = min(qty, rem_ayy) if denom == 0 else qty * ayy_ratio
            ayy_t = max(0.0, min(qty, ayy_t))
            phh_t = qty - ayy_t

            vid_counter = (vid_counter % TOT_V) + 1
            dispatch_cg_rows.append({
                "Day": int(day),
                "Vehicle_ID": int(vid_counter),
                "LG_ID": int(lgid),
                "Quantity_tons": float(qty),
                "AYY_tons": float(ayy_t),
                "PHH_tons": float(phh_t),
            })
            rem_total -= qty
            rem_ayy   -= ayy_t
            rem_phh   -= phh_t

        # Last (possibly partial) trip
        if rem_total > 1e-9:
            qty = rem_total
            # Put the exact remaining split on the last trip to close any rounding drift
            ayy_t = max(0.0, rem_ayy)
            phh_t = max(0.0, qty - ayy_t) if rem_ayy <= qty else 0.0
            # Boundaries
            ayy_t = min(ayy_t, qty)
            phh_t = qty - ayy_t

            vid_counter = (vid_counter % TOT_V) + 1
            dispatch_cg_rows.append({
                "Day": int(day),
                "Vehicle_ID": int(vid_counter),
                "LG_ID": int(lgid),
                "Quantity_tons": float(qty),
                "AYY_tons": float(ayy_t),
                "PHH_tons": float(phh_t),
            })

    # Emit mirrored CG→LG rows for every LG & Day
    for lgid in req_pivot_total.index:
        for day in range(1, DAYS + 1):
            req_total = float(req_pivot_total.at[lgid, day]) if day in req_pivot_total.columns else 0.0
            req_ayy   = float(req_pivot_ayy.at[lgid, day])   if day in req_pivot_ayy.columns   else 0.0
            req_phh   = float(req_pivot_phh.at[lgid, day])   if day in req_pivot_phh.columns   else 0.0
            if req_total <= 0:
                continue
            emit_mirrored_trips(day, lgid, req_total, req_ayy, req_phh)

    dispatch_cg = pd.DataFrame(
        dispatch_cg_rows,
        columns=["Day","Vehicle_ID","LG_ID","Quantity_tons","AYY_tons","PHH_tons"]
    )
    if dispatch_cg.empty:
        dispatch_cg = pd.DataFrame(columns=["Day","Vehicle_ID","LG_ID","Quantity_tons","AYY_tons","PHH_tons"])

    return dispatch_cg, dispatch_lg, stock_levels
