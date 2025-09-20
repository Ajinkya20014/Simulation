# simulation.py
import math
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd


def run_simulation(
    master_workbook,          # str path or file-like buffer
    settings: pd.DataFrame,
    lgs: pd.DataFrame,
    fps: pd.DataFrame,
    vehicles: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Phase 1) LG → FPS dispatch (priority with per-vehicle trip caps)
        -> dispatch_lg columns:
           Day, Vehicle_ID, LG_ID, FPS_ID, Quantity_tons, AAY_Dispatched_tons, PHH_Dispatched_tons
        -> stock_levels (LG & FPS end-of-day stock)

    Phase 2) CG → LG pre-dispatch (meet LG daily requirement derived from phase 1)
        -> dispatch_cg columns: Day, Vehicle_ID, LG_ID, Quantity_tons
    """

    def _get_setting(param_name, default=None, cast=float):
        try:
            val = settings.loc[settings["Parameter"] == param_name, "Value"].iloc[0]
            return cast(val)
        except Exception:
            if default is None:
                raise ValueError(f"Missing required setting: {param_name}")
            return cast(default)

    DAYS         = _get_setting("Distribution_Days", 30, int)
    TRUCK_CAP    = _get_setting("Vehicle_Capacity_tons", 11.5, float)
    TOT_V        = _get_setting("Vehicles_Total", 30, int)
    MAX_TRIPS    = _get_setting("Max_Trips_Per_Vehicle_Per_Day", 3, int)
    DEFAULT_LEAD = _get_setting("Default_Lead_Time_days", 3, float)

    AAY_per_card_kg = _get_setting("AAY_per_card_kg", 0.0, float)
    PHH_per_ben_kg  = _get_setting("PHH_per_beneficiary_kg", 0.0, float)

    lgs = lgs.copy()
    if "LG_ID" not in lgs.columns or "LG_Name" not in lgs.columns:
        raise ValueError("LGs sheet must contain columns: LG_ID, LG_Name")

    lgid_by_name = {str(nm).strip().lower(): int(lg_id) for lg_id, nm in zip(lgs["LG_ID"], lgs["LG_Name"])}
    valid_lg_ids = set(int(x) for x in pd.to_numeric(lgs["LG_ID"], errors="coerce").dropna().astype(int))

    def normalize_lg_ref(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        try:
            i = int(float(s))
            return i if i in valid_lg_ids else None
        except ValueError:
            return lgid_by_name.get(s.lower())

    req_cols = {"FPS_ID", "Monthly_Demand_tons", "Max_Capacity_tons", "Linked_LG_ID"}
    missing = req_cols - set(fps.columns)
    if missing:
        raise ValueError(f"FPS sheet missing required columns: {missing}")

    fps = fps.copy()

    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def get_numeric_col(df: pd.DataFrame, candidates: list[str], default_value: float = 0.0) -> pd.Series:
        if not isinstance(df, pd.DataFrame):
            return pd.Series([], dtype=float)
        if df.empty:
            return pd.Series(0.0, index=df.index, dtype=float)

        lookup = {}
        for c in df.columns:
            k = _norm(c)
            if k not in lookup:
                lookup[k] = c

        for cand in candidates:
            key = _norm(cand)
            if key in lookup:
                col = df[lookup[key]]
                if not isinstance(col, pd.Series):
                    col = pd.Series(col, index=df.index)
                ser = pd.to_numeric(col, errors="coerce")
                return ser.fillna(default_value).astype(float)

        return pd.Series(default_value, index=df.index, dtype=float)

    lead_series = get_numeric_col(fps, ["Lead_Time_days"])
    lead_series = lead_series.where(~lead_series.isna(), other=DEFAULT_LEAD).fillna(DEFAULT_LEAD)
    fps["Lead_Time_days"] = lead_series

    monthly = get_numeric_col(fps, ["Monthly_Demand_tons"])
    fps["Monthly_Demand_tons"] = monthly
    fps["Daily_Demand_tons"] = fps["Monthly_Demand_tons"] / 30.0
    fps["Reorder_Threshold_tons"] = fps["Daily_Demand_tons"] * fps["Lead_Time_days"]

    fps["LG_ID"] = fps["Linked_LG_ID"].apply(normalize_lg_ref)
    if fps["LG_ID"].isna().any():
        bad_rows = fps[fps["LG_ID"].isna()][["FPS_ID", "Linked_LG_ID"]]
        raise ValueError(
            "Some FPS rows couldn't map Linked_LG_ID to a valid LG_ID. "
            f"Examples:\n{bad_rows.head(5).to_string(index=False)}\n"
            "Ensure Linked_LG_ID is either a valid LG_ID or a valid LG_Name."
        )
    fps["LG_ID"] = fps["LG_ID"].astype(int)

    aay_candidates = [
        "No. of AAY Cards", "No of AAY Cards", "AAY Cards", "AAY_Cards", "AAY cards",
    ]
    phh_candidates = [
        "No. of PHH Beneficiaries", "No. of PHH Benificiaries",
        "No of PHH Beneficiaries", "PHH Beneficiaries", "PHH Benificiaries", "PHH_Beneficiaries",
    ]
    aay_cards = get_numeric_col(fps, aay_candidates, default_value=0.0)
    phh_bens  = get_numeric_col(fps, phh_candidates,  default_value=0.0)

    ent_aay_series = (aay_cards * float(AAY_per_card_kg)) / 1000.0
    ent_phh_series = (phh_bens  * float(PHH_per_ben_kg))  / 1000.0

    ent_aay = dict(zip(fps["FPS_ID"], ent_aay_series))
    ent_phh = dict(zip(fps["FPS_ID"], ent_phh_series))

    deliv_aay = defaultdict(float)
    deliv_phh = defaultdict(float)

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
        cap_ser = get_numeric_col(vehicles, ["Capacity_tons"], default_value=TRUCK_CAP)
        vehicles["Capacity_tons"] = cap_ser.where(~cap_ser.isna(), other=TRUCK_CAP).fillna(TRUCK_CAP)
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
                    out.append(i)
                    continue
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

    if "Initial_Allocation_tons" not in lgs.columns:
        lgs["Initial_Allocation_tons"] = 0.0
    lgs["Initial_Allocation_tons"] = pd.to_numeric(lgs["Initial_Allocation_tons"], errors="coerce").fillna(0.0)

    lg_stock = {int(row["LG_ID"]): float(row["Initial_Allocation_tons"]) for _, row in lgs.iterrows()}
    fps_stock = {int(fid): 0.0 for fid in fps["FPS_ID"]}

    dispatch_lg_rows = []
    stock_rows = []

    def split_aay_phh(fid: int, q: float):
        if q <= 0:
            return 0.0, 0.0
        rem_aay = max(0.0, ent_aay.get(fid, 0.0) - deliv_aay[fid])
        rem_phh = max(0.0, ent_phh.get(fid, 0.0) - deliv_phh[fid])
        total_rem = rem_aay + rem_phh
        if total_rem <= 1e-12:
            return 0.0, 0.0

        aay_prop = q * (rem_aay / total_rem) if rem_aay > 0 else 0.0
        phh_prop = q - aay_prop

        aay = min(aay_prop, rem_aay)
        phh = min(phh_prop, rem_phh)

        used = aay + phh
        leftover = q - used
        if leftover > 1e-12:
            room_aay = max(0.0, rem_aay - aay)
            add_aay = min(leftover, room_aay)
            aay += add_aay
            leftover -= add_aay
            if leftover > 1e-12:
                room_phh = max(0.0, rem_phh - phh)
                add_phh = min(leftover, room_phh)
                phh += add_phh
                leftover -= add_phh

        diff = q - (aay + phh)
        if abs(diff) > 1e-9:
            if (rem_aay - aay) >= (rem_phh - phh):
                aay += diff
            else:
                phh += diff

        return float(aay), float(phh)

    for day in range(1, DAYS + 1):
        for _, r in fps.iterrows():
            fid = int(r["FPS_ID"])
            consume = float(r["Daily_Demand_tons"])
            fps_stock[fid] = max(0.0, fps_stock[fid] - consume)

        needs = []
        for _, r in fps.iterrows():
            fid = int(r["FPS_ID"])
            lgid = int(r["LG_ID"])
            current = fps_stock[fid]
            threshold = float(r["Reorder_Threshold_tons"])
            max_cap = float(r["Max_Capacity_tons"])
            if current <= threshold:
                available_at_lg = lg_stock.get(lgid, 0.0)
                need_qty = min(max_cap - current, available_at_lg)
                if need_qty > 0:
                    dd = float(r["Daily_Demand_tons"])
                    urgency = (threshold - current) / dd if dd > 0 else 0.0
                    needs.append((urgency, fid, lgid, need_qty))
        needs.sort(reverse=True, key=lambda x: x[0])

        vehicles["Trips_Used"] = 0

        for urgency, fid, lgid, need_qty in needs:
            cand = vehicles[vehicles["Mapped_LGs_List"].apply(lambda lst: lgid in lst)].copy()
            cand = cand[cand["Trips_Used"] < MAX_TRIPS]
            if cand.empty:
                continue

            cand["is_shared"] = cand["Mapped_LGs_List"].apply(lambda lst: len(lst) > 1)
            cand = cand.sort_values(["is_shared"], ascending=False)
            chosen = cand.iloc[0]

            vid = chosen["Vehicle_ID"]         # ← keep as-is (string/int), don't cast to int
            cap = float(chosen["Capacity_tons"])
            qty = min(cap, need_qty, lg_stock.get(lgid, 0.0))
            if qty <= 0:
                continue

            aay_t, phh_t = split_aay_phh(fid, qty)
            deliv_aay[fid] += aay_t
            deliv_phh[fid] += phh_t

            dispatch_lg_rows.append({
                "Day": int(day),
                "Vehicle_ID": vid,             # ← FIX: removed int() cast
                "LG_ID": int(lgid),
                "FPS_ID": int(fid),
                "Quantity_tons": float(qty),
                "AAY_Dispatched_tons": float(aay_t),
                "PHH_Dispatched_tons": float(phh_t),
            })

            lg_stock[lgid] = lg_stock.get(lgid, 0.0) - qty
            fps_stock[fid] = fps_stock.get(fid, 0.0) + qty
            vehicles.loc[vehicles["Vehicle_ID"] == vid, "Trips_Used"] += 1

        for lgid, st in lg_stock.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "LG",  "Entity_ID": int(lgid), "Stock_Level_tons": float(st)})
        for fid, st in fps_stock.items():
            stock_rows.append({"Day": int(day), "Entity_Type": "FPS", "Entity_ID": int(fid),  "Stock_Level_tons": float(st)})

    dispatch_lg = pd.DataFrame(
        dispatch_lg_rows,
        columns=["Day", "Vehicle_ID", "LG_ID", "FPS_ID", "Quantity_tons", "AAY_Dispatched_tons", "PHH_Dispatched_tons"]
    )
    stock_levels = pd.DataFrame(stock_rows, columns=["Day","Entity_Type","Entity_ID","Stock_Level_tons"])

    if dispatch_lg.empty:
        dispatch_lg = pd.DataFrame(columns=["Day","Vehicle_ID","LG_ID","FPS_ID","Quantity_tons","AAY_Dispatched_tons","PHH_Dispatched_tons"])

    if dispatch_lg.empty:
        lg_daily_req = (
            pd.MultiIndex.from_product([sorted(valid_lg_ids), range(1, DAYS + 1)], names=["LG_ID","Day"])
            .to_frame(index=False)
            .assign(Daily_Requirement_tons=0.0)
        )
    else:
        tmp = dispatch_lg.copy()
        tmp["LG_ID"] = pd.to_numeric(tmp["LG_ID"], errors="coerce").astype("Int64")
        tmp["Day"]   = pd.to_numeric(tmp["Day"], errors="coerce").astype("Int64")
        lg_daily_req = (
            tmp.groupby(["LG_ID", "Day"], dropna=True)["Quantity_tons"]
               .sum()
               .reset_index()
               .rename(columns={"Quantity_tons": "Daily_Requirement_tons"})
        )

    req_pivot = lg_daily_req.pivot_table(
        index="LG_ID", columns="Day",
        values="Daily_Requirement_tons",
        aggfunc="sum", fill_value=0.0
    )

    try:
        cap_df = pd.read_excel(master_workbook, sheet_name="LG_Capacity")
        if {"LG_ID", "Capacity_tons"} <= set(cap_df.columns):
            capacity = {int(r["LG_ID"]): float(r["Capacity_tons"]) for _, r in cap_df.iterrows()}
        else:
            raise ValueError
    except Exception:
        if "Storage_Capacity_tons" not in lgs.columns:
            raise ValueError("Provide LG_Capacity sheet or 'Storage_Capacity_tons' in LGs.")
        capacity = {int(r["LG_ID"]): float(r["Storage_Capacity_tons"]) for _, r in lgs.iterrows()}

    if "Initial_LG_stock" in lgs.columns:
        lgs["Initial_LG_stock"] = pd.to_numeric(lgs["Initial_LG_stock"], errors="coerce").fillna(0.0)
    else:
        lgs["Initial_LG_stock"] = 0.0
    lg_stock_base = {int(r["LG_ID"]): float(r["Initial_LG_stock"]) for _, r in lgs.iterrows()}

    req_pivot = req_pivot.copy()
    req_pivot.index = [int(x) for x in req_pivot.index]
    req_pivot.columns = [int(c) for c in req_pivot.columns]

    lg_ids = list(req_pivot.index)

    def _get_demand(lg_id: int, day: int) -> float:
        try:
            return float(req_pivot.at[lg_id, day])
        except Exception:
            return 0.0

    def _free_room(stock: dict, lg_id: int) -> float:
        return max(0.0, capacity.get(lg_id, 0.0) - stock.get(lg_id, 0.0))

    def _simulate(pre_days: int, collect_rows: bool = False, include_pre_days: bool = False):
        start_day = 1 - pre_days
        stock = {lg: lg_stock_base.get(lg, 0.0) for lg in lg_ids}
        rows = [] if collect_rows else None

        for day in range(start_day, DAYS + 1):
            trips_left = TOT_V

            if day >= 1:
                order = sorted(lg_ids, key=lambda lg: -(_get_demand(lg, day) - stock[lg]))
                for lg in order:
                    demand_today = _get_demand(lg, day)
                    need_today = max(0.0, demand_today - stock[lg])

                    while trips_left > 0 and need_today > 1e-9:
                        room = _free_room(stock, lg)
                        if room <= 1e-9:
                            break
                        qty = min(TRUCK_CAP, need_today, room)
                        if qty <= 1e-9:
                            break

                        if collect_rows and (include_pre_days or day >= 1):
                            vid = TOT_V - trips_left + 1
                            rows.append({
                                "Day": int(day),
                                "Vehicle_ID": int(vid),
                                "LG_ID": int(lg),
                                "Quantity_tons": float(qty)
                            })

                        stock[lg] += qty
                        trips_left -= 1
                        need_today -= qty

                    if stock[lg] + 1e-6 < demand_today:
                        return False, (rows or []), start_day, stock

            if trips_left > 0:
                future_unmet = {
                    lg: max(0.0, sum(_get_demand(lg, d) for d in range(max(1, day + 1), DAYS + 1)) - stock[lg])
                    for lg in lg_ids
                }
                candidates = [lg for lg, fu in future_unmet.items() if fu > 1e-6 and _free_room(stock, lg) > 1e-6]
                idx = 0
                while trips_left > 0 and candidates:
                    lg = candidates[idx % len(candidates)]
                    room = _free_room(stock, lg)
                    deliver = min(TRUCK_CAP, future_unmet[lg], room)

                    if deliver > 1e-9:
                        if collect_rows and (include_pre_days or day >= 1):
                            vid = TOT_V - trips_left + 1
                            rows.append({
                                "Day": int(day),
                                "Vehicle_ID": int(vid),
                                "LG_ID": int(lg),
                                "Quantity_tons": float(deliver)
                            })
                        stock[lg] += deliver
                        future_unmet[lg] = max(0.0, future_unmet[lg] - deliver)
                        trips_left -= 1

                    if future_unmet[lg] < 1e-6 or _free_room(stock, lg) < 1e-6:
                        candidates.remove(lg)
                        idx -= 1
                    idx += 1

            if day >= 1:
                for lg in lg_ids:
                    stock[lg] = max(0.0, stock[lg] - _get_demand(lg, day))

        return True, (rows or []), start_day, stock

    MAX_PRE_DAYS = 30
    pre_days = None
    for x in range(0, MAX_PRE_DAYS + 1):
        ok, _, start_day, _ = _simulate(pre_days=x, collect_rows=False)
        if ok:
            pre_days = x
            break
    if pre_days is None:
        raise RuntimeError("Unable to meet all demands within MAX_PRE_DAYS.")

    ok, rows, start_day, _ = _simulate(pre_days=pre_days, collect_rows=True, include_pre_days=True)
    assert ok

    dispatch_cg = pd.DataFrame(rows, columns=["Day", "Vehicle_ID", "LG_ID", "Quantity_tons"])

    lg_ids_sorted = sorted(int(x) for x in pd.to_numeric(lgs["LG_ID"], errors="coerce").dropna().astype(int).unique())

    if "Initial_LG_stock" in lgs.columns:
        init_series = (
            lgs.assign(LG_ID=lgs["LG_ID"].astype(int))
               .set_index("LG_ID")["Initial_LG_stock"]
               .reindex(lg_ids_sorted).fillna(0.0)
        )
    else:
        init_series = pd.Series(0.0, index=lg_ids_sorted)

    if not dispatch_cg.empty:
        dcg = dispatch_cg.copy()
        dcg["LG_ID"] = pd.to_numeric(dcg["LG_ID"], errors="coerce").astype(int)
        dcg["Day"]   = pd.to_numeric(dcg["Day"], errors="coerce").astype(int)
        cg_piv = dcg.pivot_table(index="LG_ID", columns="Day",
                                 values="Quantity_tons", aggfunc="sum", fill_value=0.0)
        full_cols = list(range(start_day, DAYS + 1))
        cg_piv = cg_piv.reindex(index=lg_ids_sorted, columns=full_cols, fill_value=0.0)
        cg_cum = cg_piv.cumsum(axis=1).reindex(columns=list(range(1, DAYS + 1)), fill_value=0.0)
    else:
        cg_cum = pd.DataFrame(0.0, index=lg_ids_sorted, columns=list(range(1, DAYS + 1)))

    if not dispatch_lg.empty:
        dlg = dispatch_lg.copy()
        dlg["LG_ID"] = pd.to_numeric(dlg["LG_ID"], errors="coerce").astype(int)
        dlg["Day"]   = pd.to_numeric(dlg["Day"], errors="coerce").astype(int)
        lg_piv = dlg.pivot_table(index="LG_ID", columns="Day",
                                 values="Quantity_tons", aggfunc="sum", fill_value=0.0)
        lg_piv = lg_piv.reindex(index=lg_ids_sorted, columns=list(range(1, DAYS + 1)), fill_value=0.0)
        lg_cum = lg_piv.cumsum(axis=1)
    else:
        lg_cum = pd.DataFrame(0.0, index=lg_ids_sorted, columns=list(range(1, DAYS + 1)))

    stock_matrix = init_series.to_numpy()[:, None] + cg_cum.to_numpy() - lg_cum.to_numpy()
    stock_matrix = np.where(np.abs(stock_matrix) < 1e-9, 0.0, stock_matrix)

    lg_stock_levels = (
        pd.DataFrame(stock_matrix, index=lg_ids_sorted, columns=list(range(1, DAYS + 1)))
          .stack().rename("Stock_Level_tons")
          .rename_axis(index=["LG_ID", "Day"]).reset_index()
          .rename(columns={"LG_ID": "Entity_ID"})
          .assign(Entity_Type="LG")[["Day", "Entity_Type", "Entity_ID", "Stock_Level_tons"]]
    )

    pre_cols = list(range(start_day, 1))
    if pre_cols:
        if not dispatch_cg.empty:
            cg_pre = cg_piv.reindex(index=lg_ids_sorted, columns=pre_cols, fill_value=0.0)
            cg_pre_cum = cg_pre.cumsum(axis=1)
        else:
            cg_pre_cum = pd.DataFrame(0.0, index=lg_ids_sorted, columns=pre_cols)

        stock_pre_matrix = init_series.to_numpy()[:, None] + cg_pre_cum.to_numpy()
        stock_pre_matrix = np.where(np.abs(stock_pre_matrix) < 1e-9, 0.0, stock_pre_matrix)

        lg_stock_levels_pre = (
            pd.DataFrame(stock_pre_matrix, index=lg_ids_sorted, columns=pre_cols)
              .stack().rename("Stock_Level_tons")
              .rename_axis(index=["LG_ID", "Day"]).reset_index()
              .rename(columns={"LG_ID": "Entity_ID"})
              .assign(Entity_Type="LG")[["Day", "Entity_Type", "Entity_ID", "Stock_Level_tons"]]
        )
        lg_stock_levels = pd.concat([lg_stock_levels_pre, lg_stock_levels], ignore_index=True)

    stock_levels = pd.concat(
        [stock_levels[stock_levels["Entity_Type"] == "FPS"], lg_stock_levels],
        ignore_index=True
    )

    return dispatch_cg, dispatch_lg, stock_levels
