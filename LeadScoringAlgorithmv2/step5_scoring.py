#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5 — Scoring (Compute Buying Intent per Lead) + Explanations

Reads:
  - features_master.csv  (output of step4_merge_datasets.py)

Writes:
  - scores_master.csv    (full scored table with component columns)
  - top_leads.csv        (top-N compact table)
  - scores_explain.csv   (human-friendly explanation of drivers for each lead)

Key points:
  - Default lead/company blend is 50% / 50% (changeable via CLI)
  - Robust percentile clipping (5th/95th) used for scaling features
  - Time-window weighting: 7d > 30d > 90d
  - Hiring component blends intensity, counts, relevance and optional size-normalization
  - Explanations include numeric contributions and top reasons
"""

from __future__ import annotations
import argparse
import os
import sys
import re
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
import pandas as pd

# ----------------------------
# I/O helpers
# ----------------------------

def read_csv_safe(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"❌ Missing required file: {label} -> {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {label}: {path} ({len(df):,} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        print(f"❌ Failed to read {label}: {path} ({e})")
        sys.exit(1)

def write_csv_safe(df: pd.DataFrame, path: str, label: str):
    try:
        df.to_csv(path, index=False)
        print(f"✅ Saved {label} to: {path} ({len(df):,} rows)")
    except Exception as e:
        print(f"❌ Failed to write {label}: {path} ({e})")
        sys.exit(1)

# ----------------------------
# Column/regex utilities
# ----------------------------

PERSON_ID_CANDIDATES = ["person_id", "profileUrl", "linkedin_profile_url"]

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """Return df columns matching ANY of the regexes in patterns, preserving column order."""
    rx = [re.compile(p) for p in patterns]
    out: List[str] = []
    for c in df.columns:
        for r in rx:
            if r.search(c):
                out.append(c)
                break
    return out

def numeric_only(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))]

# ----------------------------
# Robust normalization & window combiners
# ----------------------------

def robust_minmax(s: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """
    Robust [0,1] scaling by clipping to quantiles; returns 0 for constants or all-NaN.
    Uses q_low and q_high quantiles (default 5th/95th).
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    lo = float(np.nanquantile(s, q_low))
    hi = float(np.nanquantile(s, q_high))
    if not np.isfinite(lo):
        lo = float(np.nanmin(s))
    if not np.isfinite(hi):
        hi = float(np.nanmax(s))
    if hi <= lo:
        # constant or degenerate -> return zeros
        return pd.Series(0.0, index=s.index)
    s_clip = s.clip(lower=lo, upper=hi)
    out = ((s_clip - lo) / (hi - lo)).fillna(0.0)
    return out

def weighted_sum_scaled(df: pd.DataFrame, columns: List[Tuple[str, float]]) -> pd.Series:
    """
    For each (col, weight), robust-scale col to [0,1], multiply by weight, and sum.
    Missing cols are silently skipped.
    """
    parts = []
    for col, w in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            parts.append(robust_minmax(df[col]) * float(w))
    if not parts:
        return pd.Series(0.0, index=df.index)
    arr = np.sum(parts, axis=0)
    return pd.Series(arr, index=df.index).fillna(0.0)

def combine_time_windows(
    df: pd.DataFrame,
    base_patterns: List[str],
    window_weights: Dict[str, float],
    co_prefix: Optional[str] = None
) -> Tuple[pd.Series, Dict[str, List[str]]]:
    """
    Auto-detect available time-windowed columns and combine them with weights.
    base_patterns should contain regex strings with {win} placeholder for window labels (7d/30d/90d).
    co_prefix if provided is inserted before each pattern (useful for company columns prefixed by 'co_').

    Returns:
      - combined_series (values roughly in [0,1])
      - dictionary of used columns per window
    """
    wcols: Dict[str, List[str]] = {w: [] for w in window_weights}
    total = pd.Series(0.0, index=df.index)
    for win, w in window_weights.items():
        ptns = []
        for p in base_patterns:
            ptn = p.replace("{win}", win)
            if co_prefix:
                # ensure anchoring of start ^
                if ptn.startswith("^"):
                    ptn = "^" + co_prefix + ptn[1:]
                else:
                    ptn = "^" + co_prefix + ptn
            ptns.append(ptn)
        cols = find_cols(df, ptns)
        cols = numeric_only(df, cols)
        wcols[win] = cols
        if cols:
            s = df[cols].sum(axis=1, min_count=1)
            total = total + robust_minmax(s) * float(w)
    return total.fillna(0.0), wcols

# ----------------------------
# Decision-maker / title boost
# ----------------------------

DECISION_MAKER_PATTERNS = {
    r"\b(cfo|chief\s*financial\s*officer)\b": 1.00,
    r"\b(finance\s*director|head\s*of\s*finance|finance\s*manager)\b": 0.80,
    r"\b(fc|financial\s*controller)\b": 0.70,
    r"\b(head\s*of\s*accounting|accounting\s*manager)\b": 0.60,
    r"\b(partner|managing\s*partner|director)\b": 0.50,
}

def decision_maker_boost(text: Any) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower().strip()
    for rx, score in DECISION_MAKER_PATTERNS.items():
        if re.search(rx, t):
            return float(score)
    return 0.0

def decision_maker_series(df: pd.DataFrame) -> pd.Series:
    title_cols = [c for c in df.columns if re.search(r"(?:^|_)(title|job\s*title|headline)$", c, flags=re.I)]
    if not title_cols:
        return pd.Series(0.0, index=df.index)
    preferred = [c for c in title_cols if re.search(r"(^current.*title$|^job.*title$|^headline$)", c, flags=re.I)]
    col = (preferred or title_cols)[0]
    return df[col].astype(str).map(decision_maker_boost).fillna(0.0)

# ----------------------------
# Hiring component builder (detailed)
# ----------------------------

def hiring_component(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, List[str]]]:
    """
    Build a company hiring signal by blending across time windows (7d > 30d > 90d).
    Uses intensity, total score, and relevant counts as main positives,
    adjacent counts as mild positives, and irrelevant counts as a penalty.
    """
    window_weights = {"7d": 1.0, "30d": 0.7, "90d": 0.4}
    used: Dict[str, List[str]] = {}

    # Base patterns for hiring features
    base_patterns = {
        "intensity": r"^co_hiring_intensity_{win}$",
        "total": r"^co_total_hiring_score_{win}$",
        "relevant": r"^co_relevant_hiring_count_{win}$",
        "adjacent": r"^co_adjacent_hiring_count_{win}$",
        "irrelevant": r"^co_irrelevant_hiring_count_{win}$",
    }

    # Container for blended series
    blended = {k: pd.Series(0.0, index=df.index) for k in base_patterns}

    # Loop per window
    for win, w in window_weights.items():
        for key, pattern in base_patterns.items():
            cols = find_cols(df, [pattern.replace("{win}", win)])
            cols = numeric_only(df, cols)
            if not cols:
                continue
            s = df[cols].sum(axis=1, min_count=1)
            blended[key] = blended[key] + robust_minmax(s) * w
            used.setdefault(key, []).extend(cols)

    # Normalize each blended component to 0–1
    s_intensity = robust_minmax(blended["intensity"])
    s_total     = robust_minmax(blended["total"])
    s_rel       = robust_minmax(blended["relevant"])
    s_adj       = robust_minmax(blended["adjacent"])
    s_irr       = robust_minmax(blended["irrelevant"])

    # Blend with weights
    mix = (
        0.50 * s_intensity +
        0.20 * s_total +
        0.20 * s_rel +
        0.10 * s_adj -
        0.10 * s_irr
    )

    out = mix.clip(0.0, 1.0)
    return out, used

# ----------------------------
# Build lead & company scores
# ----------------------------

def build_lead_score(
    df: pd.DataFrame,
    lead_kw_weight: float = 0.60,
    lead_act_weight: float = 0.40
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    used: Dict[str, List[str]] = {}

    kw_windows = {"7d": 1.0, "30d": 0.70, "90d": 0.40}
    lead_kw_base_patterns = [r"^keyword_intent_score_{win}$", r"^kw_total_{win}$"]
    lead_kw_mix, used_kw = combine_time_windows(df, lead_kw_base_patterns, kw_windows, co_prefix=None)
    used["lead_kw_windows"] = [c for cols in used_kw.values() for c in cols]

    act_windows = {"7d": 1.0, "30d": 0.70, "90d": 0.40}
    lead_act_patterns = [r"^(?:n_)?actions_{win}$", r"^(?:n_)?engagements_{win}$", r"^(?:n_)?posts_{win}$", r"^(?:n_)?activities_{win}$"]
    lead_act_mix, used_act = combine_time_windows(df, lead_act_patterns, act_windows, co_prefix=None)
    used["lead_act_windows"] = [c for cols in used_act.values() for c in cols]

    tot_cols = find_cols(df, [r"^n_(?:actions|engagements|posts|activities)_total$"])
    used["lead_totals"] = tot_cols
    tot_boost = (robust_minmax(df[tot_cols].sum(axis=1, min_count=1)) * 0.20) if tot_cols else pd.Series(0.0, index=df.index)

    dm = decision_maker_series(df)  # [0..1]
    lead_profile_boost_01 = (dm * 0.10).clip(0.0, 0.10)

    lead_kw_component_01 = lead_kw_mix.clip(0.0, 1.0)
    lead_act_component_01 = (lead_act_mix + tot_boost * 0.25).clip(0.0, 1.0)

    lead_score_01 = (lead_kw_weight * lead_kw_component_01 +
                     lead_act_weight * lead_act_component_01 +
                     lead_profile_boost_01).clip(0.0, 1.0)

    out = pd.DataFrame({
        "lead_kw_component_01": lead_kw_component_01,
        "lead_act_component_01": lead_act_component_01,
        "lead_profile_boost_01": lead_profile_boost_01,
        "lead_score_01": lead_score_01,
        "lead_score": (lead_score_01 * 100.0).round(3),
    }, index=df.index)

    return out, used

def build_company_score(
    df: pd.DataFrame,
    co_kw_weight: float = 0.40,
    co_act_weight: float = 0.20,
    co_hiring_weight: float = 0.40
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    used: Dict[str, List[str]] = {}

    kw_windows = {"7d": 1.0, "30d": 0.70, "90d": 0.40}
    co_kw_patterns = [r"^keyword_intent_score_{win}$", r"^kw_total_{win}$"]
    co_kw_mix, used_kw = combine_time_windows(df, co_kw_patterns, kw_windows, co_prefix="co_")
    used["co_kw_windows"] = [c for cols in used_kw.values() for c in cols]

    act_windows = {"7d": 1.0, "30d": 0.70, "90d": 0.40}
    co_act_patterns = [r"^(?:n_)?actions_{win}$", r"^(?:n_)?engagements_{win}$", r"^(?:n_)?posts_{win}$", r"^(?:n_)?activities_{win}$"]
    co_act_mix, used_act = combine_time_windows(df, co_act_patterns, act_windows, co_prefix="co_")
    used["co_act_windows"] = [c for cols in used_act.values() for c in cols]

    co_tot_cols = find_cols(df, [r"^co_n_(?:actions|engagements|posts|activities)_total$"])
    used["co_totals"] = co_tot_cols
    co_tot_boost = (robust_minmax(df[co_tot_cols].sum(axis=1, min_count=1)) * 0.20) if co_tot_cols else pd.Series(0.0, index=df.index)

    co_hiring_01, used_hiring = hiring_component(df)
    used["co_hiring"] = [x for v in used_hiring.values() for x in v]

    co_kw_component_01  = co_kw_mix.clip(0.0, 1.0)
    co_act_component_01 = (co_act_mix + co_tot_boost * 0.25).clip(0.0, 1.0)
    co_hiring_component_01 = co_hiring_01.clip(0.0, 1.0)

    co_score_01 = (co_kw_weight * co_kw_component_01 +
                   co_act_weight * co_act_component_01 +
                   co_hiring_weight * co_hiring_component_01).clip(0.0, 1.0)

    out = pd.DataFrame({
        "co_kw_component_01": co_kw_component_01,
        "co_act_component_01": co_act_component_01,
        "co_hiring_component_01": co_hiring_component_01,
        "company_score_01": co_score_01,
        "company_score": (co_score_01 * 100.0).round(3),
    }, index=df.index)

    return out, used

def build_overall_score(
    lead_score_01: pd.Series,
    company_score_01: pd.Series,
    lead_kw_component_01: pd.Series,
    co_kw_component_01: pd.Series,
    w_lead: float = 0.50,
    w_company: float = 0.50,
    synergy_w: float = 0.15
) -> pd.DataFrame:
    synergy = (pd.concat([lead_kw_component_01, co_kw_component_01], axis=1).min(axis=1) * synergy_w).clip(0.0, synergy_w)
    overall_01 = (w_lead * lead_score_01 + w_company * company_score_01 + synergy).clip(0.0, 1.0)
    return pd.DataFrame({
        "synergy_component_01": synergy,
        "overall_score_01": overall_01,
        "overall_score": (overall_01 * 100.0).round(3),
    }, index=lead_score_01.index)

# ----------------------------
# Explanation builder
# ----------------------------

def _format_pct(x: float) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "0.0%"

def build_explanation_row(row: pd.Series, person_id_col: str) -> Dict[str, Any]:
    """
    Build a detailed explanation for a single lead row.
    Returns a dict with keys we'll write to scores_explain.csv
    """
    # Gather numeric contributions (all on 0..1 scale except final percentage fields)
    lead_kw = float(row.get("lead_kw_component_01", 0.0))
    lead_act = float(row.get("lead_act_component_01", 0.0))
    lead_prof = float(row.get("lead_profile_boost_01", 0.0))
    lead_score01 = float(row.get("lead_score_01", 0.0))

    co_kw = float(row.get("co_kw_component_01", 0.0))
    co_act = float(row.get("co_act_component_01", 0.0))
    co_hir = float(row.get("co_hiring_component_01", 0.0))
    co_score01 = float(row.get("company_score_01", 0.0))

    synergy = float(row.get("synergy_component_01", 0.0))
    overall01 = float(row.get("overall_score_01", 0.0))

    # Component contributions to final overall score
    # Respect default weights used above (w_lead=0.5,w_company=0.5, synergy included separately)
    contrib_lead = 0.5 * lead_score01
    contrib_company = 0.5 * co_score01
    contrib_synergy = synergy
    total_calc = contrib_lead + contrib_company + contrib_synergy

    # Build human readable reasons
    reasons = []
    drivers = []  # tuples (label, magnitude) for top-n selection

    # Lead side analysis
    if lead_kw >= 0.6:
        reasons.append("Strong lead keyword intent (recent/sustained)")
        drivers.append(("Lead keywords", lead_kw))
    elif lead_kw >= 0.25:
        reasons.append("Moderate lead keyword mentions")
        drivers.append(("Lead keywords", lead_kw))

    if lead_act >= 0.6:
        reasons.append("High recent lead activity (posting/engaging)")
        drivers.append(("Lead activity", lead_act))
    elif lead_act >= 0.25:
        reasons.append("Moderate recent lead activity")
        drivers.append(("Lead activity", lead_act))

    if lead_prof > 0.0:
        reasons.append("Lead appears to be a decision-maker based on title")
        drivers.append(("Decision-maker boost", lead_prof))

    # Company side analysis
    if co_hir >= 0.5:
        # if hiring component is strong, try to include role context if available
        uniq_roles = row.get("co_unique_relevant_roles", None)
        if pd.notna(uniq_roles) and int(uniq_roles) > 0:
            reasons.append(f"Company actively hiring for relevant roles ({int(uniq_roles)} unique)")
        else:
            reasons.append("Company hiring intensity is strong")
        drivers.append(("Company hiring", co_hir))
    elif co_hir >= 0.25:
        reasons.append("Company shows moderate hiring activity")
        drivers.append(("Company hiring", co_hir))

    if co_kw >= 0.6:
        reasons.append("Company-level keyword intent is strong")
        drivers.append(("Company keywords", co_kw))
    elif co_kw >= 0.25:
        reasons.append("Some company-level keyword mentions")
        drivers.append(("Company keywords", co_kw))

    if co_act >= 0.6:
        reasons.append("Company has high content activity (posts/shares)")
        drivers.append(("Company activity", co_act))
    elif co_act >= 0.25:
        reasons.append("Company activity is moderate")
        drivers.append(("Company activity", co_act))

    if synergy > 0.0:
        reasons.append("Lead and company show aligned keyword signals (synergy)")

    if not reasons:
        reasons = ["Low activity and weak keyword/hiring signals"]

    # Pick top 2 numeric drivers by magnitude
    drivers_sorted = sorted(drivers, key=lambda x: x[1], reverse=True)
    top_drivers = drivers_sorted[:2]
    top_driver_texts = [f"{label} ({_format_pct(m)})" for label, m in top_drivers] if top_drivers else []

    # Create an explanation summary with some numeric details
    expl = {
        person_id_col: row.get(person_id_col),
        "lead_score": round(lead_score01 * 100.0, 3),
        "company_score": round(co_score01 * 100.0, 3),
        "overall_score": round(overall01 * 100.0, 3),
        "contrib_lead_pct": round(contrib_lead * 100.0, 3),
        "contrib_company_pct": round(contrib_company * 100.0, 3),
        "contrib_synergy_pct": round(contrib_synergy * 100.0, 3),
        "top_drivers": "; ".join(top_driver_texts) if top_driver_texts else "",
        "reasons": " | ".join(reasons),
        # also give a short diagnostics string that may help triage
        "diagnostics": f"lead_kw={_format_pct(lead_kw)}, lead_act={_format_pct(lead_act)}, co_kw={_format_pct(co_kw)}, co_act={_format_pct(co_act)}, co_hir={_format_pct(co_hir)}"
    }

    # If hiring detail columns exist, attach them for context (prefer 7d window if present)
    if "co_total_hiring_score_7d" in row.index and not pd.isna(row["co_total_hiring_score_7d"]):
        expl["co_total_hiring_score_7d"] = float(row["co_total_hiring_score_7d"])
    if "co_hiring_posts_7d" in row.index and not pd.isna(row["co_hiring_posts_7d"]):
        expl["co_hiring_posts_7d"] = int(row["co_hiring_posts_7d"])
    if "co_unique_relevant_roles_7d" in row.index and not pd.isna(row["co_unique_relevant_roles_7d"]):
        expl["co_unique_relevant_roles_7d"] = int(row["co_unique_relevant_roles_7d"])

    return expl

def build_explanations_table(scored: pd.DataFrame, person_id_col: str) -> pd.DataFrame:
    rows = []
    for _, r in scored.iterrows():
        rows.append(build_explanation_row(r, person_id_col))
    expl_df = pd.DataFrame(rows)
    return expl_df

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_master.csv", help="Input merged feature table")
    ap.add_argument("--out", default="scores_master.csv", help="Full scored output")
    ap.add_argument("--top_out", default="top_leads.csv", help="Top-N compact output")
    ap.add_argument("--explain_out", default="scores_explain.csv", help="Per-lead explanations")
    ap.add_argument("--top_n", type=int, default=50)

    # Top-level blending weights (default to 50/50 per your instruction)
    ap.add_argument("--w_lead", type=float, default=0.50, help="Weight for lead-level score in overall")
    ap.add_argument("--w_company", type=float, default=0.50, help="Weight for company-level score in overall")
    ap.add_argument("--synergy_w", type=float, default=0.15, help="Synergy multiplier added to overall")

    # Lead component weights (internal to lead component)
    ap.add_argument("--lead_kw_w", type=float, default=0.60)
    ap.add_argument("--lead_act_w", type=float, default=0.40)

    # Company component weights
    ap.add_argument("--co_kw_w", type=float, default=0.40)
    ap.add_argument("--co_act_w", type=float, default=0.20)
    ap.add_argument("--co_hiring_w", type=float, default=0.40)

    args = ap.parse_args()

    df = read_csv_safe(args.features, "features_master")
    person_id = pick_first_existing(df, PERSON_ID_CANDIDATES)
    if not person_id:
        print(f"❌ Could not find a person_id-like column. Checked: {PERSON_ID_CANDIDATES}")
        sys.exit(1)

    # Deduplicate (keep first)
    dupe_ct = df.duplicated(subset=[person_id]).sum()
    if dupe_ct > 0:
        print(f"⚠️  Found {dupe_ct:,} duplicate person_id rows; keeping first occurrence.")
        df = df.drop_duplicates(subset=[person_id], keep="first").reset_index(drop=True)

    # Build lead component
    lead_df, used_lead = build_lead_score(df, lead_kw_weight=args.lead_kw_w, lead_act_weight=args.lead_act_w)

    # Build company component
    co_df, used_co = build_company_score(df, co_kw_weight=args.co_kw_w, co_act_weight=args.co_act_w, co_hiring_weight=args.co_hiring_w)

    # Build overall
    overall_df = build_overall_score(
        lead_df["lead_score_01"],
        co_df["company_score_01"],
        lead_df["lead_kw_component_01"],
        co_df["co_kw_component_01"],
        w_lead=args.w_lead,
        w_company=args.w_company,
        synergy_w=args.synergy_w
    )

    # Assemble final scored table
    scored = pd.concat([df.reset_index(drop=True), lead_df.reset_index(drop=True), co_df.reset_index(drop=True), overall_df.reset_index(drop=True)], axis=1)

    # Ranking
    scored["rank_overall"] = scored["overall_score"].rank(method="first", ascending=False).astype(int)
    scored["rank_lead"] = scored["lead_score"].rank(method="first", ascending=False).astype(int)
    scored["rank_company"] = scored["company_score"].rank(method="first", ascending=False).astype(int)

    # Coverage diagnostics
    n = len(scored)
    def coverage(cols: List[str]) -> int:
        cols = [c for c in cols if c in scored.columns]
        if not cols:
            return 0
        return scored[cols].notna().any(axis=1).sum()

    print("\n=== Coverage diagnostics ===")
    print(f"Leads scored: {n:,}")
    print(f"Leads with LEAD keyword windows present: {coverage(used_lead.get('lead_kw_windows', [])):,} / {n:,}")
    print(f"Leads with LEAD activity windows present: {coverage(used_lead.get('lead_act_windows', [])):,} / {n:,}")
    print(f"Leads with COMPANY keyword windows present: {coverage(used_co.get('co_kw_windows', [])):,} / {n:,}")
    print(f"Leads with COMPANY activity windows present: {coverage(used_co.get('co_act_windows', [])):,} / {n:,}")
    print(f"Leads linked to COMPANY hiring features: {coverage(used_co.get('co_hiring', [])):,} / {n:,}")

    # Score summaries
    def describe(col: str):
        s = scored[col]
        return {
            "mean": round(float(np.nanmean(s)), 3),
            "p50": round(float(np.nanquantile(s, 0.50)), 3),
            "p75": round(float(np.nanquantile(s, 0.75)), 3),
            "p90": round(float(np.nanquantile(s, 0.90)), 3),
            "max": round(float(np.nanmax(s)), 3),
        }

    print("\n=== Score summaries (0-100 scale) ===")
    print("Lead score   :", describe("lead_score"))
    print("Company score:", describe("company_score"))
    print("Overall score:", describe("overall_score"))

    # Save full scored table
    write_csv_safe(scored, args.out, "scored leads (full)")

    # Top-N compact table
    top_n = max(1, int(args.top_n))
    company_name_cols = [c for c in ["co_company_name", "co_name", "co_companyName", "company_name"] if c in scored.columns]
    col_company = company_name_cols[0] if company_name_cols else None

    show_cols = [person_id]
    if col_company:
        show_cols.append(col_company)
    show_cols += ["lead_score", "company_score", "overall_score", "rank_overall"]
    # add helpful components if present
    for c in ["lead_kw_component_01", "lead_act_component_01", "lead_profile_boost_01", "co_kw_component_01", "co_act_component_01", "co_hiring_component_01"]:
        if c in scored.columns:
            show_cols.append(c)

    top = scored.sort_values("overall_score", ascending=False).head(top_n)[show_cols].copy()
    write_csv_safe(top, args.top_out, f"top-{top_n} leads (compact)")

    # Build explanations (detailed)
    print("\nBuilding explanations...")
    explain_df = build_explanations_table(scored, person_id)
    # Reorder explanation columns for readability if possible
    preferred_order = [person_id, "overall_score", "lead_score", "company_score", "contrib_lead_pct", "contrib_company_pct", "contrib_synergy_pct", "top_drivers", "reasons", "diagnostics"]
    rest = [c for c in explain_df.columns if c not in preferred_order]
    explain_df = explain_df[[c for c in preferred_order if c in explain_df.columns] + rest]
    write_csv_safe(explain_df, args.explain_out, "scores explanations")

    # Quick preview
    with pd.option_context("display.max_columns", 160, "display.width", 220):
        print("\nTop 5 leads (preview):")
        print(top.head(5).to_string(index=False))
        print("\nTop 5 explanations (preview):")
        print(explain_df.sort_values("overall_score", ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    main()
