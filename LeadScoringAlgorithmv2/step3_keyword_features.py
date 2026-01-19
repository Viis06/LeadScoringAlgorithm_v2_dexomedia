#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Keyword Features (Buyer Intent Signals from Content)

Enhanced: supports both lead-level and company-level activity.
Outputs separate feature + snippet CSVs for leads and companies.

Run:
  python step3_keyword_features.py --input_dir split_activity
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# -----------------------------
# 1) CONFIG — Weights & Keywords
# -----------------------------

SOURCE_WEIGHTS = {
    "post": 1.00,
    "comment_own": 0.75,
    "repost": 0.50,
    "comment_others": 0.40,
    "like": 0.20,
    "comment_reaction": 0.15,
    "uncategorized": 0.30
}

# Updated category weights for Pangolin AI targeting digital marketing agencies
CATEGORY_WEIGHTS = {
    "lead_generation": 3.0,
    "outreach_sales": 2.5,
    "growth_scaling": 2.0,
    "automation_ai": 1.8,
    "business_triggers": 1.2
}

# Updated keywords optimized for measuring buying intent for Sales Automation (digital marketing agencies)
KEYWORDS = {
    "lead_generation": [
        r"lead generation",
        r"generate leads?",
        r"client acquisition",
        r"appointment setting",
        r"outbound lead(s)?",
        r"inbound lead(s)?",
        r"demand generation",
        r"prospect(ing)?",
        r"pipeline build(ing)?",
        r"lead quality",
        r"lead funnel(s)?",
        r"\bSQLs?\b",
        r"\bMQLs?\b",
        r"meetings booked",
        r"sales ready leads?",
        r"\bb2b leads?\b",
        r"new clients?",
        r"customer acquisition"
    ],
    "outreach_sales": [
        r"cold email(s|ing)?",
        r"email campaign(s)?",
        r"sales outreach",
        r"multi-?channel outreach",
        r"sales engagement",
        r"deliverability",
        r"reply rates?",
        r"open rates?",
        r"personalized email(s)?",
        r"drip campaign(s)?",
        r"sales sequence(s)?",
        r"sales cadence",
        r"ABM outreach",
        r"ICP targeting",
        r"sales messaging"
    ],
    "growth_scaling": [
        r"scal(ing|e) agency",
        r"grow(ing)? agency",
        r"new market(s)?",
        r"expansion",
        r"client growth",
        r"business growth",
        r"predictable pipeline",
        r"revenue growth",
        r"sales growth",
        r"doubl(ing|e) revenue",
        r"\bROI\b",
        r"customer lifetime value",
        r"scalable system(s)?"
    ],
    "automation_ai": [
        r"sales automation",
        r"marketing automation",
        r"process automation",
        r"AI outreach",
        r"AI prospecting",
        r"AI-?powered",
        r"chatgpt",
        r"gpt-?4",
        r"openai",
        r"automation tool(s)?",
        r"workflow automation",
        r"autonomous agent(s)?",
        r"smart workflows?",
        r"automated follow-?ups?"
    ],
    "business_triggers": [
        r"capacity issues?",
        r"scaling challenges?",
        r"operational bottlenecks?",
        r"cost reduction",
        r"efficienc(y|ies)",
        r"tech adoption",
        r"resource constraint(s)?",
        r"manual process(es)?",
        r"low productivity",
        r"competition",
        r"market shift(s)?",
        r"revenue plateau",
        r"sales target(s)? miss(ed)?"
    ]
}

# Candidate columns
PERSON_ID_COLS = [
    "person_id", "profileUrl", "profile_url", "authorUrl",
    "author_url", "authorProfileUrl", "linkedin_profile_url"
]
COMPANY_ID_COLS = [
    "company_id", "companyId", "company_id_clean", "companyUrl",
    "company_profile_url", "salesNavigatorCompanyUrl", "linkedin_company_url", "authorUrl", "author"
]
# Prefer postTimestamp first (actual action time) then action_ts/timestamp
TIMESTAMP_COLS = [
    "postTimestamp", "action_ts", "timestamp", "time", "date", "posted_at",
    "created_at", "datetime", "post_date"
]

TEXT_COLS = [
    "text", "content", "post_text", "post", "body", "comment",
    "comment_text", "summary", "headline", "title", "subtitle",
    "article", "articleTitle", "articleSubtitle", "description",
    "snippet", "share_text", "action_text", "commentContent",
    "message", "caption", "action", "postContent"
]

# Windows config
WINDOWS = {
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90)
}


# -----------------------------
# Helpers
# -----------------------------
def read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
        return pd.DataFrame()


def pick_first_existing(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def combine_text_fields(row: pd.Series, text_cols: list) -> str:
    parts = []
    for c in text_cols:
        val = row.get(c, None)
        if pd.notna(val):
            parts.append(str(val))
    return " ".join(parts).strip()


def compile_category_regex() -> dict:
    compiled = {}
    for cat, patterns in KEYWORDS.items():
        compiled[cat] = [re.compile(p, re.I) for p in patterns]
    return compiled


def count_matches_by_category(text: str, compiled_regex: dict) -> dict:
    counts = {cat: 0 for cat in KEYWORDS.keys()}
    if not text:
        return counts
    # normalize separators that sometimes break patterns
    t = text.replace("#", " ").replace("/", " ")
    for cat, regs in compiled_regex.items():
        total = 0
        for rgx in regs:
            try:
                total += len(rgx.findall(t))
            except re.error:
                # in case a pattern is malformed, skip it but warn once
                print(f"⚠️ Regex error for category '{cat}' pattern '{rgx.pattern}'")
        counts[cat] = total
    return counts


def detect_anchor_ts(all_ts: list) -> pd.Timestamp:
    max_ts = None
    for s in all_ts:
        if s is None or len(s) == 0:
            continue
        mx = pd.to_datetime(s, utc=True, errors="coerce").max()
        if pd.isna(mx):
            continue
        if (max_ts is None) or (mx > max_ts):
            max_ts = mx
    if max_ts is None:
        max_ts = pd.Timestamp(datetime.now(timezone.utc))
    return max_ts


# -----------------------------
# Core processing for an entity (lead / company)
# -----------------------------
def process_matches(all_df: pd.DataFrame, id_col: str, compiled_regex: dict, anchor_ts: pd.Timestamp):
    """
    all_df: dataframe with columns:
        - id_col (person_id or company_id)
        - _source (post/repost/like/...)
        - _ts (utc timestamp)
        - plus any text columns present
    Returns: (features_df, snippet_rows)
    """
    if all_df.empty:
        return pd.DataFrame(), []

    # Ensure id column exists and is string
    all_df[id_col] = all_df[id_col].astype(str)

    # Determine text columns present globally for this DF
    text_cols_present = [c for c in TEXT_COLS if c in all_df.columns]
    if not text_cols_present:
        print("⚠️ No text columns detected for entity; keyword counts will be zero for this entity.")
    # Prepare records
    records = []
    snippets = []

    # iterate rows (ok for moderate datasets)
    for _, row in all_df.iterrows():
        eid = row.get(id_col)
        src = row.get("_source", "uncategorized")
        ts = row.get("_ts", pd.NaT)

        text = combine_text_fields(row, text_cols_present)
        if not isinstance(text, str):
            text = ""

        counts = count_matches_by_category(text, compiled_regex)
        total_mentions = sum(counts.values())
        if total_mentions == 0:
            continue

        src_w = SOURCE_WEIGHTS.get(src, SOURCE_WEIGHTS["uncategorized"])
        weighted_score = 0.0
        for cat, cnt in counts.items():
            if cnt > 0:
                weighted_score += cnt * CATEGORY_WEIGHTS.get(cat, 1.0) * src_w

        authored = 1 if src in {"post", "comment_own"} else 0
        engagement = 0 if authored else 1

        rec = {
            id_col: eid,
            "_ts": ts,
            "_source": src,
            "authored_mentions": total_mentions if authored else 0,
            "engagement_mentions": total_mentions if engagement else 0,
            "weighted_score": weighted_score,
            "total_mentions": total_mentions
        }
        for cat in KEYWORDS.keys():
            rec[f"{cat}_mentions"] = counts.get(cat, 0)
        records.append(rec)

        # snippets (QA)
        if len(snippets) < 500:
            cats_present = [c for c, v in counts.items() if v > 0]
            snip = text[:240].replace("\n", " ").strip()
            snippets.append({
                id_col: eid,
                "ts": ts,
                "source": src,
                "categories": ", ".join(cats_present),
                "snippet": snip
            })

    if not records:
        # return empty features shell (list of unique ids)
        base = pd.DataFrame({id_col: all_df[id_col].unique()})
        return base, snippets

    match_df = pd.DataFrame.from_records(records)
    match_df["_ts"] = pd.to_datetime(match_df["_ts"], utc=True, errors="coerce")

    # Aggregate per window
    per_features = {}
    for win_label, win_delta in WINDOWS.items():
        cutoff = anchor_ts - win_delta
        mask = (match_df["_ts"].notna()) & (match_df["_ts"] >= cutoff)
        win_df = match_df[mask].copy()

        if win_df.empty:
            per_features[win_label] = pd.DataFrame(columns=[id_col])
            # debug
            print(f"[DEBUG] Window {win_label}: 0 rows included for {id_col}")
            continue

        grp = win_df.groupby(id_col, dropna=False)
        agg = grp.agg({
            "total_mentions": "sum",
            "authored_mentions": "sum",
            "engagement_mentions": "sum",
            "weighted_score": "sum",
            **{f"{cat}_mentions": "sum" for cat in KEYWORDS.keys()}
        }).reset_index()

        # compute unique categories with >=1 mention
        cat_cols = [f"{cat}_mentions" for cat in KEYWORDS.keys() if f"{cat}_mentions" in agg.columns]
        if cat_cols:
            agg[f"kw_unique_cats_{win_label}"] = (agg[cat_cols] > 0).sum(axis=1)
        else:
            agg[f"kw_unique_cats_{win_label}"] = 0

        # rename columns with window suffix
        rename_map = {
            "total_mentions": f"kw_total_{win_label}",
            "authored_mentions": f"kw_authored_{win_label}",
            "engagement_mentions": f"kw_engagement_{win_label}",
            "weighted_score": f"keyword_intent_score_{win_label}",
        }
        for cat in KEYWORDS.keys():
            rename_map[f"{cat}_mentions"] = f"kw_{cat}_{win_label}"
        agg = agg.rename(columns=rename_map)

        # generic flag example retained for backward compatibility — adapt if needed
        # keep an example outsourcing flag behavior safe-guard: if category exists, create flag
        example_flag_cat = "lead_generation"
        out_col = f"kw_{example_flag_cat}_{win_label}"
        flag_col = f"kw_{example_flag_cat}_flag_{win_label}"
        if out_col in agg.columns:
            agg[flag_col] = (agg[out_col] > 0).astype(int)
        else:
            agg[flag_col] = 0

        per_features[win_label] = agg
        print(f"[DEBUG] Window {win_label}: {len(win_df):,} rows included for {id_col}")

    # last keyword timestamp
    hits_with_ts = match_df[match_df["_ts"].notna()].copy()
    if not hits_with_ts.empty:
        last_kw = hits_with_ts.sort_values("_ts").groupby(id_col, as_index=False).agg(last_keyword_ts=("_ts", "max"))
        last_kw["days_since_last_keyword"] = (anchor_ts - last_kw["last_keyword_ts"]).dt.days
    else:
        last_kw = pd.DataFrame(columns=[id_col, "last_keyword_ts", "days_since_last_keyword"])

    # Build final feature table starting from union of ids seen anywhere (including those with no matches)
    all_ids = pd.DataFrame({id_col: all_df[id_col].unique()})
    features = all_ids.copy()
    for win_label in WINDOWS.keys():
        features = features.merge(per_features.get(win_label, pd.DataFrame(columns=[id_col])), on=id_col, how="left")
    features = features.merge(last_kw, on=id_col, how="left")

    # Safely fill numeric NaNs
    for c in features.columns:
        if c == id_col:
            continue
        dt = features[c].dtype
        if pd.api.types.is_numeric_dtype(dt):
            features[c] = features[c].fillna(0)
        # leave datetime columns as-is

    return features, snippets


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="split_activity", help="Folder containing split CSVs from Step 2a")
    ap.add_argument("--out_features_leads", default="keyword_features_leads.csv")
    ap.add_argument("--out_features_companies", default="keyword_features_companies.csv")
    ap.add_argument("--out_snippets_leads", default="keyword_snippets_leads.csv")
    ap.add_argument("--out_snippets_companies", default="keyword_snippets_companies.csv")
    args = ap.parse_args()

    input_dir = args.input_dir

    compiled_regex = compile_category_regex()

    # Try to detect leads and companies subfolders
    leads_dir = os.path.join(input_dir, "leads") if os.path.isdir(os.path.join(input_dir, "leads")) else input_dir
    companies_dir = os.path.join(input_dir, "companies") if os.path.isdir(os.path.join(input_dir, "companies")) else None

    # file mapping (same names expected)
    file_map = {
        "post.csv": "post",
        "repost.csv": "repost",
        "comment_own.csv": "comment_own",
        "comment_others.csv": "comment_others",
        "like.csv": "like",
        "comment_reaction.csv": "comment_reaction",
        "uncategorized.csv": "uncategorized"
    }

    # Load all lead files
    lead_frames = []
    lead_ts_series = []
    print("\n--- Loading lead activity ---")
    for fname, source in file_map.items():
        path = os.path.join(leads_dir, fname)
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue
        df["_source"] = source

        # person id pick
        person_col = pick_first_existing(df, PERSON_ID_COLS)
        if not person_col and "author" in df.columns:
            person_col = "author"
        if not person_col:
            print(f"⚠️ Skipping {path} — no person-like id column found.")
            continue

        # timestamp pick (prefers postTimestamp)
        ts_col = pick_first_existing(df, TIMESTAMP_COLS)
        df["_ts"] = coerce_utc(df[ts_col]) if ts_col else pd.NaT

        # keep only needed + text columns
        text_cols_present = [c for c in TEXT_COLS if c in df.columns]
        keep = [person_col, "_source", "_ts"] + text_cols_present
        # include some common extras for QA if present
        for extra in ["postUrl", "postContent", "commentUrl", "commentContent", "authorUrl", "author"]:
            if extra in df.columns and extra not in keep:
                keep.append(extra)
        df = df[keep].rename(columns={person_col: "person_id"})
        lead_frames.append(df)
        lead_ts_series.append(df["_ts"])
        print(f"Loaded leads file: {path} ({len(df):,} rows)")

    # Load all company files (if companies_dir present) else try to find company files in input_dir if labelled
    company_frames = []
    company_ts_series = []
    if companies_dir:
        print("\n--- Loading company activity ---")
        for fname, source in file_map.items():
            path = os.path.join(companies_dir, fname)
            df = read_csv_if_exists(path)
            if df is None or df.empty:
                continue
            df["_source"] = source

            company_col = pick_first_existing(df, COMPANY_ID_COLS)
            if not company_col and "author" in df.columns:
                company_col = "author"
            if not company_col:
                print(f"⚠️ Skipping {path} — no company-like id column found.")
                continue

            ts_col = pick_first_existing(df, TIMESTAMP_COLS)
            df["_ts"] = coerce_utc(df[ts_col]) if ts_col else pd.NaT

            text_cols_present = [c for c in TEXT_COLS if c in df.columns]
            keep = [company_col, "_source", "_ts"] + text_cols_present
            for extra in ["postUrl", "postContent", "commentUrl", "commentContent", "authorUrl", "author"]:
                if extra in df.columns and extra not in keep:
                    keep.append(extra)
            df = df[keep].rename(columns={company_col: "company_id"})
            company_frames.append(df)
            company_ts_series.append(df["_ts"])
            print(f"Loaded companies file: {path} ({len(df):,} rows)")
    else:
        print("\nNo 'companies' subfolder found — skipping company-level keyword extraction.")

    # Combine frames per entity
    all_leads_df = pd.concat(lead_frames, ignore_index=True) if lead_frames else pd.DataFrame()
    all_companies_df = pd.concat(company_frames, ignore_index=True) if company_frames else pd.DataFrame()

    # Anchor time uses both sources (makes windows consistent)
    anchor_ts = detect_anchor_ts(lead_ts_series + company_ts_series)
    print(f"\nAnchor time (max postTimestamp in data or now): {anchor_ts}\n")

    # Process leads
    lead_features, lead_snippets = process_matches(all_leads_df, "person_id", compiled_regex, anchor_ts)
    # Process companies
    company_features, company_snippets = process_matches(all_companies_df, "company_id", compiled_regex, anchor_ts)

    # Write outputs
    # leads
    if not lead_features.empty:
        lead_features.to_csv(args.out_features_leads, index=False)
        pd.DataFrame(lead_snippets).to_csv(args.out_snippets_leads, index=False)
        print(f"✅ Saved per-lead keyword features to: {args.out_features_leads}")
        print(f"🧪 Saved lead QA snippets to: {args.out_snippets_leads}")
        # sample peek
        if "keyword_intent_score_30d" in lead_features.columns:
            sortcol = "keyword_intent_score_30d"
        elif "keyword_intent_score_90d" in lead_features.columns:
            sortcol = "keyword_intent_score_90d"
        else:
            sortcol = None
        if sortcol:
            print("\nTop 5 leads by", sortcol)
            print(lead_features.sort_values(sortcol, ascending=False).head(5)[["person_id", sortcol]].to_string(index=False))
    else:
        print("⚠️ No lead keyword features to save.")

    # companies
    if not company_features.empty:
        company_features.to_csv(args.out_features_companies, index=False)
        pd.DataFrame(company_snippets).to_csv(args.out_snippets_companies, index=False)
        print(f"✅ Saved per-company keyword features to: {args.out_features_companies}")
        print(f"🧪 Saved company QA snippets to: {args.out_snippets_companies}")
        if "keyword_intent_score_30d" in company_features.columns:
            sortcol = "keyword_intent_score_30d"
        elif "keyword_intent_score_90d" in company_features.columns:
            sortcol = "keyword_intent_score_90d"
        else:
            sortcol = None
        if sortcol:
            print("\nTop 5 companies by", sortcol)
            print(company_features.sort_values(sortcol, ascending=False).head(5)[["company_id", sortcol]].to_string(index=False))
    else:
        print("ℹ️ No company keyword features to save (no companies folder or no matches).")


if __name__ == "__main__":
    main()
