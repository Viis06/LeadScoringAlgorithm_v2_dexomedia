#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4 — Merge Datasets (create per-lead master feature table)

Inputs (defaults can be overridden with CLI args):
- leads_clean.csv
- companies_clean.csv
- activity_summary_leads.csv
- activity_summary_companies.csv
- keyword_features_leads.csv
- keyword_features_companies.csv
- hiring_features.csv

Output:
- features_master.csv  (one row per person_id enriched with lead, company,
  lead/company activity & keyword features, and company hiring features)
"""

import argparse
import os
import sys
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import numpy as np

# ----------------------------
# Candidate column name pools
# ----------------------------
PERSON_ID_COLS = ["person_id", "profileUrl", "profile_url", "authorUrl", "linkedin_profile_url"]
COMPANY_ID_COLS = ["company_id", "companyId", "company_id_clean"]

COMPANY_URL_COLS_LEADS = [
    "company_url", "companyProfileUrl", "company_profile_url", "companyUrl",
    "currentCompanyUrl", "employer_profile_url"
]
COMPANY_URL_COLS_COMPANIES = [
    "company_url", "salesNavigatorCompanyUrl", "companyProfileUrl", "linkedin_company_url", "companyUrl"
]

COMPANY_DOMAIN_COLS_LEADS = [
    "company_domain", "companyWebsiteDomain", "company_website", "website", "domain"
]
COMPANY_DOMAIN_COLS_COMPANIES = [
    "domain", "website_domain", "website", "companyDomain", "company_website"
]

# ----------------------------
# Helpers
# ----------------------------
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_url(u: str) -> str:
    if pd.isna(u):
        return ""
    s = str(u).strip()
    if not s:
        return ""
    # remove zero-width chars sometimes found in scraped strings
    s = s.replace("\u200b", "").replace("\u200c", "")
    try:
        if "://" not in s:
            s = "http://" + s
        p = urlparse(s)
        base = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
        base = base.replace("://www.", "://")
        return base.lower()
    except Exception:
        return s.rstrip("/").lower()

def extract_domain(u: str) -> str:
    if pd.isna(u):
        return ""
    s = str(u).strip().lower()
    if not s:
        return ""
    try:
        if "://" not in s:
            if "/" not in s and " " not in s:
                d = s
            else:
                p = urlparse("http://" + s)
                d = p.netloc or p.path
        else:
            p = urlparse(s)
            d = p.netloc or p.path
    except Exception:
        d = s
    d = d.replace("www.", "").strip("/")
    return d

def ensure_column(df: pd.DataFrame, name: str, fill=""):
    if name not in df.columns:
        df[name] = fill

def read_csv_required(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"❌ Missing required {label}: {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"✅ Loaded {label}: {path} ({len(df):,} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        print(f"❌ Failed to read {label}: {path} ({e})")
        sys.exit(1)

def read_csv_optional(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️  Optional file not found (skipping): {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"✅ Loaded {label}: {path} ({len(df):,} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        print(f"⚠️  Could not read {label}: {path} ({e}) — skipping")
        return pd.DataFrame()



def rename_key_if_present(df: pd.DataFrame, candidates: List[str], target: str) -> pd.DataFrame:
    """If df has any of candidate keys rename first one to target; else return unchanged."""
    if df.empty:
        return df
    key = pick_first_existing(df, candidates)
    if key and key != target:
        df = df.rename(columns={key: target})
    return df

def enforce_key(df: pd.DataFrame, candidates: List[str], target: str):
    """Ensure a required key exists in df, rename first found candidate to target or error."""
    if df.empty:
        return df
    key = pick_first_existing(df, candidates)
    if not key:
        print(f"❌ Could not find a '{target}'-like column in dataframe. Candidates: {candidates}")
        sys.exit(1)
    if key != target:
        df = df.rename(columns={key: target})
    return df

def drop_dupe_on(df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    dupe_ct = int(df.duplicated(subset=[key]).sum())
    if dupe_ct:
        print(f"⚠️  {label}: {dupe_ct:,} duplicate '{key}' rows found; keeping first occurrence.")
        df = df.drop_duplicates(subset=[key], keep="first")
    return df

def add_company_norm_keys(df: pd.DataFrame,
                          company_id_cols: List[str],
                          company_url_cols: List[str],
                          company_domain_cols: List[str],
                          prefix: str = "") -> pd.DataFrame:
    """
    Compute normalized company keys: prefix + _company_id_norm, prefix + _company_url_norm, prefix + _company_domain_norm.
    prefix=='' makes column names exactly '_company_*' (used on leads/companies/hiring).
    """
    # company id
    key = pick_first_existing(df, company_id_cols)
    if key:
        df[prefix + "_company_id_norm"] = df[key].astype(str).fillna("").str.strip().str.lower()
    else:
        ensure_column(df, prefix + "_company_id_norm", "")

    # company url
    curl = pick_first_existing(df, company_url_cols)
    if curl:
        df[prefix + "_company_url_norm"] = df[curl].map(normalize_url)
    else:
        ensure_column(df, prefix + "_company_url_norm", "")

    # company domain - prefer explicit domain column else derive from url
    cdom = pick_first_existing(df, company_domain_cols)
    if cdom:
        df[prefix + "_company_domain_norm"] = df[cdom].map(extract_domain)
    elif curl:
        df[prefix + "_company_domain_norm"] = df[curl].map(extract_domain)
    else:
        ensure_column(df, prefix + "_company_domain_norm", "")

    return df

def safe_left_merge(base: pd.DataFrame, addon: pd.DataFrame, on: str) -> pd.DataFrame:
    """Left merge where 'addon' uses column 'on' as key. Avoid adding duplicate key column twice."""
    if addon.empty:
        return base
    addon_cols = [c for c in addon.columns if c != on]
    return base.merge(addon[[on] + addon_cols], on=on, how="left")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", default="leads_clean.csv")
    ap.add_argument("--companies", default="companies_clean.csv")

    ap.add_argument("--activity_leads", default="activity_summary_leads.csv")
    ap.add_argument("--activity_companies", default="activity_summary_companies.csv")

    ap.add_argument("--keywords_leads", default="keyword_features_leads.csv")
    ap.add_argument("--keywords_companies", default="keyword_features_companies.csv")

    ap.add_argument("--hiring", default="hiring_features.csv")  # company-level hiring features
    ap.add_argument("--out", default="features_master.csv")
    args = ap.parse_args()

    # --- Read identity tables (required) ---
    print("\n=== Loading identity tables ===")
    leads = read_csv_required(args.leads, "leads")
    companies = read_csv_required(args.companies, "companies")

    # --- Read feature tables (optional) ---
    print("\n=== Loading feature tables (lead-level) ===")
    act_leads = read_csv_optional(args.activity_leads, "lead activity summary")
    kw_leads = read_csv_optional(args.keywords_leads, "lead keyword features")

    print("\n=== Loading feature tables (company-level) ===")
    act_comp = read_csv_optional(args.activity_companies, "company activity summary")
    kw_comp = read_csv_optional(args.keywords_companies, "company keyword features")
    hiring = read_csv_optional(args.hiring, "company hiring features")

    # ----------------------------
    # Standardize & validate keys
    # ----------------------------

    # Leads: ensure we have a person_id and dedupe
    leads = enforce_key(leads, PERSON_ID_COLS, "person_id")
    leads = drop_dupe_on(leads, "person_id", "leads")

    # Lead-level features: rename person_id if present, dedupe by person_id
    if not act_leads.empty:
        act_leads = rename_key_if_present(act_leads := act_leads.copy(), PERSON_ID_COLS, "person_id")
        act_leads = drop_dupe_on(act_leads, "person_id", "lead activity")
    if not kw_leads.empty:
        kw_leads = rename_key_if_present(kw_leads := kw_leads.copy(), PERSON_ID_COLS, "person_id")
        kw_leads = drop_dupe_on(kw_leads, "person_id", "lead keyword")

    # Companies: ensure company_id and dedupe
    companies = enforce_key(companies, COMPANY_ID_COLS, "company_id")
    companies = drop_dupe_on(companies, "company_id", "companies")

    # Company-level features: rename company_id if present, dedupe
    if not act_comp.empty:
        act_comp = rename_key_if_present(act_comp := act_comp.copy(), COMPANY_ID_COLS, "company_id")
        act_comp = drop_dupe_on(act_comp, "company_id", "company activity")
    if not kw_comp.empty:
        kw_comp = rename_key_if_present(kw_comp := kw_comp.copy(), COMPANY_ID_COLS, "company_id")
        kw_comp = drop_dupe_on(kw_comp, "company_id", "company keyword")
    if not hiring.empty:
        hiring = rename_key_if_present(hiring := hiring.copy(), COMPANY_ID_COLS, "company_id")
        hiring = drop_dupe_on(hiring, "company_id", "hiring features")

    # ----------------------------
    # Add normalized company keys to leads, companies, hiring
    # ----------------------------
    leads = add_company_norm_keys(
        leads,
        company_id_cols=COMPANY_ID_COLS + ["company_id"],
        company_url_cols=COMPANY_URL_COLS_LEADS,
        company_domain_cols=COMPANY_DOMAIN_COLS_LEADS,
        prefix=""  # results: _company_id_norm, _company_url_norm, _company_domain_norm
    )

    companies = add_company_norm_keys(
        companies,
        company_id_cols=["company_id"] + COMPANY_ID_COLS,
        company_url_cols=COMPANY_URL_COLS_COMPANIES,
        company_domain_cols=COMPANY_DOMAIN_COLS_COMPANIES,
        prefix=""
    )

    if not hiring.empty:
        hiring = add_company_norm_keys(
            hiring,
            company_id_cols=["company_id"] + COMPANY_ID_COLS,
            company_url_cols=COMPANY_URL_COLS_COMPANIES,
            company_domain_cols=COMPANY_DOMAIN_COLS_COMPANIES,
            prefix=""
        )
    else:
        # ensure hiring has the normalization columns so later code can refer to them
        hiring = pd.DataFrame(columns=["company_id", "_company_id_norm", "_company_url_norm", "_company_domain_norm"])

    # ----------------------------
    # Build company-augmented table: companies + company-level features + hiring (all company-level)
    # ----------------------------
    comp_aug = companies.copy()

    # Merge company activity (on company_id) if present
    if not act_comp.empty:
        comp_aug = safe_left_merge(comp_aug, act_comp, on="company_id")

    # Merge company-level keywords
    if not kw_comp.empty:
        comp_aug = safe_left_merge(comp_aug, kw_comp, on="company_id")

    # Merge hiring features into company augmentation
    if not hiring.empty:
        # Hiring may already include company_id + hiring columns; merge on company_id
        comp_aug = safe_left_merge(comp_aug, hiring, on="company_id")

    # Deduplicate comp_aug by company_id (after merges)
    comp_aug = drop_dupe_on(comp_aug, "company_id", "comp_aug (companies + features)")

    # Ensure normalized keys exist in comp_aug
    comp_aug = add_company_norm_keys(
        comp_aug,
        company_id_cols=["company_id"] + COMPANY_ID_COLS,
        company_url_cols=COMPANY_URL_COLS_COMPANIES,
        company_domain_cols=COMPANY_DOMAIN_COLS_COMPANIES,
        prefix=""
    )

    # ----------------------------
    # Start building master from leads
    # ----------------------------
    master = leads.copy()

    # --- Merge lead-level activity & keyword features ON person_id (if present) ---
    if not act_leads.empty:
        add_cols = [c for c in act_leads.columns if c != "person_id"]
        master = master.merge(act_leads[["person_id"] + add_cols], on="person_id", how="left")

    if not kw_leads.empty:
        add_cols = [c for c in kw_leads.columns if c != "person_id"]
        master = master.merge(kw_leads[["person_id"] + add_cols], on="person_id", how="left")

    # ----------------------------
    # Merge company-augmented data to master using 3-stage fallback:
    #  1) _company_id_norm -> co__company_id_norm
    #  2) _company_url_norm  -> co__company_url_norm
    #  3) _company_domain_norm-> co__company_domain_norm
    # ----------------------------
    # Prepare comp_join (ensure normalized keys exist)
    comp_join = comp_aug.copy()
    for k in ["_company_id_norm", "_company_url_norm", "_company_domain_norm"]:
        ensure_column(comp_join, k, "")

    # 1) id-based join
    merged = master.merge(
        comp_join.add_prefix("co_"),
        left_on="_company_id_norm",
        right_on="co__company_id_norm",
        how="left",
        suffixes=("", "_dupco")
    )

    # Determine which rows didn't match on id
    need_url_fallback = merged["co__company_id_norm"].isna()

    # 2) url-based join for those remaining
    if need_url_fallback.any():
        fb_url = master.loc[need_url_fallback].merge(
            comp_join.add_prefix("co_"),
            left_on="_company_url_norm",
            right_on="co__company_url_norm",
            how="left"
        )
        # Ensure all 'co_' columns exist on merged
        co_cols = [c for c in fb_url.columns if c.startswith("co_")]
        for col in co_cols:
            if col not in merged.columns:
                merged[col] = pd.NA
        # Robust fill: prefer comp_join mapping by _company_url_norm; fallback to index-aligned fb_url series
        for col in co_cols:
            orig_col = col[3:] if col.startswith("co_") else col
            if "_company_url_norm" in comp_join.columns and orig_col in comp_join.columns:
                # Build mapping: url_norm -> value
                mapping = dict(zip(comp_join["_company_url_norm"].fillna(""), comp_join[orig_col]))
                # fill only missing values using master._company_url_norm mapping
                merged[col] = merged[col].fillna(merged["_company_url_norm"].map(mapping))
            else:
                # Fallback: create index-aligned series from fb_url and fill by index (safe shape)
                fb_series = pd.Series(fb_url[col].values, index=fb_url.index)
                merged[col] = merged[col].fillna(fb_series)

    # Update mask for those still unmatched after url fallback
    need_domain_fallback = merged["co__company_id_norm"].isna() & merged["co__company_url_norm"].isna()

    # 3) domain-based join for remaining
    if need_domain_fallback.any():
        fb_dom = master.loc[need_domain_fallback].merge(
            comp_join.add_prefix("co_"),
            left_on="_company_domain_norm",
            right_on="co__company_domain_norm",
            how="left"
        )
        co_cols = [c for c in fb_dom.columns if c.startswith("co_")]
        for col in co_cols:
            if col not in merged.columns:
                merged[col] = pd.NA
        # Robust fill: prefer comp_join mapping by _company_domain_norm; fallback to index-aligned fb_dom series
        for col in co_cols:
            orig_col = col[3:] if col.startswith("co_") else col
            if "_company_domain_norm" in comp_join.columns and orig_col in comp_join.columns:
                mapping = dict(zip(comp_join["_company_domain_norm"].fillna(""), comp_join[orig_col]))
                merged[col] = merged[col].fillna(merged["_company_domain_norm"].map(mapping))
            else:
                fb_series = pd.Series(fb_dom[col].values, index=fb_dom.index)
                merged[col] = merged[col].fillna(fb_series)

    master = merged

    # ----------------------------
    # Coverage diagnostics
    # ----------------------------
    n_leads = len(leads)
    print("\n=== Coverage diagnostics ===")
    print(f"Leads in base: {n_leads:,}")

    # Lead feature columns presence (activity + keywords)
    lead_feat_cols = []
    if not act_leads.empty:
        lead_feat_cols += [c for c in act_leads.columns if c != "person_id"]
    if not kw_leads.empty:
        lead_feat_cols += [c for c in kw_leads.columns if c != "person_id"]

    have_lead_features = master[lead_feat_cols].notna().any(axis=1) if lead_feat_cols else pd.Series([False]*n_leads)
    print(f"Leads with ANY lead-level features (activity/keywords): {int(have_lead_features.sum()):,} / {n_leads:,}")

    # Company linking presence
    co_key_cols = ["co__company_id_norm", "co__company_url_norm", "co__company_domain_norm"]
    for k in co_key_cols:
        if k not in master.columns:
            master[k] = pd.NA
    linked_company = master[co_key_cols].notna().any(axis=1)
    print(f"Leads linked to a company row (via id/url/domain): {int(linked_company.sum()):,} / {n_leads:,}")

    # Company-side features presence (after co_ prefix)
    comp_feat_cols = []
    if not act_comp.empty:
        comp_feat_cols += [f"co_{c}" for c in act_comp.columns if c != "company_id"]
    if not kw_comp.empty:
        comp_feat_cols += [f"co_{c}" for c in kw_comp.columns if c != "company_id"]
    if not hiring.empty:
        comp_feat_cols += [f"co_{c}" for c in hiring.columns if c != "company_id"]

    comp_feat_cols = [c for c in comp_feat_cols if c in master.columns]
    have_company_features = master[comp_feat_cols].notna().any(axis=1) if comp_feat_cols else pd.Series([False]*n_leads)
    print(f"Leads with ANY company-level features: {int(have_company_features.sum()):,} / {n_leads:,}")

    # Hiring features coverage specifically (co_hiring_*)
    hiring_cols = [c for c in master.columns if c.startswith("co_hiring_") or c.startswith("co_total_hiring") or c.startswith("co_hiring_posts")]
    hiring_present = master[hiring_cols].notna().any(axis=1) if hiring_cols else pd.Series([False]*n_leads)
    print(f"Leads linked to companies WITH hiring features: {int(hiring_present.sum()):,} / {n_leads:,}")

    # Print simple hiring intensity stats if available
    # Common column names from your hiring step: 'total_hiring_score', 'hiring_intensity', 'hiring_posts'
    co_hiring_intensity_candidates = [c for c in master.columns if c.endswith("hiring_intensity") or c.endswith("_hiring_intensity") or c == "co_hiring_intensity" or c == "co_total_hiring_score"]
    # look for 'co_hiring_intensity' if created, else try other candidates
    hi_col = None
    for cand in ["co_hiring_intensity", "co_total_hiring_score", "co_total_hiring_score", "co_total_hiring_score"]:
        if cand in master.columns:
            hi_col = cand
            break
    # fallback search by substring
    if hi_col is None:
        for c in master.columns:
            if "hiring" in c and "intensity" in c:
                hi_col = c
                break

    if hi_col:
        ser = pd.to_numeric(master[hi_col], errors="coerce").dropna()
        if not ser.empty:
            print("\nHiring intensity summary (mapped company -> leads):")
            print(f"  mean: {ser.mean():.3f} | max: {ser.max():.3f} | median: {ser.median():.3f}")
            q = ser.quantile([0.5, 0.75, 0.9, 0.95]).to_dict()
            print(f"  quantiles (50/75/90/95): {q}")
    else:
        # if no explicit hiring intensity found, print top 3 hiring related numeric columns if any
        hiring_numeric = [c for c in master.columns if "hiring" in c and pd.api.types.is_numeric_dtype(master[c].dtype)]
        if hiring_numeric:
            print("\nFound hiring numeric columns:", hiring_numeric[:5])

    # ----------------------------
    # Final tidy-up: fill numeric NaNs with 0, keep text/datetimes as-is
    # ----------------------------
    for c in master.columns:
        try:
            dt = master[c].dtype
        except Exception:
            dt = object
        if pd.api.types.is_numeric_dtype(dt):
            master[c] = master[c].fillna(0)

    # Save master table
    out_path = args.out
    master.to_csv(out_path, index=False)
    print(f"\n✅ Saved merged per-lead feature table to: {out_path}")
    print(f"Total rows: {len(master):,} | Total columns: {len(master.columns)}")

    # Quick peeks
    # Top 5 by lead keyword intent
    lead_kw_cols = [c for c in ["keyword_intent_score_30d", "keyword_intent_score_90d", "keyword_intent_score_7d"] if c in master.columns]
    if lead_kw_cols:
        col = lead_kw_cols[0]
        sample_cols = ["person_id", col]
        for maybe in ["co_company_name", "co_name", "co_companyName"]:
            if maybe in master.columns:
                sample_cols.append(maybe)
                break
        print("\nTop 5 leads by lead-level keyword intent (sample):")
        with pd.option_context("display.max_columns", 120, "display.width", 200):
            print(master[sample_cols].sort_values(col, ascending=False).head(5).to_string(index=False))

    co_kw_cols = [c for c in master.columns if c.startswith("co_keyword_intent_score_") or c == "co_keyword_intent_score_30d"]
    if co_kw_cols:
        col = co_kw_cols[0]
        sample_cols = ["person_id", col]
        for maybe in ["co_company_name", "co_name", "co_companyName"]:
            if maybe in master.columns:
                sample_cols.append(maybe)
                break
        print("\nTop 5 leads by company-level keyword intent (mapped score):")
        with pd.option_context("display.max_columns", 120, "display.width", 200):
            print(master[sample_cols].sort_values(col, ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
