#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_final_dataset.py

Create a unified final dataset by merging:
  - scores_master.csv  (detailed features + canonical scores)
  - scores_explain.csv (explainability: top_drivers, reasons, contrib_* etc.)

Behavior:
 - Joins on `person_id` (left join: keep master rows)
 - If scores_explain contains multiple rows per person_id, collapse them:
     * numeric columns are summed
     * text columns are deduplicated and joined with ' || '
 - Any column in explain that would overwrite a master column is renamed with an `ex_` prefix
 - Final CSV places core ID and score columns up front and explainability columns at the end

Usage:
    python step7_final_dataset.py \
      --master scores_master.csv \
      --explain scores_explain.csv \
      --output scores_final.csv

If you run without args, defaults will be used (files in current working dir).
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List

import numpy as np
import pandas as pd

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -------------------------
# Utilities
# -------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def collapse_explain_df(df: pd.DataFrame, key: str = "person_id") -> pd.DataFrame:
    """
    If explain dataframe has >1 row per person_id, collapse to one row:
      - numeric columns: sum
      - non-numeric columns: dedupe and join with ' || '
    Returns collapsed dataframe (person_id unique).
    """
    if df.empty:
        return df

    dup_counts = df[key].duplicated().sum()
    if dup_counts == 0:
        return df

    logging.info(f"Found {dup_counts:,} duplicated rows in explain data — collapsing to one row per {key}.")

    def agg_func(series):
        if pd.api.types.is_numeric_dtype(series.dtype):
            return series.sum(min_count=1)
        else:
            # join unique non-null values
            vals = [str(x).strip() for x in series.dropna().astype(str)]
            uniq = list(dict.fromkeys([v for v in vals if v != "" and v.lower() not in ("nan", "none")]))
            return " || ".join(uniq) if uniq else np.nan

    grouped = df.groupby(key, as_index=False).agg(agg_func)
    return grouped


def rename_explain_overlaps(master_cols: List[str], explain_cols: List[str], key: str = "person_id") -> dict:
    """
    Build rename map for explain cols that overlap with master cols (excluding key).
    All overlapping explain column names will be renamed with prefix 'ex_'.
    Returns dict suitable for DataFrame.rename(columns=...)
    """
    rename_map = {}
    master_set = set(master_cols)
    for col in explain_cols:
        if col == key:
            continue
        if col in master_set:
            rename_map[col] = f"ex_{col}"
    return rename_map


def order_columns(master_cols: List[str], explain_cols_after_rename: List[str]) -> List[str]:
    """
    Build final column order:
      1) person_id (if exists)
      2) company_id (if exists)
      3) profileUrl, fullName, firstName, lastName, companyName, title, companyUrl (if present)
      4) canonical scores if present: lead_score, company_score, overall_score
      5) remaining master columns (in their original order)
      6) explainability columns (explain_cols_after_rename) at the end
    """
    priority_ids = ["person_id", "company_id"]
    header_personal = ["profileUrl", "fullName", "firstName", "lastName", "companyName", "title", "companyUrl"]
    score_priority = ["lead_score", "company_score", "overall_score"]

    final = []
    # 1) IDs
    for c in priority_ids:
        if c in master_cols and c not in final:
            final.append(c)
    # 2) personal header
    for c in header_personal:
        if c in master_cols and c not in final:
            final.append(c)
    # 3) scores
    for c in score_priority:
        if c in master_cols and c not in final:
            final.append(c)
    # 4) remaining master columns (preserve order but avoid duplicates)
    for c in master_cols:
        if c not in final:
            final.append(c)
    # 5) append all explain columns (they should not yet be in final)
    for c in explain_cols_after_rename:
        if c not in final:
            final.append(c)
    return final


# -------------------------
# Main merge function
# -------------------------
def merge_scores(master_path: str, explain_path: str, output_path: str, preview_top_n: int = 5) -> pd.DataFrame:
    logging.info("Loading master CSV...")
    master_df = safe_read_csv(master_path)
    logging.info("Loading explain CSV...")
    explain_df = safe_read_csv(explain_path)

    logging.info(f"Master shape: {master_df.shape}")
    logging.info(f"Explain shape: {explain_df.shape}")

    # Validate keys
    if "person_id" not in master_df.columns:
        raise KeyError("`person_id` column not found in scores_master file.")
    if "person_id" not in explain_df.columns:
        raise KeyError("`person_id` column not found in scores_explain file.")

    # Collapse explain duplicates if any
    explain_df = collapse_explain_df(explain_df, key="person_id")

    # Rename overlapping explain columns so we don't overwrite master columns
    rename_map = rename_explain_overlaps(master_df.columns.tolist(), explain_df.columns.tolist(), key="person_id")
    if rename_map:
        logging.info(f"Renaming overlapping explain columns to avoid overwrite: {rename_map}")
        explain_df = explain_df.rename(columns=rename_map)

    # Keep list of explain columns after rename (excluding person_id)
    explain_cols_after_rename = [c for c in explain_df.columns if c != "person_id"]

    # Merge (left join: keep master rows)
    logging.info("Merging explain data into master (left join on person_id)...")
    merged = master_df.merge(explain_df, on="person_id", how="left", sort=False)

    logging.info(f"Merged shape: {merged.shape}")

    # Reorder columns for readability
    master_cols = master_df.columns.tolist()
    final_order = order_columns(master_cols, explain_cols_after_rename)

    # Filter final_order to only existing columns in merged
    final_order = [c for c in final_order if c in merged.columns]
    # add any columns that weren't in final_order at the end (to avoid dropping anything)
    tail_cols = [c for c in merged.columns if c not in final_order]
    final_cols = final_order + tail_cols

    merged = merged[final_cols]

    # Ensure output directory exists only if provided (handles bare filenames)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    merged.to_csv(output_path, index=False)
    logging.info(f"Saved merged dataset to: {output_path}")

    # Preview top N if overall_score present
    if "overall_score" in merged.columns:
        try:
            n = min(preview_top_n, len(merged))
            top = merged.sort_values("overall_score", ascending=False).head(n)
            logging.info(f"Top {n} rows by overall_score (sample):")
            with pd.option_context("display.max_columns", 120, "display.width", 220):
                logging.info("\n" + top[["person_id", "company_id", "fullName", "overall_score"]].to_string(index=False))
        except Exception:
            logging.debug("Could not display top sample (missing columns or other issue).")
    else:
        logging.info("overall_score not present in merged dataset; skipping top-sample display.")

    return merged


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge scores_master and scores_explain into a unified final dataset.")
    parser.add_argument("--master", default="scores_master.csv", help="Path to scores_master.csv (default: scores_master.csv)")
    parser.add_argument("--explain", default="scores_explain.csv", help="Path to scores_explain.csv (default: scores_explain.csv)")
    parser.add_argument("--output", default="scores_final.csv", help="Path to save final merged CSV (default: scores_final.csv)")
    parser.add_argument("--preview_top_n", type=int, default=5, help="Show top N leads by overall_score after merge (if present)")
    args = parser.parse_args()

    try:
        merged = merge_scores(args.master, args.explain, args.output, preview_top_n=args.preview_top_n)
    except Exception as e:
        logging.exception("Failed to create final dataset:")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
