#!/usr/bin/env python3
"""
step6_generate_report.py

Reads the outputs from step5_scoring.py (scores_master.csv and scores_explain.csv),
merges them into a clean, human-readable file containing only the essential columns.

Keeps:
- name
- company_name
- linkedin_url
- company_id
- lead_score
- company_score
- overall_score
- reasons (simplified for sales)

Outputs:
- scores_report.csv
"""

import pandas as pd
import re


def clean_reason(text: str) -> str:
    """Simplify raw reason text into a sales-friendly explanation."""
    if pd.isna(text):
        return "No strong signals detected"

    # Normalize spaces
    text = re.sub(r"\s+", " ", str(text)).strip()

    # Replace technical terms with friendlier versions
    replacements = {
        "hiring_intensity": "hiring activity",
        "hiring_velocity": "recent hiring momentum",
        "relevant_hiring_count": "relevant roles hiring",
        "adjacent_hiring_count": "adjacent roles hiring",
        "irrelevant_hiring_count": "other hiring",
        "lead_score": "lead engagement",
        "company_score": "company potential",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove overly technical score mentions (like weights, decimals, etc.)
    text = re.sub(r"\bscore[s]?:?\s*\d+(\.\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+\.\d{2,}", "", text)  # remove floating-point numbers

    # Simplify connectors
    text = text.replace("because", "-").replace("due to", "-")

    # Truncate if too long
    if len(text) > 180:
        text = text[:180].rsplit(" ", 1)[0] + "…"

    return text.strip(" ,.-")


def generate_report(scores_master_path: str, scores_explain_path: str, output_path: str):
    print("🔄 Reading input files...")
    df_scores = pd.read_csv(scores_master_path)
    df_explain = pd.read_csv(scores_explain_path)

    print(f"✅ scores_master.csv loaded: {df_scores.shape[0]} rows")
    print(f"✅ scores_explain.csv loaded: {df_explain.shape[0]} rows")

    if "person_id" not in df_scores.columns or "person_id" not in df_explain.columns:
        raise KeyError("Both input files must contain 'person_id' for joining.")

    # Merge explanations into scores
    df = pd.merge(df_scores, df_explain[["person_id", "reasons"]], on="person_id", how="inner")

    print(f"🔗 Merged dataset: {df.shape[0]} rows")

    # Select and rename columns
    columns_to_keep = [
        "fullName",
        "companyName",
        "person_id",
        "company_id",
        "lead_score",
        "company_score",
        "overall_score",
        "reasons",
    ]
    missing_cols = [c for c in columns_to_keep if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")

    df_clean = df[columns_to_keep].copy()

    # Clean up reasons for readability
    df_clean["reasons"] = df_clean["reasons"].apply(clean_reason)

    # Sort by overall score descending
    df_clean = df_clean.sort_values(by="overall_score", ascending=False).reset_index(drop=True)

    # Save output
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Report generated: {output_path}")
    print(df_clean.head(10))  # preview top 10


if __name__ == "__main__":
    generate_report("scores_master.csv", "scores_explain.csv", "scores_report.csv")
