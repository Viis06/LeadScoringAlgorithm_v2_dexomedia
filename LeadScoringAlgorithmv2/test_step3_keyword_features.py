#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Keyword Features (Buyer Intent Signals from Content)

Enhanced: supports both lead-level and company-level activity.
Outputs separate feature + snippet CSVs for leads and companies.

Now also includes:
- matched_keywords: all keywords identified per entity
- keyword_contexts: context snippets around each keyword
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

CATEGORY_WEIGHTS = {
    "paid_social": 3.1,
    "paid_search_display": 3.0,
    "performance_marketing_core": 2.8,
    "analytics_attribution": 2.6,
    "cro_ux_conversion": 3.0,
    "ecommerce_unit_econ": 2.2,
    "creative_ugc_video": 3.0,
    "creative_testing_iteration": 2.6,
    "marketplace_ads": 2.8,
    "marketplace_ops_listings": 2.4,
    "lead_generation_b2b": 2.0,
    "business_triggers": 1.6
}

KEYWORDS = {
    "paid_social": [
        r"\bmeta\s+ads\b", r"\bfacebook\s+ads\b", r"\binstagram\s+ads\b",
        r"\btiktok\s+ads\b", r"\bsnap(chat)?\s+ads\b", r"\bpinterest\s+ads\b",
        r"\blinkedin\s+ads\b", r"\byoutube\s+ads\b",
        r"\bpaid\s+social\b", r"\bsocial\s+ads?\b",
        r"\bad\s+account\b", r"\bad\s+set(s)?\b", r"\bcampaign(s)?\b",
        r"\bcreative\s+fatigue\b", r"\bad\s+frequency\b", r"\bthumbstop(per)?\b",
        r"\bscal(ing|e)\s+ads?\b"
    ],

    "paid_search_display": [
        r"\bpaid\s+search\b", r"\bsearch\s+ads?\b", r"\bgoogle\s+ads\b",
        r"\bshopping\s+ads?\b", r"\bperformance\s+max\b|\bpmax\b",
        r"\bdisplay\s+ads?\b", r"\bgoogle\s+display\b|\bgdn\b",
        r"\byoutube\s+campaign(s)?\b", r"\bvideo\s+ads?\b",
        r"\bretarget(ing)?\b|\bremarket(ing)?\b", r"\blookalike\b|\blal\b",
        r"\bkeyword(s)?\b.*\b(bid|bidding|match\s+type)\b",
        r"\bsem\b", r"\bppc\b"
    ],

    "performance_marketing_core": [
        r"\bperformance\s+marketing\b", r"\bpaid\s+growth\b",
        r"\bdigital\s+marketing\b.*\b(leads?|growth|scale)\b",
        r"\bmedia\s+buy(ing|er)\b", r"\bcampaign\s+architecture\b",
        r"\baccountable\s+growth\b", r"\bscale\b.*\b(ads?|spend|budget)\b",
        r"\blead\s+gen(eration)?\b", r"\bdemand\s+gen(eration)?\b",
        r"\bfunnel(s)?\b", r"\bacquisition\b", r"\bcustomer\s+acquisition\b"
    ],

    "analytics_attribution": [
        r"\btracking\b", r"\bconversion\s+tracking\b", r"\bevent\s+tracking\b",
        r"\battribution\b", r"\bmulti[-\s]?touch\b|\bMTA\b",
        r"\bincremental(ity)?\b|\bholdout\b|\blift\s+test\b",
        r"\bpixel(s)?\b", r"\bserver[-\s]?side\b|\bcapi\b|\bconversion\s+api\b",
        r"\bUTM(s)?\b", r"\bga4\b|\bgoogle\s+analytics\b",
        r"\bgtm\b|\bgoogle\s+tag\s+manager\b",
        r"\bsearch\s+console\b|\bgsc\b",
        r"\blook(er)?\s+studio\b|\bdata\s+studio\b",
        r"\bmarketing\s+dashboard\b|\breporting\b"
    ],

    "cro_ux_conversion": [
        r"\bcro\b|\bconversion\s+rate\s+optim(iz|is)ation\b",
        r"\bconversion\s+rate(s)?\b", r"\bcheckout\b.*\boptim(iz|is)ation\b",
        r"\bcheckout\s+flow\b", r"\bfunnel\s+(drop[-\s]?off|leak|leaky)\b",
        r"\blanding\s+page(s)?\b.*\boptim(iz|is)ation\b",
        r"\bproduct\s+page\b|\bpdp\b", r"\badd\s+to\s+cart\b",
        r"\ba\/b\s+test(ing)?\b|\bsplit\s+test(ing)?\b",
        r"\bheatmap(s)?\b|\bsession\s+recording(s)?\b",
        r"\bpage\s+speed\b|\bsite\s+speed\b|\bcore\s+web\s+vitals\b|\bcwv\b",
        r"\bux\b|\buser\s+experience\b"
    ],

    "ecommerce_unit_econ": [
        r"\baov\b|\baverage\s+order\s+value\b",
        r"\bbasket\s+size\b", r"\bcart\s+value\b",
        r"\bunit\s+economics\b", r"\bmargins?\b",
        r"\bcac\b|\bcpl\b|\bcpa\b", r"\broas\b|\broi\b",
        r"\bltv\b|\bcustomer\s+lifetime\s+value\b",
        r"\bconversion\s+rate\b.*\bprofit\b",
        r"\breturn\s+rate\b|\brefund(s)?\b"
    ],

    "creative_ugc_video": [
        r"\bugc\b|\buser\s+generated\s+content\b",
        r"\bperformance\s+creative(s)?\b",
        r"\bvideo[-\s]?first\b", r"\bshort[-\s]?form\b",
        r"\bhook(s)?\b", r"\bthumbstop(per)?\b",
        r"\bproblem[-\s]?solution\b", r"\bbefore[-\s]?after\b",
        r"\btestimonial(s)?\b", r"\bcreator(s)?\b|\binfluencer(s)?\b",
        r"\bcreative\s+strategy\b", r"\bscript(s|ing)?\b", r"\bstoryboard(s)?\b",
        r"\bad\s+creative(s)?\b"
    ],

    "creative_testing_iteration": [
        r"\bcreative\s+test(ing)?\b", r"\btest\s+and\s+learn\b",
        r"\ba\/b\s+test(ing)?\b", r"\bvariant(s)?\b",
        r"\biteration(s)?\b|\biterate\b",
        r"\bcreative\s+iteration\b", r"\bconcept(s)?\b",
        r"\bangle(s)?\b", r"\bmessage\s+testing\b",
        r"\bwinning\s+creative(s)?\b", r"\bcreative\s+insight(s)?\b"
    ],

    "marketplace_ads": [
        r"\bamazon\s+ads?\b|\bamazon\s+sponsored\b",
        r"\bflipkart\s+ads?\b", r"\bmyntra\b", r"\bajio\b",
        r"\btata\s+cliq\b", r"\bnykaa\b",
        r"\bmarketplace\s+ads?\b|\bmarketplace\s+advertis(ing|ement)\b",
        r"\bsponsored\s+products?\b|\bsponsored\s+brands?\b",
        r"\bmarketplace\s+ppc\b", r"\bsearch\s+rank(ing)?\b.*\bmarketplace\b"
    ],

    "marketplace_ops_listings": [
        r"\bmarketplace\s+growth\b|\bmarketplace\s+ops\b|\bmarketplace\s+operations\b",
        r"\blisting(s)?\b.*\boptim(iz|is)ation\b",
        r"\bproduct\s+listing(s)?\b", r"\bcatalog(ue)?\b",
        r"\bcontent\s+optim(iz|is)ation\b.*\b(listing|catalog)\b",
        r"\bproduct\s+title(s)?\b|\bbullet\s+point(s)?\b|\bproduct\s+description(s)?\b",
        r"\bsearch\s+rank(ing)?\b", r"\bcategory\s+ranking\b",
        r"\binventory\b|\bstock[-\s]?out\b", r"\bpricing\b|\bdiscount(s)?\b"
    ],

    "lead_generation_b2b": [
        r"\blead\s+gen(eration)?\b", r"\bclient\s+acquisition\b",
        r"\bpipeline\b", r"\bappointments?\b|\bmeetings?\b",
        r"\binbound\s+leads?\b|\boutbound\b",
        r"\bqualified\s+lead(s)?\b", r"\bhigh[-\s]?intent\b"
    ],

    "business_triggers": [
        r"\blaunch(ing)?\b", r"\bnew\s+product\b|\bproduct\s+launch\b",
        r"\bscal(ing|e)\b|\bexpansion\b|\bgrowth\b",
        r"\btraffic\s+drop\b|\broas\s+drop\b|\bconversion\s+drop\b",
        r"\bneed\s+more\s+leads\b|\bleads?\s+down\b",
        r"\bhiring\b.*\b(marketing|growth|performance|paid|creative|seo)\b",
        r"\brebrand(ing)?\b|\bnew\s+website\b|\bsite\s+redesign\b",
        r"\bmarketplace\b.*\bexpansion\b"
    ]
}

PERSON_ID_COLS = [
    "person_id", "profileUrl", "profile_url", "authorUrl",
    "author_url", "authorProfileUrl", "linkedin_profile_url"
]
COMPANY_ID_COLS = [
    "company_id", "companyId", "company_id_clean", "companyUrl",
    "company_profile_url", "salesNavigatorCompanyUrl", "linkedin_company_url", "authorUrl", "author"
]
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

# === NEW: extract keyword context ===
def extract_context(text: str, start: int, end: int) -> str:
    tokens = re.findall(r"\w+|\S", text)
    # find token index for match
    char_to_token = {}
    char_idx = 0
    for i, tok in enumerate(tokens):
        for _ in tok:
            char_to_token[char_idx] = i
            char_idx += 1
        # add space offset
        char_idx += 1
    start_tok = char_to_token.get(start, 0)
    end_tok = char_to_token.get(end-1, len(tokens)-1)

    if start_tok > 0 and end_tok < len(tokens)-1:
        # middle of sentence → 2 before, 2 after
        left = max(0, start_tok-2)
        right = min(len(tokens), end_tok+3)
    elif start_tok == 0:
        # first word → 4 after
        left = 0
        right = min(len(tokens), end_tok+5)
    else:
        # last word → 4 before
        left = max(0, start_tok-4)
        right = len(tokens)
    return " ".join(tokens[left:right])

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
# Core processing for an entity
# -----------------------------
def process_matches(all_df: pd.DataFrame, id_col: str, compiled_regex: dict, anchor_ts: pd.Timestamp):
    if all_df.empty:
        return pd.DataFrame(), []

    all_df[id_col] = all_df[id_col].astype(str)
    text_cols_present = [c for c in TEXT_COLS if c in all_df.columns]

    records = []
    snippets = []

    for _, row in all_df.iterrows():
        eid = row.get(id_col)
        src = row.get("_source", "uncategorized")
        ts = row.get("_ts", pd.NaT)
        text = combine_text_fields(row, text_cols_present)
        if not isinstance(text, str):
            text = ""

        counts = {cat: 0 for cat in KEYWORDS.keys()}
        matched_terms = []
        matched_contexts = []

        # === NEW: capture matches + contexts ===
        for cat, regs in compiled_regex.items():
            for rgx in regs:
                for m in rgx.finditer(text):
                    counts[cat] += 1
                    matched_terms.append(m.group(0))
                    ctx = extract_context(text, m.start(), m.end())
                    matched_contexts.append(ctx)

        total_mentions = sum(counts.values())
        if total_mentions == 0:
            continue

        src_w = SOURCE_WEIGHTS.get(src, SOURCE_WEIGHTS["uncategorized"])
        weighted_score = sum(
            cnt * CATEGORY_WEIGHTS.get(cat, 1.0) * src_w
            for cat, cnt in counts.items() if cnt > 0
        )
        authored = 1 if src in {"post", "comment_own"} else 0
        engagement = 0 if authored else 1

        rec = {
            id_col: eid,
            "_ts": ts,
            "_source": src,
            "authored_mentions": total_mentions if authored else 0,
            "engagement_mentions": total_mentions if engagement else 0,
            "weighted_score": weighted_score,
            "total_mentions": total_mentions,
            "matched_keywords": ", ".join(matched_terms) if matched_terms else "",
            "keyword_contexts": ", ".join(matched_contexts) if matched_contexts else ""
        }
        for cat in KEYWORDS.keys():
            rec[f"{cat}_mentions"] = counts.get(cat, 0)
        records.append(rec)

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
        base = pd.DataFrame({id_col: all_df[id_col].unique()})
        return base, snippets

    match_df = pd.DataFrame.from_records(records)
    match_df["_ts"] = pd.to_datetime(match_df["_ts"], utc=True, errors="coerce")

    # === NEW: aggregate matched keywords + contexts ===
    agg_extra = match_df.groupby(id_col).agg({
        "matched_keywords": lambda x: ", ".join(filter(None, x.unique())),
        "keyword_contexts": lambda x: ", ".join(filter(None, x.unique()))
    }).reset_index()

    # --- existing aggregation windows (unchanged) ---
    per_features = {}
    for win_label, win_delta in WINDOWS.items():
        cutoff = anchor_ts - win_delta
        mask = (match_df["_ts"].notna()) & (match_df["_ts"] >= cutoff)
        win_df = match_df[mask].copy()
        if win_df.empty:
            per_features[win_label] = pd.DataFrame(columns=[id_col])
            continue
        grp = win_df.groupby(id_col, dropna=False)
        agg = grp.agg({
            "total_mentions": "sum",
            "authored_mentions": "sum",
            "engagement_mentions": "sum",
            "weighted_score": "sum",
            **{f"{cat}_mentions": "sum" for cat in KEYWORDS.keys()}
        }).reset_index()
        cat_cols = [f"{cat}_mentions" for cat in KEYWORDS.keys() if f"{cat}_mentions" in agg.columns]
        agg[f"kw_unique_cats_{win_label}"] = (agg[cat_cols] > 0).sum(axis=1) if cat_cols else 0
        rename_map = {
            "total_mentions": f"kw_total_{win_label}",
            "authored_mentions": f"kw_authored_{win_label}",
            "engagement_mentions": f"kw_engagement_{win_label}",
            "weighted_score": f"keyword_intent_score_{win_label}",
        }
        for cat in KEYWORDS.keys():
            rename_map[f"{cat}_mentions"] = f"kw_{cat}_{win_label}"
        agg = agg.rename(columns=rename_map)
        example_flag_cat = "lead_generation"
        out_col = f"kw_{example_flag_cat}_{win_label}"
        flag_col = f"kw_{example_flag_cat}_flag_{win_label}"
        agg[flag_col] = (agg[out_col] > 0).astype(int) if out_col in agg.columns else 0
        per_features[win_label] = agg

    hits_with_ts = match_df[match_df["_ts"].notna()].copy()
    if not hits_with_ts.empty:
        last_kw = hits_with_ts.sort_values("_ts").groupby(id_col, as_index=False).agg(last_keyword_ts=("_ts", "max"))
        last_kw["days_since_last_keyword"] = (anchor_ts - last_kw["last_keyword_ts"]).dt.days
    else:
        last_kw = pd.DataFrame(columns=[id_col, "last_keyword_ts", "days_since_last_keyword"])

    all_ids = pd.DataFrame({id_col: all_df[id_col].unique()})
    features = all_ids.copy()
    for win_label in WINDOWS.keys():
        features = features.merge(per_features.get(win_label, pd.DataFrame(columns=[id_col])), on=id_col, how="left")
    features = features.merge(last_kw, on=id_col, how="left")
    features = features.merge(agg_extra, on=id_col, how="left")  # === NEW merge ===

    for c in features.columns:
        if c == id_col:
            continue
        dt = features[c].dtype
        if pd.api.types.is_numeric_dtype(dt):
            features[c] = features[c].fillna(0)

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
