#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step3b_hiring_features.py

High-precision hiring intent extraction (company-level + lead-level).

Inputs:
  - company_activity_clean.csv  (required)
  - activity_clean.csv     (optional)  <-- NEW

Outputs:
  - hiring_features.csv      (per-company engineered features across windows)
  - hiring_snippets.csv      (QA: all detected hiring posts with snippets)

This script preserves existing behavior for company-level inputs, while also
allowing lead-level activity to be processed and merged into company-level
hiring signals. We maintain separate aggregates for company-origin hits and
lead-origin hits and produce a derived combined hiring score (weighted).

Run:
  python step3b_hiring_features.py \
    --input company_activity_clean.csv \
    --lead_input activity_clean.csv \
    --out_features hiring_features.csv \
    --out_snippets hiring_snippets.csv \
    --half_life_days 45 \
    --windows 7,30,90 \
    --company_hiring_weight 0.6 \
    --lead_hiring_weight 0.4
"""

import argparse
import os
import re
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# ================================
# 0) Config: Columns & Weights
# ================================

COMPANY_ID_COLS = ["company_id", "companyId", "employerCompanyId", "company_id_clean"]
TIMESTAMP_COLS = [
    "action_ts", "postTimestamp", "time", "date", "posted_at", "created_at",
    "datetime", "post_date", "date_posted", "ts"
]
TEXT_COLS = [
    "postContent", "text", "content", "body", "message", "caption",
    "comment", "comment_text", "description", "snippet", "share_text", "action_text"
]
SOURCE_COLS = ["source", "_source", "action", "type", "activity_type"]

SOURCE_NORMALIZE = {
    "post": "post",
    "article": "post",
    "original": "post",
    "share": "repost",
    "repost": "repost",
    "reshare": "repost",
    "comment": "comment",
    "reply": "comment",
    "job": "job",
    "unknown": "unknown"
}

SOURCE_WEIGHTS = {
    "post": 1.00,
    "job": 1.00,
    "repost": 0.85,
    "comment": 0.50,
    "unknown": 0.70
}

JOB_LINK_DOMAINS = [
    r"linkedin\.com/jobs", r"indeed\.", r"greenhouse\.io", r"lever\.co", r"workable\.com",
    r"workday\.", r"bamboohr\.", r"icims\.", r"smartrecruiters\.", r"jazzhr\.",
    r"boards\.greenhouse\.io", r"jobs\.", r"recruitee\.", r"ashbyhq\.", r"jobvite\."
]

HIRING_SIGNALS_STRONG = [
    r"\bapply\s+now\b",
    r"\bapplications?\s+open\b",
    r"\bvacanc(y|ies)\b",
    r"\bjob\s+(opening|post(ing)?)\b",
    r"\bnow\s+hiring\b",
    r"\bwe(?:'|’)?re\s+hiring\b",
    r"\bwe\s+are\s+hiring\b",
    r"\bjoin\s+our\s+team\b",
    r"\bcareers?\s+(page|site)\b",
    r"\brecruit(ing|ment)\b"
]
HIRING_SIGNALS_MODERATE = [
    r"\bhiring\s+for\b",
    r"\blooking\s+for\b",
    r"\bwe\s+are\s+expanding\b",
    r"\bteam\s+is\s+growing\b"
]
HIRING_HASHTAGS = [
    r"#hiring\b", r"#nowhiring\b", r"#werehiring\b", r"#we?re?hiring\b",
    r"#career(s)?\b", r"#job(s)?\b", r"#vacancy\b"
]

NEGATIONS = [
    r"\bnot\s+hiring\b", r"\bno\s+longer\s+hiring\b", r"\bhiring\s+freeze\b",
    r"\b(pos(ition)?|role)\s+(has been|is)\s+filled\b",
    r"\bapplications?\s+(closed|are closed)\b", r"\bwe\s+stopped\s+hiring\b"
]

SENIORITY_WEIGHTS = {
    r"\bintern(ship)?\b": 0.8,
    r"\bjunior\b": 0.9,
    r"\bgraduate\b": 0.9,
    r"\bassistant\b": 0.95,
    r"\bassociate\b": 1.0,
    r"\bsemi\s*senior\b": 1.05,
    r"\bsenior\b": 1.1,
    r"\blead\b": 1.15,
    r"\bmanager\b": 1.2,
    r"\bhead\b": 1.25,
    r"\bdirector\b": 1.3,
    r"\bvp\b": 1.3,
    r"\bchief\b|\bcfo\b|\bcoo\b": 1.35,
}

# Role relevance aligned to Pangolin AI (sales automation for digital marketing agencies)
ROLE_PATTERNS = {
    # ===== Highly Relevant: 1.0 =====
    "founder_ceo_owner": (1.0, [
        r"\bfounder(s)?\b", r"\bco[-\s]?founder(s)?\b", r"\bceo\b", r"\bchief\s+executive\b",
        r"\bmanaging\s+director\b", r"\bmd\b", r"\bowner\b", r"\bpartner\b",
        r"\bprincipal\b", r"\bproprietor\b"
    ]),
    "chief_revenue_officer": (1.0, [
        r"\bCRO\b", r"\bchief\s+revenue\s+officer\b", r"\bchief\s+commercial\s+officer\b",
        r"\bCCO\b", r"\bvp\s+revenue\b", r"\bvice\s+president\s+of\s+revenue\b"
    ]),
    "sdr_bdr": (1.0, [
        r"\bSDR(s)?\b", r"\bsales\s+development\s+rep(s)?\b",
        r"\bBDR(s)?\b", r"\bbusiness\s+development\s+rep(s)?\b",
        r"\bappointment\s+setter(s)?\b"
    ]),
    "account_executive": (1.0, [
        r"\baccount\s+executive(s)?\b", r"\bAE(s)?\b", r"\bsenior\s+account\s+executive\b",
        r"\bclient\s+executive(s)?\b", r"\bnew\s+business\s+executive\b"
    ]),
    "sales_manager_director": (1.0, [
        r"\bsales\s+manager(s)?\b", r"\bhead\s+of\s+sales\b", r"\bdirector\s+of\s+sales\b",
        r"\bvp\s+sales\b", r"\bvice\s+president\s+of\sales\b", r"\bsales\s+lead\b"
    ]),
    "business_development_lead": (1.0, [
        r"\bbusiness\s+development\b", r"\bBDM(s)?\b", r"\bbusiness\s+development\s+manager\b",
        r"\bbusiness\s+development\s+director\b", r"\bnew\s+business\s+director\b",
        r"\bnew\s+business\s+manager\b"
    ]),

    # ===== High-Adjacent: 0.9–0.95 =====
    "growth_manager": (0.95, [
        r"\bgrowth\s+manager(s)?\b", r"\bgrowth\s+lead(s)?\b", r"\bgrowth\s+hacker(s)?\b",
        r"\bhead\s+of\s+growth\b", r"\bgrowth\s+marketing\b", r"\bperformance\s+growth\b"
    ]),
    "demand_generation": (0.9, [
        r"\bdemand\s+generation\b", r"\bdemand\s+gen\b", r"\bdemand\s+gen(eration)?\b",
        r"\bdemand\s+gen\s+manager\b", r"\bdemand\s+gen\s+specialist\b",
        r"\bperformance\s+marketing\b", r"\bclient\s+acquisition\s+manager\b"
    ]),
    "revenue_operations": (0.9, [
        r"\bRevOps\b", r"\brevenue\s+operations?\b", r"\bsales\s+operations?\b",
        r"\bsales\s+ops\b", r"\brevenue\s+ops\s+manager\b", r"\bcommercial\s+operations\b"
    ]),
    "client_services_account_director": (0.85, [
        r"\baccount\s+director(s)?\b", r"\bclient\s+partner(s)?\b", r"\bclient\s+services?\b",
        r"\bhead\s+of\s+client\s+services\b", r"\bclient\s+director\b", r"\baccount\s+manager\b"
    ]),
    "partnerships_channel": (0.85, [
        r"\bpartnerships?\b", r"\bchannel\s+partner(s)?\b", r"\balliances?\b",
        r"\bpartner\s+manager\b", r"\bchannel\s+manager\b"
    ]),
    "sales_enablement": (0.8, [
        r"\bsales\s+enablement\b", r"\benablement\s+manager\b", r"\bsales\s+training\b",
        r"\bonboarding\s+manager\b", r"\benablement\s+specialist\b"
    ]),
    "marketing_leadership": (0.8, [
        r"\bCMO\b", r"\bchief\s+marketing\s+officer\b", r"\bvp\s+marketing\b",
        r"\bhead\s+of\s+marketing\b", r"\bmarketing\s+director\b", r"\bmarketing\s+lead\b"
    ]),
    "customer_success": (0.8, [
        r"\bcustomer\s+success\b", r"\bcustomer\s+experience\b", r"\bclient\s+success\b",
        r"\bsuccess\s+manager\b", r"\bhead\s+of\s+customer\s+success\b"
    ]),
    "commercial_ops": (0.8, [
        r"\bcommercial\s+director\b", r"\bhead\s+of\s+commercial\b", r"\brevenue\s+director\b"
    ]),
    "data_analytics_growth": (0.75, [
        r"\bgrowth\s+analyst\b", r"\bmarketing\s+analyst\b", r"\bperformance\s+analyst\b",
        r"\bdata\s+analyst\b", r"\binsights?\s+lead\b", r"\banalytics\s+manager\b"
    ]),

    # ===== Adjacent-Lower (still useful): 0.7–0.75 =====
    "new_business_manager": (0.75, [
        r"\bnew\s+business\b", r"\bnew\s+business\s+manager\b", r"\bnew\s+business\s+lead\b"
    ]),
    "commercial_and_strategy": (0.7, [
        r"\bbusiness\s+strategy\b", r"\bstrategy\s+lead\b", r"\bbusiness\s+operations\b",
        r"\bcommercial\s+strategy\b"
    ]),

    # ===== Peripheral / Less Relevant: ≤ 0.5 =====
    "marketing_creative": (0.5, [
        r"\bcreative\s+director\b", r"\bart\s+director\b", r"\bcreative\s+lead\b",
        r"\bcopywriter(s)?\b", r"\bcontent\s+manager\b", r"\bcontent\s+writer\b",
        r"\bsocial\s+media\s+manager\b", r"\bcommunity\s+manager\b"
    ]),
    "design_development": (0.3, [
        r"\bweb\s+developer\b", r"\bfront[-\s]?end\b", r"\bui\/ux\b", r"\bproduct\s+designer\b",
        r"\bfull[-\s]?stack\b", r"\bsoftware\s+engineer\b"
    ]),
    "hr_recruiting": (0.2, [
        r"\bhr\b", r"\bhuman\s+resources?\b", r"\brecruit(ing|er)\b", r"\btalent\s+acquisition\b",
        r"\bpeople\s+ops\b"
    ]),
    "finance_accounts_admin": (0.2, [
        r"\bcfo\b", r"\bchief\s+financial\s+officer\b", r"\bfinance\s+manager\b",
        r"\baccounts?\b", r"\bbookkeeper\b", r"\bpayroll\b"
    ]),
    "operations_admin": (0.2, [
        r"\boperations?\b", r"\bops\b", r"\boffice\s+manager\b", r"\badministrat(ion|or)\b"
    ]),
    "legal_compliance": (0.15, [
        r"\blegal\b", r"\bcompliance\b", r"\blegal\s+counsel\b", r"\bcompany\s+secretary\b"
    ])
}

# Detection thresholds & multipliers
BASE_SCORE_STRONG = 12.0
BASE_SCORE_MODERATE = 8.0
BASE_SCORE_HASHTAG_ONLY = 5.0
JOB_LINK_BONUS = 1.15           # multiplier if job-board/ATS link appears
MULTI_ROLE_STEP = 0.10          # +10% per extra role (capped)
MULTI_ROLE_CAP = 1.40           # cap multi-role multiplier to 1.4x

# ================================
# 1) Utils
# ================================

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s

def detect_anchor_ts(ts_series_list: list[pd.Series]) -> pd.Timestamp:
    max_ts = None
    for s in ts_series_list:
        if s is None or s.empty:
            continue
        mx = pd.to_datetime(s, utc=True, errors="coerce").max()
        if pd.isna(mx):
            continue
        if (max_ts is None) or (mx > max_ts):
            max_ts = mx
    if max_ts is None:
        max_ts = pd.Timestamp(datetime.now(timezone.utc))
    return max_ts

def combine_text_fields(row: pd.Series, text_cols_present: list[str]) -> str:
    parts = []
    for c in text_cols_present:
        v = row.get(c, None)
        if pd.notna(v) and str(v).strip() != "":
            parts.append(str(v))
    return " ".join(parts).strip()

def _normalize_apostrophes_and_spaces(s: str) -> str:
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("\u200b", "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def compile_patterns():
    def comp_list(patterns):
        return [re.compile(p, flags=re.I) for p in patterns]
    compiled = {
        "negations": comp_list(NEGATIONS),
        "strong": comp_list(HIRING_SIGNALS_STRONG),
        "moderate": comp_list(HIRING_SIGNALS_MODERATE),
        "hashtags": comp_list(HIRING_HASHTAGS),
        "joblinks": comp_list(JOB_LINK_DOMAINS),
        "seniority": [(re.compile(p, re.I), w) for p, w in SENIORITY_WEIGHTS.items()],
        "roles": {k: (w, [re.compile(p, re.I) for p in pats]) for k, (w, pats) in ROLE_PATTERNS.items()}
    }
    return compiled

def has_any(patterns: list[re.Pattern], text: str) -> bool:
    if not text:
        return False
    for p in patterns:
        if p.search(text):
            return True
    return False

def find_all_roles(text: str, compiled_roles: dict, raw_text: str = None) -> list[tuple[str, float, str, int, int]]:
    matches = []
    source = raw_text if raw_text is not None else text
    if not source:
        return matches
    for role_key, (w, regs) in compiled_roles.items():
        for r in regs:
            m = r.search(source)
            if m:
                matches.append((role_key, w, m.group(0), m.start(), m.end()))
                break
    return matches

def seniority_multiplier(text: str, compiled_seniority: list[tuple[re.Pattern, float]]) -> float:
    mul = 1.0
    for rgx, w in compiled_seniority:
        if rgx.search(text):
            mul = max(mul, w)
    return mul

def normalize_source(raw: str | None) -> str:
    if raw is None or pd.isna(raw) or str(raw).strip() == "":
        return "unknown"
    s = str(raw).strip().lower()
    for k, v in SOURCE_NORMALIZE.items():
        if k in s:
            return v
    return "unknown"

def source_weight(src: str) -> float:
    return SOURCE_WEIGHTS.get(src, SOURCE_WEIGHTS["unknown"])

def recency_decay(anchor_ts: pd.Timestamp, ts: pd.Timestamp, half_life_days: float) -> float:
    if pd.isna(ts):
        return 1.0
    delta = (anchor_ts - ts).days
    if delta < 0:
        delta = 0
    return 2.0 ** (-(delta / max(half_life_days, 1e-6)))

_sentence_split_re = re.compile(r'(?<=[\.\?\!])\s+')
_word_re = re.compile(r"\b[\w\'\-\&\.]+\b")

def extract_context_for_span(raw_text: str, span_start: int, span_end: int) -> str:
    if not raw_text:
        return ""
    raw_text = str(raw_text).replace("\n", " ").strip()
    sentences = _sentence_split_re.split(raw_text)
    spans = []
    off = 0
    for s in sentences:
        length = len(s)
        spans.append((off, off + length))
        off += length
        while off < len(raw_text) and raw_text[off].isspace():
            off += 1
    sent_idx = None
    for i, (s0, s1) in enumerate(spans):
        if span_start >= s0 and span_start < s1:
            sent_idx = i
            break
    if sent_idx is None:
        sent0, sent1 = 0, len(raw_text)
        sent_text = raw_text
    else:
        sent0, sent1 = spans[sent_idx]
        sent_text = raw_text[sent0:sent1]
    words = []
    for m in _word_re.finditer(sent_text):
        wstart = sent0 + m.start()
        wend = sent0 + m.end()
        words.append((m.group(0), wstart, wend))
    if not words:
        s0 = max(0, span_start - 30)
        s1 = min(len(raw_text), span_end + 30)
        return raw_text[s0:s1].strip()
    start_word_idx = None
    end_word_idx = None
    for i, (_, wstart, wend) in enumerate(words):
        if wend > span_start and start_word_idx is None:
            start_word_idx = i
        if wstart < span_end:
            end_word_idx = i
    if start_word_idx is None:
        start_word_idx = 0
    if end_word_idx is None:
        end_word_idx = len(words) - 1
    n_words = len(words)
    if start_word_idx == 0:
        take_start = 0
        take_end = min(n_words - 1, end_word_idx + 4)
    elif end_word_idx == n_words - 1:
        take_start = max(0, start_word_idx - 4)
        take_end = n_words - 1
    else:
        take_start = max(0, start_word_idx - 2)
        take_end = min(n_words - 1, end_word_idx + 2)
    selected = [words[i][0] for i in range(take_start, take_end + 1)]
    ctx = " ".join(selected).strip()
    return ctx

# ================================
# 2) Row-level detection & scoring
# ================================

def detect_hiring_and_score(row, text_cols_present, compiled, ts_col, src_col, anchor_ts, half_life_days, origin="company"):
    raw_text = combine_text_fields(row, text_cols_present)
    if not raw_text or str(raw_text).strip() == "":
        return None

    raw_text_norm = _normalize_apostrophes_and_spaces(raw_text)
    text = raw_text_norm.lower()
    if not text or text.strip() == "":
        return None

    # Negation check
    if has_any(compiled["negations"], text):
        return None

    strong = has_any(compiled["strong"], text)
    moderate = has_any(compiled["moderate"], text)
    hashtag = has_any(compiled["hashtags"], text)
    joblink = has_any(compiled["joblinks"], text)

    roles_found = find_all_roles(text, compiled["roles"], raw_text=raw_text_norm)
    has_role = len(roles_found) > 0

    # Apply rules:
    # - Strong signals => always hit (we treat "we're hiring" etc. as hits even if role missing)
    # - Moderate signals require role/joblink to be treated as hit
    # - Hashtag-only => count as hit (per your instruction)
    # - Role-only without any hiring signal => NOT a hit
    passes = False
    signal_strength = "none"
    base = 0.0

    if strong:
        passes = True
        signal_strength = "strong"
        base = BASE_SCORE_STRONG
    elif moderate and (has_role or joblink):
        passes = True
        signal_strength = "moderate"
        base = BASE_SCORE_MODERATE
    elif hashtag:
        passes = True
        signal_strength = "hashtag"
        base = BASE_SCORE_HASHTAG_ONLY

    if not passes:
        return None

    if has_role:
        role_weight_max = max(w for _, w, _, _, _ in roles_found)
    else:
        role_weight_max = 0.7 if joblink else 0.6

    relevant_count = sum(1 for _, w, _, _, _ in roles_found if w >= 0.9)
    adjacent_count = sum(1 for _, w, _, _, _ in roles_found if 0.7 <= w < 0.9)
    irrelevant_count = sum(1 for _, w, _, _, _ in roles_found if w < 0.7)

    senior_mul = seniority_multiplier(text, compiled["seniority"])

    src = normalize_source(row.get(src_col)) if src_col else "unknown"
    src_w = source_weight(src)

    link_mul = JOB_LINK_BONUS if joblink else 1.0
    multi_role_mul = min(1.0 + MULTI_ROLE_STEP * max(len(roles_found) - 1, 0), MULTI_ROLE_CAP)

    ts = row.get(ts_col) if ts_col else pd.NaT
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    decay = recency_decay(anchor_ts, ts, half_life_days)

    score = base
    score *= role_weight_max
    score *= senior_mul
    score *= src_w
    score *= link_mul
    score *= multi_role_mul
    score *= decay

    if roles_found:
        categories = ", ".join(sorted({rk for rk, _, _, _, _ in roles_found}))
    else:
        categories = "general"

    snippet = raw_text.replace("\n", " ").strip()[:320]

    matched_role_strings = []
    matched_contexts = []
    for role_key, w, matched_str, s0, s1 in roles_found:
        ms = matched_str.strip()
        if ms:
            matched_role_strings.append(ms)
            ctx = extract_context_for_span(raw_text, s0, s1)
            if ctx:
                matched_contexts.append(ctx)

    def uniq_keep_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    matched_role_strings = uniq_keep_order(matched_role_strings)
    matched_contexts = uniq_keep_order(matched_contexts)

    # Robust company_id retrieval (works for both company and lead rows if present)
    company_id_val = None
    for k in ("company_id", "_company_id", "companyId", "company_id_clean", "employerCompanyId", "co_company_id"):
        try:
            v = row.get(k)
        except Exception:
            v = None
        if pd.notna(v) and str(v).strip() != "":
            company_id_val = v
            break
    if company_id_val is None:
        # Fallback: look for any column name containing 'company' with non-null values
        for col in row.index:
            if "company" in col.lower():
                v = row.get(col)
                if pd.notna(v) and str(v).strip() != "":
                    company_id_val = v
                    break

    return {
        "company_id": company_id_val,
        "_ts": ts,
        "_source": src,
        "origin": origin,  # 'company' or 'lead'
        "signal_strength": signal_strength,
        "job_link": int(joblink),
        "hiring_post": 1,
        "hiring_score": float(score),
        "relevant_roles_count": int(relevant_count),
        "adjacent_roles_count": int(adjacent_count),
        "irrelevant_roles_count": int(irrelevant_count),
        "matched_roles": [m for m in matched_role_strings],
        "matched_roles_str": ", ".join(matched_role_strings) if matched_role_strings else "",
        "matched_roles_contexts": ", ".join(matched_contexts) if matched_contexts else "",
        "categories": categories,
        "snippet": snippet
    }

# ================================
# 3) Aggregation helpers
# ================================

def aggregate_company_level(hit_df: pd.DataFrame, anchor_ts: pd.Timestamp, windows: dict[str, timedelta], origin_label: str):
    """
    Aggregate hits per company for the provided hits DataFrame.
    origin_label: 'co' or 'lead' (prefix for column names)
    Returns aggregated DataFrame with columns like:
      {origin_label}_hiring_posts_7d, {origin_label}_total_hiring_score_7d, ...
    Also returns matched_roles_all / contexts for that origin as '<origin_label>_matched_roles_all'
    """
    if "company_id" not in hit_df.columns and "_company_id" in hit_df.columns:
        hit_df = hit_df.copy()
        hit_df["company_id"] = hit_df["_company_id"]

    companies = pd.DataFrame({"company_id": hit_df["company_id"].unique()}) if not hit_df.empty else pd.DataFrame(columns=["company_id"])
    out = companies.copy()

    for label, delta in windows.items():
        cutoff = anchor_ts - delta
        wmask = hit_df["_ts"].notna() & (hit_df["_ts"] >= cutoff)
        wdf = hit_df[wmask].copy()

        if "matched_roles_str" not in wdf.columns:
            wdf["matched_roles_str"] = ""
        if "matched_roles_contexts" not in wdf.columns:
            wdf["matched_roles_contexts"] = ""

        if wdf.empty:
            agg = pd.DataFrame({"company_id": out["company_id"]})
            agg[f"{origin_label}_hiring_posts_{label}"] = 0
            agg[f"{origin_label}_total_hiring_score_{label}"] = 0.0
            agg[f"{origin_label}_relevant_hiring_count_{label}"] = 0
            agg[f"{origin_label}_adjacent_hiring_count_{label}"] = 0
            agg[f"{origin_label}_irrelevant_hiring_count_{label}"] = 0
            agg[f"{origin_label}_hiring_intensity_{label}"] = 0.0
            agg[f"{origin_label}_unique_relevant_roles_{label}"] = 0
            agg[f"{origin_label}_hiring_velocity_{label}"] = 0.0
            agg[f"{origin_label}_matched_roles_{label}"] = ""
            agg[f"{origin_label}_matched_roles_contexts_{label}"] = ""
            out = out.merge(agg, on="company_id", how="left")
            continue

        wdf_roles = wdf.assign(role_tokens=wdf["categories"].str.split(r"\s*,\s*"))
        wdf_roles = wdf_roles.explode("role_tokens")
        wdf_roles["role_tokens"] = wdf_roles["role_tokens"].fillna("")

        base_agg = (wdf
                    .groupby("company_id", as_index=False)
                    .agg(**{
                        f"{origin_label}_hiring_posts_{label}": ("hiring_post", "sum"),
                        f"{origin_label}_total_hiring_score_{label}": ("hiring_score", "sum"),
                        f"{origin_label}_relevant_hiring_count_{label}": ("relevant_roles_count", "sum"),
                        f"{origin_label}_adjacent_hiring_count_{label}": ("adjacent_roles_count", "sum"),
                        f"{origin_label}_irrelevant_hiring_count_{label}": ("irrelevant_roles_count", "sum"),
                    }))

        if f"{origin_label}_hiring_posts_{label}" in base_agg.columns:
            base_agg[f"{origin_label}_hiring_intensity_{label}"] = np.where(
                base_agg[f"{origin_label}_hiring_posts_{label}"] > 0,
                base_agg[f"{origin_label}_total_hiring_score_{label}"] / base_agg[f"{origin_label}_hiring_posts_{label}"],
                0.0
            )
            days = max(int(delta.days), 1)
            base_agg[f"{origin_label}_hiring_velocity_{label}"] = base_agg[f"{origin_label}_hiring_posts_{label}"] / (days / 7.0)

        def is_rel(role_key: str) -> bool:
            rk = (role_key or "").strip().lower()
            if rk == "" or rk in ("general", "unspecified"):
                return False
            wt = None
            for k, (w, _) in ROLE_PATTERNS.items():
                if k == rk:
                    wt = w
                    break
            if wt is None:
                return False
            return wt >= 0.7

        uniq = (wdf_roles
                .assign(_is_rel=wdf_roles["role_tokens"].map(is_rel))
                .query("_is_rel == True")
                .groupby("company_id", as_index=False)
                .agg(**{f"{origin_label}_unique_relevant_roles_{label}": ("role_tokens", lambda s: len(set([x for x in s if x])))} )
                )

        def concat_unique_strs(series):
            items = []
            for v in series.dropna().astype(str):
                if not v:
                    continue
                for part in [p.strip() for p in v.split(",")]:
                    if part and part not in items:
                        items.append(part)
            return ", ".join(items)

        roles_agg = wdf.groupby("company_id", as_index=False).agg({
            "matched_roles_str": lambda s: concat_unique_strs(s),
            "matched_roles_contexts": lambda s: concat_unique_strs(s)
        }).rename(columns={
            "matched_roles_str": f"{origin_label}_matched_roles_{label}",
            "matched_roles_contexts": f"{origin_label}_matched_roles_contexts_{label}"
        })

        agg = base_agg.merge(uniq, on="company_id", how="left")
        agg = agg.merge(roles_agg, on="company_id", how="left")
        agg = agg.fillna({f"{origin_label}_matched_roles_{label}": "", f"{origin_label}_matched_roles_contexts_{label}": ""})
        agg = agg.fillna({f"{origin_label}_unique_relevant_roles_{label}": 0})

        for c in agg.columns:
            if c == "company_id":
                continue
            if pd.api.types.is_numeric_dtype(agg[c].dtype):
                agg[c] = agg[c].fillna(0)

        out = out.merge(agg, on="company_id", how="left")

    # Freshness for this origin
    if not hit_df.empty:
        last = (hit_df[hit_df["_ts"].notna()]
                .sort_values("_ts")
                .groupby("company_id", as_index=False)
                .agg(**{f"{origin_label}_last_hiring_ts": ("_ts", "max")}))
        last[f"{origin_label}_days_since_last_hiring"] = (anchor_ts - last[f"{origin_label}_last_hiring_ts"]).dt.days
        out = out.merge(last, on="company_id", how="left")
    else:
        out[f"{origin_label}_last_hiring_ts"] = pd.NaT
        out[f"{origin_label}_days_since_last_hiring"] = np.nan

    # Aggregate matched roles across ALL hits for this origin
    if not hit_df.empty:
        if "matched_roles" not in hit_df.columns:
            hit_df["matched_roles"] = [[] for _ in range(len(hit_df))]
        if "matched_roles_contexts" not in hit_df.columns:
            hit_df["matched_roles_contexts"] = ""

        def combine_role_lists(series):
            outlist = []
            for l in series:
                if isinstance(l, (list, tuple)):
                    for item in l:
                        if item and item not in outlist:
                            outlist.append(item)
                else:
                    s = str(l)
                    for part in [p.strip() for p in s.split(",")]:
                        if part and part not in outlist:
                            outlist.append(part)
            return ", ".join(outlist)

        def combine_contexts(series):
            outlist = []
            for v in series.dropna().astype(str):
                if not v:
                    continue
                for part in [p.strip() for p in v.split(",")]:
                    if part and part not in outlist:
                        outlist.append(part)
            return ", ".join(outlist)

        roles_all = (hit_df
                     .groupby("company_id", as_index=False)
                     .agg(**{f"{origin_label}_matched_roles_all": ("matched_roles", combine_role_lists),
                            f"{origin_label}_matched_roles_contexts_all": ("matched_roles_contexts", combine_contexts)}))
        out = out.merge(roles_all, on="company_id", how="left")
    else:
        out[f"{origin_label}_matched_roles_all"] = ""
        out[f"{origin_label}_matched_roles_contexts_all"] = ""

    # Fill numeric NaNs and string NaNs
    for c in out.columns:
        if c == "company_id":
            continue
        if pd.api.types.is_numeric_dtype(out[c].dtype):
            out[c] = out[c].fillna(0)
        if out[c].dtype == object:
            out[c] = out[c].fillna("")
    return out

# ================================
# 4) Main
# ================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="company_activity_clean.csv", help="Company-level activity cleaned file (required)")
    ap.add_argument("--lead_input", default="activity_clean.csv", help="Lead-level activity cleaned file (optional)")
    ap.add_argument("--out_features", default="hiring_features.csv", help="Per-company hiring features")
    ap.add_argument("--out_snippets", default="hiring_snippets.csv", help="Per-post hiring snippets (QA)")
    ap.add_argument("--half_life_days", type=float, default=45.0, help="Recency half-life in days")
    ap.add_argument("--windows", default="7,30,90", help="Comma-separated day windows for aggregation")
    ap.add_argument("--company_hiring_weight", type=float, default=0.6, help="Weight for company-origin hiring score when combining")
    ap.add_argument("--lead_hiring_weight", type=float, default=0.4, help="Weight for lead-origin hiring score when combining")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df_co = pd.read_csv(args.input, dtype=str)
    print(f"✅ Loaded company activity: {args.input} ({len(df_co):,} rows, {len(df_co.columns)} cols)")

    df_le = None
    if args.lead_input:
        if not os.path.exists(args.lead_input):
            raise FileNotFoundError(f"Lead input file not found: {args.lead_input}")
        df_le = pd.read_csv(args.lead_input, dtype=str)
        print(f"✅ Loaded lead activity: {args.lead_input} ({len(df_le):,} rows, {len(df_le.columns)} cols)")

    # Identify columns using company dataset (fallback to lead if not found)
    col_source_df = df_co if not df_co.empty else (df_le if df_le is not None else pd.DataFrame())
    company_col = pick_first_existing(col_source_df, COMPANY_ID_COLS)
    if not company_col:
        # We will rely on per-row fallback heuristics later, but warn user
        print("⚠️ Warning: Could not find a canonical company id column in the primary input. We'll attempt fallback detection per-row.")

    ts_col_co = pick_first_existing(df_co, TIMESTAMP_COLS) if not df_co.empty else None
    ts_col_le = pick_first_existing(df_le, TIMESTAMP_COLS) if (df_le is not None and not df_le.empty) else None

    text_cols_present_co = [c for c in TEXT_COLS if c in df_co.columns] if not df_co.empty else []
    text_cols_present_le = [c for c in TEXT_COLS if c in df_le.columns] if (df_le is not None and not df_le.empty) else []

    if not text_cols_present_co and not text_cols_present_le:
        raise ValueError(f"No text/content columns found in either input. Tried: {TEXT_COLS}")

    src_col_co = pick_first_existing(df_co, SOURCE_COLS) if not df_co.empty else None
    src_col_le = pick_first_existing(df_le, SOURCE_COLS) if (df_le is not None and not df_le.empty) else None

    # Normalize working copies and annotate origin
    work_co = df_co.copy()
    if company_col:
        work_co["_company_id"] = work_co[company_col]
    else:
        work_co["_company_id"] = work_co.get("company_id", pd.Series([None]*len(work_co))).astype(object)

    work_co["_ts"] = coerce_utc(work_co[ts_col_co]) if ts_col_co else pd.NaT
    work_co["_src_norm"] = work_co[src_col_co].map(normalize_source) if src_col_co else "unknown"
    work_co["_origin"] = "company"

    work_le = None
    if df_le is not None:
        work_le = df_le.copy()
        # attempt to use company mapping in lead file
        lead_company_col = pick_first_existing(work_le, COMPANY_ID_COLS)
        if lead_company_col:
            work_le["_company_id"] = work_le[lead_company_col]
        else:
            work_le["_company_id"] = work_le.get("company_id", pd.Series([None]*len(work_le))).astype(object)

        work_le["_ts"] = coerce_utc(work_le[ts_col_le]) if ts_col_le else pd.NaT
        work_le["_src_norm"] = work_le[src_col_le].map(normalize_source) if src_col_le else "unknown"
        work_le["_origin"] = "lead"

    # Determine anchor time across both datasets
    anchor_ts = detect_anchor_ts([work_co["_ts"], (work_le["_ts"] if work_le is not None else pd.Series([], dtype="datetime64[ns]"))])
    print(f"Anchor time (max ts or now): {anchor_ts}")

    compiled = compile_patterns()

    # Detect row-level hiring & score for both origins
    rows = []
    # company origin
    if not work_co.empty:
        for _, row in work_co.iterrows():
            res = detect_hiring_and_score(
                row=row,
                text_cols_present=text_cols_present_co,
                compiled=compiled,
                ts_col="_ts",
                src_col="_src_norm",
                anchor_ts=anchor_ts,
                half_life_days=args.half_life_days,
                origin="company"
            )
            if res is not None:
                rows.append(res)

    # lead origin
    if work_le is not None and not work_le.empty:
        for _, row in work_le.iterrows():
            res = detect_hiring_and_score(
                row=row,
                text_cols_present=text_cols_present_le,
                compiled=compiled,
                ts_col="_ts",
                src_col="_src_norm",
                anchor_ts=anchor_ts,
                half_life_days=args.half_life_days,
                origin="lead"
            )
            if res is not None:
                rows.append(res)

    if not rows:
        print("⚠️ No hiring posts detected in either input. Writing empty outputs.")
        pd.DataFrame(columns=["company_id"]).to_csv(args.out_features, index=False)
        pd.DataFrame(columns=["company_id", "ts", "source", "origin", "categories", "snippet", "matched_roles_str", "matched_roles_contexts"]).to_csv(args.out_snippets, index=False)
        print(f"✅ Saved empty hiring features to: {args.out_features}")
        print(f"✅ Saved empty hiring snippets to: {args.out_snippets}")
        return

    hits = pd.DataFrame(rows)
    print(f"✅ Detected hiring posts (total): {len(hits):,}")

    # Defensive: ensure company_id present for hits; best-effort inference
    if "company_id" not in hits.columns:
        if "_company_id" in hits.columns:
            hits["company_id"] = hits["_company_id"]
            print("ℹ️ 'company_id' created from '_company_id' in hits.")
        else:
            for fallback_col in ("companyId", "company_id_clean", "employerCompanyId"):
                if fallback_col in hits.columns:
                    hits["company_id"] = hits[fallback_col]
                    print(f"ℹ️ 'company_id' created from '{fallback_col}' in hits.")
                    break

    if "company_id" not in hits.columns:
        inferred = None
        for col in hits.columns:
            if "company" in col.lower():
                if hits[col].notna().any():
                    inferred = col
                    break
        if inferred:
            hits["company_id"] = hits[inferred]
            print(f"⚠️ 'company_id' inferred from column '{inferred}'.")
        else:
            hits["company_id"] = np.nan
            print("⚠️ Could not find any company id in hits rows; 'company_id' filled with NaN. Downstream merges may be sparse.")

    hits["company_id"] = hits["company_id"].astype(object)

    # Parse windows
    win_map = {}
    for w in [w.strip() for w in args.windows.split(",") if w.strip()]:
        try:
            d = int(w)
            win_map[f"{d}d"] = timedelta(days=d)
        except Exception:
            pass
    if not win_map:
        win_map = {"30d": timedelta(days=30)}

    # Split hits by origin and aggregate separately
    hits_co = hits[hits["origin"] == "company"].copy()
    hits_le = hits[hits["origin"] == "lead"].copy()

    features_co = aggregate_company_level(hits_co, anchor_ts, win_map, origin_label="co")
    features_le = aggregate_company_level(hits_le, anchor_ts, win_map, origin_label="lead")

    # Ensure union of companies from both
    # Combine and clean company_id lists safely
    company_ids_co = [str(x) for x in features_co["company_id"].dropna().unique() if str(x).lower() != "none"]
    company_ids_le = [str(x) for x in features_le["company_id"].dropna().unique() if str(x).lower() != "none"]

    # Merge, deduplicate, and sort safely
    all_company_ids = sorted(set(company_ids_co + company_ids_le))

    all_companies = pd.DataFrame({"company_id": all_company_ids})

    features = all_companies.merge(features_co, on="company_id", how="left").merge(features_le, on="company_id", how="left")

    # Combine matched role lists across origins into unified fields
    def combine_unique_comma(a: str, b: str) -> str:
        a_parts = [p.strip() for p in str(a).split(",") if p and str(p).strip()]
        b_parts = [p.strip() for p in str(b).split(",") if p and str(p).strip()]
        out = []
        for p in a_parts + b_parts:
            if p not in out:
                out.append(p)
        return ", ".join(out)

    # Create combined / derived columns per window using weights
    cw_co = float(args.company_hiring_weight)
    cw_le = float(args.lead_hiring_weight)
    # normalize weights if they don't sum to 1
    s = cw_co + cw_le
    if s <= 0:
        cw_co, cw_le = 0.6, 0.4
        s = 1.0
    cw_co = cw_co / s
    cw_le = cw_le / s

    for label in win_map.keys():
        # posts
        co_posts = features.get(f"co_hiring_posts_{label}", 0)
        le_posts = features.get(f"lead_hiring_posts_{label}", 0)
        features[f"combined_hiring_posts_{label}"] = (co_posts.fillna(0) if hasattr(co_posts, "fillna") else co_posts) + (le_posts.fillna(0) if hasattr(le_posts, "fillna") else le_posts)

        # total scores
        co_total = features.get(f"co_total_hiring_score_{label}", 0)
        le_total = features.get(f"lead_total_hiring_score_{label}", 0)
        co_total = co_total.fillna(0) if hasattr(co_total, "fillna") else co_total
        le_total = le_total.fillna(0) if hasattr(le_total, "fillna") else le_total

        features[f"combined_total_hiring_score_{label}"] = (cw_co * co_total) + (cw_le * le_total)

        # intensity: recompute from combined_total / combined_posts (if posts>0), otherwise 0
        combined_posts_series = features[f"combined_hiring_posts_{label}"]
        combined_total_series = features[f"combined_total_hiring_score_{label}"]
        features[f"combined_hiring_intensity_{label}"] = np.where(
            combined_posts_series > 0,
            combined_total_series / combined_posts_series,
            0.0
        )

        # unique relevant roles: union of both
        co_uniq = features.get(f"co_unique_relevant_roles_{label}", 0)
        le_uniq = features.get(f"lead_unique_relevant_roles_{label}", 0)
        # A conservative approach: take max of both counts (they may overlap)
        features[f"combined_unique_relevant_roles_{label}"] = np.maximum(
            co_uniq.fillna(0) if hasattr(co_uniq, "fillna") else co_uniq,
            le_uniq.fillna(0) if hasattr(le_uniq, "fillna") else le_uniq
        )

        # combine matched role lists & contexts
        features[f"matched_roles_{label}"] = features.apply(lambda r: combine_unique_comma(r.get(f"co_matched_roles_{label}", ""), r.get(f"lead_matched_roles_{label}", "")), axis=1)
        features[f"matched_roles_contexts_{label}"] = features.apply(lambda r: combine_unique_comma(r.get(f"co_matched_roles_contexts_{label}", ""), r.get(f"lead_matched_roles_contexts_{label}", "")), axis=1)

    # Combine last hiring ts (most recent across origins)
    def pick_most_recent(row, col_co, col_le):
        vco = row.get(col_co)
        vle = row.get(col_le)
        if pd.isna(vco) and pd.isna(vle):
            return pd.NaT
        if pd.isna(vco):
            return vle
        if pd.isna(vle):
            return vco
        try:
            vco_t = pd.to_datetime(vco, utc=True, errors="coerce")
            vle_t = pd.to_datetime(vle, utc=True, errors="coerce")
            return max(vco_t, vle_t)
        except Exception:
            return vco if vco >= vle else vle

    features["last_hiring_ts"] = features.apply(lambda r: pick_most_recent(r, "co_last_hiring_ts", "lead_last_hiring_ts"), axis=1)
    features["days_since_last_hiring"] = (anchor_ts - pd.to_datetime(features["last_hiring_ts"], utc=True, errors="coerce")).dt.days
    features["days_since_last_hiring"] = features["days_since_last_hiring"].fillna(np.nan)

    # Merge matched_roles_all across origins into single fields
    features["matched_roles_all"] = features.apply(lambda r: combine_unique_comma(r.get("co_matched_roles_all", ""), r.get("lead_matched_roles_all", "")), axis=1)
    features["matched_roles_contexts_all"] = features.apply(lambda r: combine_unique_comma(r.get("co_matched_roles_contexts_all", ""), r.get("lead_matched_roles_contexts_all", "")), axis=1)

    # Fill numeric NaNs with 0 and string NaNs with empty strings
    for c in features.columns:
        if c == "company_id":
            continue
        if pd.api.types.is_numeric_dtype(features[c].dtype):
            features[c] = features[c].fillna(0)
        if features[c].dtype == object:
            features[c] = features[c].fillna("")

    # Order columns for readability: company_id, last, days_since..., then by window
    ordered_cols = ["company_id", "last_hiring_ts", "days_since_last_hiring"]
    for lbl in sorted(win_map.keys(), key=lambda x: int(x.replace("d", ""))):
        ordered_cols += [
            f"co_hiring_posts_{lbl}",
            f"co_total_hiring_score_{lbl}",
            f"co_hiring_intensity_{lbl}",
            f"co_hiring_velocity_{lbl}",
            f"co_relevant_hiring_count_{lbl}",
            f"co_adjacent_hiring_count_{lbl}",
            f"co_irrelevant_hiring_count_{lbl}",
            f"co_unique_relevant_roles_{lbl}",
            f"co_matched_roles_{lbl}",
            f"co_matched_roles_contexts_{lbl}",

            f"lead_hiring_posts_{lbl}",
            f"lead_total_hiring_score_{lbl}",
            f"lead_hiring_intensity_{lbl}",
            f"lead_hiring_velocity_{lbl}",
            f"lead_relevant_hiring_count_{lbl}",
            f"lead_adjacent_hiring_count_{lbl}",
            f"lead_irrelevant_hiring_count_{lbl}",
            f"lead_unique_relevant_roles_{lbl}",
            f"lead_matched_roles_{lbl}",
            f"lead_matched_roles_contexts_{lbl}",

            f"combined_hiring_posts_{lbl}",
            f"combined_total_hiring_score_{lbl}",
            f"combined_hiring_intensity_{lbl}",
            f"combined_hiring_velocity_{lbl}" if f"combined_hiring_velocity_{lbl}" in features.columns else "",  # keep safe
            f"combined_unique_relevant_roles_{lbl}",
            f"matched_roles_{lbl}",
            f"matched_roles_contexts_{lbl}",
        ]
    # Keep only existing
    ordered_cols = [c for c in ordered_cols if c and c in features.columns]
    features = features[ordered_cols + [c for c in features.columns if c not in ordered_cols]]

    # Write features CSV
    features.to_csv(args.out_features, index=False)
    print(f"✅ Saved per-company hiring features to: {args.out_features}")

    # Write snippets file (QA) - include origin column for traceability
    for k in ["matched_roles_str", "matched_roles_contexts"]:
        if k not in hits.columns:
            hits[k] = ""
    snippets = hits.rename(columns={"_ts": "ts", "_source": "source"})[["company_id", "ts", "source", "origin", "categories", "snippet", "matched_roles_str", "matched_roles_contexts"]].copy()
    snippets.to_csv(args.out_snippets, index=False)
    print(f"✅ Saved hiring QA snippets to: {args.out_snippets}")

    # Console top sample by combined score (prefer 30d)
    pref = None
    for p in ["30d", "90d", "7d"]:
        if f"combined_total_hiring_score_{p}" in features.columns:
            pref = p
            break
    if pref:
        top = features.sort_values(f"combined_total_hiring_score_{pref}", ascending=False).head(10)
        with pd.option_context("display.max_columns", 120, "display.width", 220):
            display_cols = ["company_id", f"combined_total_hiring_score_{pref}", f"combined_hiring_posts_{pref}", f"combined_hiring_intensity_{pref}", f"combined_unique_relevant_roles_{pref}"]
            display_cols = [c for c in display_cols if c in top.columns]
            print(f"\nTop companies by combined_total_hiring_score_{pref}:")
            print(top[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
