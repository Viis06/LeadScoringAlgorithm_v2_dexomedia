#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step3b_hiring_features.py

High-precision hiring intent extraction (company-level).

Input:
  - company_activity_clean.csv

Outputs:
  - hiring_features.csv      (per-company engineered features across 7/30/90d)
  - hiring_snippets.csv      (QA: all detected hiring posts with snippets)

Key features:
  - Robust detection (positive signals + negations + job-link heuristics)
  - Role relevance aligned to client services (sales automation, growth & revenue ops)
  - Weighting by role relevance, seniority, recency (half-life), source, job-link presence, and multi-role intensity
  - Windowed aggregation (7d/30d/90d) + freshness fields

Run:
  python step3b_hiring_features.py \
    --input company_activity_clean.csv \
    --out_features hiring_features.csv \
    --out_snippets hiring_snippets.csv \
    --half_life_days 45 \
    --windows 7,30,90
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

# Map noisy source values to normalized buckets
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

# Source weights (authoritativeness of the signal)
SOURCE_WEIGHTS = {
    "post": 1.00,
    "job": 1.00,        # explicit job postings
    "repost": 0.85,     # still meaningful if they share jobs
    "comment": 0.50,
    "unknown": 0.70
}

# Job-board / ATS domains (bonus if present)
JOB_LINK_DOMAINS = [
    r"linkedin\.com/jobs", r"indeed\.", r"greenhouse\.io", r"lever\.co", r"workable\.com",
    r"workday\.", r"bamboohr\.", r"icims\.", r"smartrecruiters\.", r"jazzhr\.",
    r"boards\.greenhouse\.io", r"jobs\.", r"recruitee\.", r"ashbyhq\.", r"jobvite\."
]

# Positive hiring signals (ordered: strongest to broad)
# Note: added more robust patterns for common punctuation/apostrophes and explicit "we are hiring"
HIRING_SIGNALS_STRONG = [
    r"\bapply\s+now\b",
    r"\bapplications?\s+open\b",
    r"\bvacanc(y|ies)\b",
    r"\bjob\s+(opening|post(ing)?)\b",
    r"\bnow\s+hiring\b",
    r"\bwe(?:'|’)?re\s+hiring\b",      # matches we're, we’re
    r"\bwe\s+are\s+hiring\b",
    r"\bjoin\s+our\s+team\b",
    r"\bcareers?\s+(page|site)\b",
    r"\brecruit(ing|ment)\b"
]
HIRING_SIGNALS_MODERATE = [
    r"\bhiring\s+for\b",
    r"\blooking\s+for\b",
    r"\bwe\s+are\s+expanding\b",
    r"\bgrow(ing|th)\b",
    r"\bteam\s+is\s+growing\b"
]
HIRING_HASHTAGS = [
    r"#hiring\b", r"#nowhiring\b", r"#werehiring\b", r"#we?re?hiring\b",
    r"#career(s)?\b", r"#job(s)?\b", r"#vacancy\b"
]

# Negations / closures (to avoid false positives)
NEGATIONS = [
    r"\bnot\s+hiring\b", r"\bno\s+longer\s+hiring\b", r"\bhiring\s+freeze\b",
    r"\b(pos(ition)?|role)\s+(has been|is)\s+filled\b",
    r"\bapplications?\s+(closed|are closed)\b", r"\bwe\s+stopped\s+hiring\b"
]

# Seniority multipliers (optional boost)
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
    # ===== Highest Intent: Marketing & Growth Leadership (1.0) =====
    "founder_ceo_owner": (1.0, [
        r"\bfounder\b", r"\bco-?founder\b", r"\bowner\b", r"\bproprietor\b",
        r"\bceo\b", r"\bchief\s+executive\s+officer\b",
        r"\bmanaging\s+director\b", r"\bmd\b", r"\bpartner\b"
    ]),

    "marketing_growth_leadership": (1.0, [
        r"\bcmo\b", r"\bchief\s+marketing\s+officer\b",
        r"\bvp\s+marketing\b", r"\bavp\s+marketing\b",
        r"\bhead[-\s]?of\s+marketing\b", r"\bmarketing\s+director\b",
        r"\bmarketing\s+head\b",

        r"\bvp\s+growth\b", r"\bhead[-\s]?of\s+growth\b", r"\bgrowth\s+director\b",
        r"\bhead[-\s]?of\s+acquisition\b", r"\bacquisition\s+director\b",
        r"\bhead[-\s]?of\s+performance\b", r"\bperformance\s+marketing\s+director\b"
    ]),

    # ===== Highest Intent: Paid Media / Acquisition (1.0) =====
    "paid_media_performance": (1.0, [
        r"\bperformance\s+marketing\b", r"\bpaid\s+media\b",
        r"\bpaid\s+social\b", r"\bpaid\s+search\b",
        r"\bmedia\s+buyer\b", r"\bmedia\s+buying\b",
        r"\buser\s+acquisition\b", r"\bacquisition\s+manager\b", r"\bacquisition\s+lead\b",
        r"\bgrowth\s+marketer\b", r"\bgrowth\s+marketing\b",

        r"\bgoogle\s+ads\b", r"\bppc\b", r"\bsem\b",
        r"\bsearch\s+ads\b", r"\bdisplay\s+ads\b", r"\byoutube\s+ads\b",
        r"\bmeta\s+ads\b|\bfacebook\s+ads\b|\binstagram\s+ads\b",
        r"\btiktok\s+ads\b", r"\bsnapchat\s+ads\b", r"\bpinterest\s+ads\b",
        r"\bprogrammatic\b", r"\bperformance\s+manager\b",
        r"\bdemand\s+gen(eration)?\b", r"\blead\s+gen(eration)?\b"
    ]),

    # ===== Highest Intent: eCommerce / Marketplace Growth (0.95) =====
    "ecommerce_marketplace_growth": (0.95, [
        r"\be-?commerce\b|\becommerce\b", r"\bd2c\b|\bdirect[-\s]?to[-\s]?consumer\b",
        r"\bmarketplace\b", r"\bmarketplace\s+growth\b",
        r"\bmarketplace\s+ops\b|\bmarketplace\s+operations\b",
        r"\bamazon\b", r"\bflipkart\b", r"\bmyntra\b", r"\bajio\b", r"\btata\s+cliq\b", r"\bnykaa\b",

        r"\bamazon\s+ads\b|\bamazon\s+advertis(ing|ement)\b",
        r"\bmarketplace\s+ads\b", r"\bmarketplace\s+advertis(ing|ement)\b",
        r"\bcatalog(ue)?\b", r"\blistings?\b", r"\blisting\s+optim(iz|is)ation\b",
        r"\baccount\s+manager\b.*\b(amazon|flipkart|myntra|nykaa)\b",
        r"\bcategory\s+manager\b", r"\becom(merce)?\s+manager\b",
        r"\bkey\s+account\b.*\b(amazon|flipkart|myntra|ajio|nykaa)\b"
    ]),

    # ===== High Intent: CRO / Product Growth / UX (0.95) =====
    "cro_conversion_ux": (0.95, [
        r"\bcro\b", r"\bconversion\s+rate\s+optim(iz|is)ation\b",
        r"\bconversion\s+optim(iz|is)ation\b", r"\bconversion\s+specialist\b",
        r"\bcro\s+(lead|manager|head)\b",
        r"\bgrowth\s+product\b|\bproduct\s+growth\b",
        r"\bux\b|\buser\s+experience\b", r"\bux\s+research\b",
        r"\bfunnel\s+optim(iz|is)ation\b", r"\blanding\s+page\b.*\boptim(iz|is)ation\b",
        r"\bcheckout\b.*\boptim(iz|is)ation\b", r"\baov\b|\basket\s+size\b"
    ]),

    # ===== High Intent: Creative Performance / UGC / Video (0.95) =====
    "creative_performance_ugc": (0.95, [
        r"\bcreative\s+strategist\b", r"\bcreative\s+strategy\b",
        r"\bperformance\s+creative(s)?\b", r"\bpaid\s+social\s+creative\b",
        r"\bcreative\s+testing\b", r"\bcreative\s+iteration\b",

        r"\bugc\b|\buser\s+generated\s+content\b",
        r"\bvideo\s+editor\b", r"\bvideo\s+producer\b", r"\bcontent\s+producer\b",
        r"\bmotion\s+designer\b|\bmotion\s+graphics\b",
        r"\bshort[-\s]?form\b", r"\breels?\b", r"\btiktok\b",

        r"\bcreative\s+director\b", r"\bart\s+director\b",
        r"\bcopywriter\b", r"\bscriptwriter\b|\bscript\s+writer\b",
        r"\bbrand\s+designer\b|\bgraphic\s+designer\b"
    ]),

    # ===== Strong Adjacent: Analytics / Attribution / MarTech (0.9) =====
    "analytics_attribution_ops": (0.9, [
        r"\bmarketing\s+ops\b|\bmarketing\s+operations\b", r"\bmartech\b",
        r"\battribution\b", r"\bmeasurement\b", r"\btracking\b",
        r"\bga4\b|\bgoogle\s+analytics\b", r"\bgtm\b|\btag\s+manager\b",
        r"\bpixel(s)?\b", r"\bconversion\s+tracking\b", r"\bevent\s+tracking\b",
        r"\blook(er)?\s+studio\b|\bdata\s+studio\b",
        r"\bperformance\s+analyst\b", r"\bmarketing\s+analyst\b", r"\bgrowth\s+analyst\b",
        r"\bcrm\b", r"\bhubspot\b", r"\bsalesforce\b"
    ]),

    # ===== Adjacent: Brand / Content / Social (0.8) =====
    "brand_content_social": (0.8, [
        r"\bbrand\s+manager\b", r"\bbrand\s+marketing\b",
        r"\bcontent\s+marketing\b", r"\bcontent\s+manager\b",
        r"\bsocial\s+media\b", r"\bsocial\s+media\s+manager\b",
        r"\bcommunity\s+manager\b", r"\bcommunications?\b", r"\bpr\b"
    ]),

    # ===== Lower Priority (still useful) =====
    "client_services": (0.7, [
        r"\baccount\s+manager\b", r"\baccount\s+director\b",
        r"\bclient\s+services?\b", r"\bclient\s+partner\b",
        r"\bcustomer\s+success\b", r"\bclient\s+success\b"
    ]),

    "sales_roles": (0.55, [
        r"\bsdr\b", r"\bbdr\b", r"\bsales\s+development\b",
        r"\baccount\s+executive\b", r"\bae\b",
        r"\bsales\s+manager\b", r"\bhead[-\s]?of\s+sales\b",
        r"\bbusiness\s+development\b", r"\bbdm\b"
    ]),

    # ===== Peripheral / Noise =====
    "hr_recruiting": (0.2, [
        r"\bhr\b", r"\bhuman\s+resources?\b",
        r"\brecruit(ing|er)\b", r"\btalent\s+acquisition\b", r"\bpeople\s+ops\b"
    ]),
    "finance_admin": (0.2, [
        r"\bfinance\b", r"\baccounts?\b", r"\bbookkeeper\b",
        r"\bpayroll\b", r"\baccountant\b"
    ]),
    "legal_compliance": (0.15, [
        r"\blegal\b", r"\bcompliance\b", r"\blegal\s+counsel\b"
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
        if pd.notna(v):
            parts.append(str(v))
    return " ".join(parts).strip()

def _normalize_apostrophes_and_spaces(s: str) -> str:
    # Normalize common curly apostrophes and other unicode variants to ASCII apostrophe,
    # collapse multiple spaces and normalize newlines to spaces.
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("\u200b", "")  # zero-width
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def compile_patterns():
    def comp_list(patterns):  # compile list of patterns
        return [re.compile(p, flags=re.I) for p in patterns]

    compiled = {
        "negations": comp_list(NEGATIONS),
        "strong": comp_list(HIRING_SIGNALS_STRONG),
        "moderate": comp_list(HIRING_SIGNALS_MODERATE),
        "hashtags": comp_list(HIRING_HASHTAGS),
        "joblinks": comp_list(JOB_LINK_DOMAINS),
        "seniority": [(re.compile(p, re.I), w) for p, w in SENIORITY_WEIGHTS.items()],
        # roles: keep compiled regex list and role weight
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

# NOTE: this function returns match spans using the raw_text (not lowercased),
#       so we can extract accurate character offsets for context extraction.
def find_all_roles(text: str, compiled_roles: dict, raw_text: str = None) -> list[tuple[str, float, str, int, int]]:
    """
    Returns list of tuples: (role_key, weight, matched_string, start_idx, end_idx)
    """
    matches = []
    source = raw_text if raw_text is not None else text
    if not source:
        return matches
    for role_key, (w, regs) in compiled_roles.items():
        for r in regs:
            m = r.search(source)
            if m:
                matches.append((role_key, w, m.group(0), m.start(), m.end()))
                break  # one match per role_key is enough to mark presence
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
        return 1.0  # unknown; do not decay
    delta = (anchor_ts - ts).days
    if delta < 0:
        delta = 0
    # classic half-life decay: weight = 2^(-days / half_life)
    return 2.0 ** (-(delta / max(half_life_days, 1e-6)))

# ----------------------------
# Context extraction helper
# ----------------------------
_sentence_split_re = re.compile(r'(?<=[\.\?\!])\s+')
_word_re = re.compile(r"\b[\w\'\-\&\.]+\b")  # basic word tokenizer (keeps contractions & hyphens)

def extract_context_for_span(raw_text: str, span_start: int, span_end: int) -> str:
    """
    Given the raw_text and a character span of a matched role (span_start, span_end),
    extract context according to user's rules.
    """
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

def detect_hiring_and_score(row, text_cols_present, compiled, ts_col, src_col, anchor_ts, half_life_days):
    raw_text = combine_text_fields(row, text_cols_present)
    if not raw_text or str(raw_text).strip() == "":
        return None

    # Light normalization to avoid missing variations like curly apostrophes
    raw_text_norm = _normalize_apostrophes_and_spaces(raw_text)
    text = raw_text_norm.lower()

    # Quick skip: empty text after normalization
    if not text or text.strip() == "":
        return None

    # Negation check first (use normalized lowercase text)
    if has_any(compiled["negations"], text):
        return None

    # Signals (note: strong signals now count even without a role/joblink)
    strong = has_any(compiled["strong"], text)   # e.g. "we're hiring", "now hiring", "vacancy", etc.
    moderate = has_any(compiled["moderate"], text)
    hashtag = has_any(compiled["hashtags"], text)
    joblink = has_any(compiled["joblinks"], text)

    # Roles detection uses raw_text (to preserve spans & offsets)
    roles_found = find_all_roles(text, compiled["roles"], raw_text=raw_text_norm)
    has_role = len(roles_found) > 0

    # APPLY RULES:
    # - All posts that consist of hiring signals must be considered as a hit and recorded.
    #   -> Strong signals (explicit hiring language) => always hit (no role required)
    # - Moderate signals require role or joblink to be a hit (to avoid noisy "growing" uses)
    # - Hashtag-only posts should still be considered hits and recorded (scored as 'hashtag' base)
    # - Posts with role-only and no hiring signal MUST NOT be considered hits
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
        # Hashtag signals count as hiring signals per your rule — even without roles
        passes = True
        signal_strength = "hashtag"
        base = BASE_SCORE_HASHTAG_ONLY

    if not passes:
        # Not a hiring-related post under our rules
        return None

    # Relevance multiplier = max role weight; fallback if no role matched
    if has_role:
        role_weight_max = max(w for _, w, _, _, _ in roles_found)
    else:
        # If no explicit role matched, treat as 'general hiring' with conservative weight
        # Joblink present -> slightly higher confidence; otherwise general lower weight
        role_weight_max = 0.7 if joblink else 0.6

    relevant_count = sum(1 for _, w, _, _, _ in roles_found if w >= 0.9)
    adjacent_count = sum(1 for _, w, _, _, _ in roles_found if 0.7 <= w < 0.9)
    irrelevant_count = sum(1 for _, w, _, _, _ in roles_found if w < 0.7)

    # Seniority multiplier
    senior_mul = seniority_multiplier(text, compiled["seniority"])

    # Source normalization & weight
    src = normalize_source(row.get(src_col)) if src_col else "unknown"
    src_w = source_weight(src)

    # Job link bonus & multi-role multiplier
    link_mul = JOB_LINK_BONUS if joblink else 1.0
    multi_role_mul = min(1.0 + MULTI_ROLE_STEP * max(len(roles_found) - 1, 0), MULTI_ROLE_CAP)

    # Recency decay
    ts = row.get(ts_col) if ts_col else pd.NaT
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    decay = recency_decay(anchor_ts, ts, half_life_days)

    # Final score composition
    score = base
    score *= role_weight_max
    score *= senior_mul
    score *= src_w
    score *= link_mul
    score *= multi_role_mul
    score *= decay

    # For snippets: categories = comma roles; snippet = first 320 chars
    # roles_found items are (role_key, weight, matched_string, start, end)
    if roles_found:
        categories = ", ".join(sorted({rk for rk, _, _, _, _ in roles_found}))
    else:
        # If there are no role matches, but we have a job link or signal -> classify as general hiring
        categories = "general"

    snippet = raw_text.replace("\n", " ").strip()[:320]

    # Build matched role strings and contexts
    matched_role_strings = []
    matched_contexts = []
    for role_key, w, matched_str, s0, s1 in roles_found:
        ms = matched_str.strip()
        if ms:
            matched_role_strings.append(ms)
            ctx = extract_context_for_span(raw_text, s0, s1)
            if ctx:
                matched_contexts.append(ctx)

    # unique preserve order (stable)
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

    # ----------------------------
    # Robust company_id retrieval
    # ----------------------------
    company_id_val = None
    for k in ("company_id", "_company_id", "companyId", "company_id_clean", "employerCompanyId"):
        try:
            v = row.get(k)
        except Exception:
            v = None
        if pd.notna(v):
            company_id_val = v
            break
    if company_id_val is None:
        for col in row.index:
            if col.lower() in ("companyid", "company_id", "company_id_clean", "employercompanyid"):
                v = row.get(col)
                if pd.notna(v):
                    company_id_val = v
                    break

    return {
        "company_id": company_id_val,
        "_ts": ts,
        "_source": src,
        "signal_strength": signal_strength,
        "job_link": int(joblink),
        "hiring_post": 1,
        "hiring_score": float(score),
        "relevant_roles_count": int(relevant_count),
        "adjacent_roles_count": int(adjacent_count),
        "irrelevant_roles_count": int(irrelevant_count),
        # keep list for downstream aggregation
        "matched_roles": [m for m in matched_role_strings],
        # CSV-friendly aggregated strings
        "matched_roles_str": ", ".join(matched_role_strings) if matched_role_strings else "",
        "matched_roles_contexts": ", ".join(matched_contexts) if matched_contexts else "",
        "categories": categories,
        "snippet": snippet
    }


# ================================
# 3) Aggregation
# ================================

def aggregate_company_level(hit_df: pd.DataFrame, anchor_ts: pd.Timestamp, windows: dict[str, timedelta]) -> pd.DataFrame:
    """
    Build per-company aggregates for each window + freshness.
    Also aggregates matched role lists & contexts across all hits (all time) into columns:
      - matched_roles (comma-separated unique roles across all hits)
      - matched_roles_contexts (comma-separated unique contexts across all hits)
    """
    # Ensure hit_df has 'company_id' column (caller should ensure this, but defensive here)
    if "company_id" not in hit_df.columns and "_company_id" in hit_df.columns:
        hit_df = hit_df.copy()
        hit_df["company_id"] = hit_df["_company_id"]

    companies = pd.DataFrame({"company_id": hit_df["company_id"].unique()}) if not hit_df.empty else pd.DataFrame(columns=["company_id"])
    out = companies.copy()

    # For each window build aggregates
    for label, delta in windows.items():
        cutoff = anchor_ts - delta
        wmask = hit_df["_ts"].notna() & (hit_df["_ts"] >= cutoff)
        wdf = hit_df[wmask].copy()

        # Ensure matched_roles_str and matched_roles_contexts columns exist to operate on
        if "matched_roles_str" not in wdf.columns:
            wdf["matched_roles_str"] = ""
        if "matched_roles_contexts" not in wdf.columns:
            wdf["matched_roles_contexts"] = ""

        if wdf.empty:
            agg = pd.DataFrame({"company_id": out["company_id"]})
            agg[f"hiring_posts_{label}"] = 0
            agg[f"total_hiring_score_{label}"] = 0.0
            agg[f"relevant_hiring_count_{label}"] = 0
            agg[f"adjacent_hiring_count_{label}"] = 0
            agg[f"irrelevant_hiring_count_{label}"] = 0
            agg[f"hiring_intensity_{label}"] = 0.0
            agg[f"unique_relevant_roles_{label}"] = 0
            agg[f"hiring_velocity_{label}"] = 0.0
            # New columns: matched role listing/context (empty)
            agg[f"matched_roles_{label}"] = ""
            agg[f"matched_roles_contexts_{label}"] = ""
            # Merge with out to keep consistent index
            out = out.merge(agg, on="company_id", how="left")
            continue

        # explode role matches to compute unique counts by company
        wdf_roles = wdf.assign(role_tokens=wdf["categories"].str.split(r"\s*,\s*"))
        wdf_roles = wdf_roles.explode("role_tokens")
        wdf_roles["role_tokens"] = wdf_roles["role_tokens"].fillna("")

        # compute per-company aggregates
        base_agg = (wdf
                    .groupby("company_id", as_index=False)
                    .agg(**{
                        f"hiring_posts_{label}": ("hiring_post", "sum"),
                        f"total_hiring_score_{label}": ("hiring_score", "sum"),
                        f"relevant_hiring_count_{label}": ("relevant_roles_count", "sum"),
                        f"adjacent_hiring_count_{label}": ("adjacent_roles_count", "sum"),
                        f"irrelevant_hiring_count_{label}": ("irrelevant_roles_count", "sum"),
                    }))

        if f"hiring_posts_{label}" in base_agg.columns:
            base_agg[f"hiring_intensity_{label}"] = np.where(
                base_agg[f"hiring_posts_{label}"] > 0,
                base_agg[f"total_hiring_score_{label}"] / base_agg[f"hiring_posts_{label}"],
                0.0
            )
            # posts per week over the window
            days = max(int(delta.days), 1)
            base_agg[f"hiring_velocity_{label}"] = base_agg[f"hiring_posts_{label}"] / (days / 7.0)

        # unique relevant roles in window (categories are role keys; count those considered relevant+adjacent)
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
                .agg(**{f"unique_relevant_roles_{label}": ("role_tokens", lambda s: len(set([x for x in s if x])))} )
                )

        # Aggregate matched role strings and contexts (string fields on hits)
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
            "matched_roles_str": f"matched_roles_{label}",
            "matched_roles_contexts": f"matched_roles_contexts_{label}"
        })

        # Merge base_agg + uniq + roles_agg
        agg = base_agg.merge(uniq, on="company_id", how="left")
        agg = agg.merge(roles_agg, on="company_id", how="left")

        # fill missing matched columns with empty string
        agg = agg.fillna({f"matched_roles_{label}": "", f"matched_roles_contexts_{label}": ""})

        # Ensure unique_relevant_roles column filled
        agg = agg.fillna({f"unique_relevant_roles_{label}": 0})

        # For any remaining NaNs in numeric cols fill zero
        for c in agg.columns:
            if c == "company_id":
                continue
            if pd.api.types.is_numeric_dtype(agg[c].dtype):
                agg[c] = agg[c].fillna(0)

        out = out.merge(agg, on="company_id", how="left")

    # Freshness
    if not hit_df.empty:
        last = (hit_df[hit_df["_ts"].notna()]
                .sort_values("_ts")
                .groupby("company_id", as_index=False)
                .agg(last_hiring_ts=("_ts", "max")))
        last["days_since_last_hiring"] = (anchor_ts - last["last_hiring_ts"]).dt.days
        out = out.merge(last, on="company_id", how="left")
    else:
        out["last_hiring_ts"] = pd.NaT
        out["days_since_last_hiring"] = np.nan

    # === NEW: aggregate matched_roles & contexts across ALL hits (all time) into features ===
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
                     .agg(matched_roles_all=("matched_roles", combine_role_lists),
                          matched_roles_contexts_all=("matched_roles_contexts", combine_contexts))
                     )

        out = out.merge(roles_all, on="company_id", how="left")
    else:
        out["matched_roles_all"] = ""
        out["matched_roles_contexts_all"] = ""

    # Fill numeric NaNs with 0 for stability
    for c in out.columns:
        if c == "company_id":
            continue
        if pd.api.types.is_numeric_dtype(out[c].dtype):
            out[c] = out[c].fillna(0)
    # For string columns that may be NaN, replace with empty string for stability
    for c in out.columns:
        if c == "company_id":
            continue
        if out[c].dtype == object:
            out[c] = out[c].fillna("")
    return out


# ================================
# 4) Main
# ================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="company_activity_clean.csv", help="Company-level activity cleaned file")
    ap.add_argument("--out_features", default="hiring_features.csv", help="Per-company hiring features")
    ap.add_argument("--out_snippets", default="hiring_snippets.csv", help="Per-post hiring snippets (QA)")
    ap.add_argument("--half_life_days", type=float, default=45.0, help="Recency half-life in days")
    ap.add_argument("--windows", default="7,30,90", help="Comma-separated day windows for aggregation")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, dtype={"company_id": str})
    n0 = len(df)
    print(f"✅ Loaded company activity: {args.input} ({n0:,} rows, {len(df.columns)} cols)")

    # Identify columns
    company_col = pick_first_existing(df, COMPANY_ID_COLS)
    if not company_col:
        raise ValueError(f"Could not find company id column. Tried: {COMPANY_ID_COLS}")

    ts_col = pick_first_existing(df, TIMESTAMP_COLS)
    text_cols_present = [c for c in TEXT_COLS if c in df.columns]
    src_col = pick_first_existing(df, SOURCE_COLS)

    if not text_cols_present:
        raise ValueError(f"No text/content columns found. Tried: {TEXT_COLS}")

    # Normalize basic working columns
    work = df.copy()
    work["_company_id"] = work[company_col]
    work["_ts"] = coerce_utc(work[ts_col]) if ts_col else pd.NaT
    if src_col:
        work["_src_norm"] = work[src_col].map(normalize_source)
    else:
        work["_src_norm"] = "unknown"

    # Determine anchor time (max ts or now)
    anchor_ts = detect_anchor_ts([work["_ts"]])
    print(f"Anchor time (max ts or now): {anchor_ts}")

    # Compile patterns once
    compiled = compile_patterns()

    # Detect row-level hiring and score
    rows = []
    for _, row in work.iterrows():
        res = detect_hiring_and_score(
            row=row,
            text_cols_present=text_cols_present,
            compiled=compiled,
            ts_col="_ts",
            src_col="_src_norm",
            anchor_ts=anchor_ts,
            half_life_days=args.half_life_days
        )
        if res is not None:
            rows.append(res)

    if not rows:
        print("⚠️ No hiring posts detected. Writing empty outputs.")
        # Empty features skeleton
        outf = pd.DataFrame(columns=["company_id"])
        outf.to_csv(args.out_features, index=False)
        pd.DataFrame(columns=["company_id", "ts", "source", "categories", "snippet", "matched_roles_str", "matched_roles_contexts"]).to_csv(args.out_snippets, index=False)
        print(f"✅ Saved empty hiring features to: {args.out_features}")
        print(f"✅ Saved empty hiring snippets to: {args.out_snippets}")
        return

    hits = pd.DataFrame(rows)
    print(f"✅ Detected hiring posts: {len(hits):,}")

    # ----------------------------
    # Defensive: ensure company_id is present and sane
    # ----------------------------
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

    # Aggregate
    features = aggregate_company_level(hits, anchor_ts, win_map)

    # Order columns for readability (7d -> 30d -> 90d order if present)
    ordered_cols = ["company_id", "last_hiring_ts", "days_since_last_hiring"]
    for lbl in sorted(win_map.keys(), key=lambda x: int(x.replace("d", ""))):
        ordered_cols += [
            f"hiring_posts_{lbl}",
            f"total_hiring_score_{lbl}",
            f"hiring_intensity_{lbl}",
            f"hiring_velocity_{lbl}",
            f"relevant_hiring_count_{lbl}",
            f"adjacent_hiring_count_{lbl}",
            f"irrelevant_hiring_count_{lbl}",
            f"unique_relevant_roles_{lbl}",
            f"matched_roles_{lbl}",
            f"matched_roles_contexts_{lbl}",
        ]
    ordered_cols = [c for c in ordered_cols if c in features.columns]
    features = features[ordered_cols + [c for c in features.columns if c not in ordered_cols]]

    # Write features
    features.to_csv(args.out_features, index=False)

    # Write snippets (STRICT schema requested)
    for k in ["matched_roles_str", "matched_roles_contexts"]:
        if k not in hits.columns:
            hits[k] = ""
    snippets = hits.rename(columns={
        "postTimestamp": "ts",
        "action": "source"
    })[["company_id", "_ts", "_source", "categories", "snippet", "matched_roles_str", "matched_roles_contexts"]].copy()
    snippets = snippets.rename(columns={"_ts": "ts", "_source": "source"})
    snippets.to_csv(args.out_snippets, index=False)

    print(f"✅ Saved per-company hiring features to: {args.out_features}")
    print(f"✅ Saved hiring QA snippets to: {args.out_snippets}")

    # Console top sample by 30d if present, else max window
    pref = None
    for p in ["30d", "90d", "7d"]:
        if f"total_hiring_score_{p}" in features.columns:
            pref = p
            break
    if pref:
        top = features.sort_values(f"total_hiring_score_{pref}", ascending=False).head(5)
        with pd.option_context("display.max_columns", 120, "display.width", 220):
            print(f"\nTop 5 companies by total_hiring_score_{pref}:")
            print(top[["company_id", f"total_hiring_score_{pref}", f"hiring_posts_{pref}", f"hiring_intensity_{pref}", f"unique_relevant_roles_{pref}"]].to_string(index=False))


if __name__ == "__main__":
    main()
