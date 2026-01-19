#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2b — Activity Aggregation (Leads + Companies)

Inputs (produced by s2aSplitActivity.py):
  split_activity/
    leads/
      post.csv, repost.csv, comment_own.csv, comment_others.csv, like.csv, comment_reaction.csv (uncategorized.csv ignored)
    companies/
      post.csv, repost.csv, comment_own.csv, comment_others.csv, like.csv, comment_reaction.csv (uncategorized.csv ignored)

Output:
  activity_summary_leads.csv       (per person_id)
  activity_summary_companies.csv   (per company_id)

Notes:
- Recency & windows are computed from postTimestamp (true activity time), never from 'timestamp' (scrape time).
- Anchor time is the MAX observed postTimestamp in the dataset (fallback: current UTC). This makes windows stable even if
  data is from the past or future relative to wall clock.
- Robust to missing columns (likes/comments/reposts/views may be missing in some sources).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

# ----------------------------
# Config
# ----------------------------
LEADS_DIR = Path("split_activity/leads")
COMPS_DIR = Path("split_activity/companies")

OUT_LEADS = Path("activity_summary_leads.csv")
OUT_COMPS = Path("activity_summary_companies.csv")

# Which category CSVs to read
CATEGORY_FILES = {
    "post.csv": "post",
    "repost.csv": "repost",
    "comment_own.csv": "comment_own",
    "comment_others.csv": "comment_others",
    "like.csv": "like",
    "comment_reaction.csv": "comment_reaction",
    # "uncategorized.csv": "uncategorized",  # intentionally ignored
}

# What we treat as "active" vs "passive" actions
ACTIVE_CATS = {"post", "repost", "comment_own", "comment_others"}
PASSIVE_CATS = {"like", "comment_reaction"}

# Time windows (days)
WINDOWS = [7, 30, 90]

# Half-lives (days) for exponential decay recency scores
HALF_LIVES = [7, 30]


# ----------------------------
# Small utilities
# ----------------------------
def normalize_url(u: str) -> str:
    """Normalize LinkedIn URLs so joins/grouping are stable."""
    if pd.isna(u):
        return ""
    s = str(u).strip().lower()
    if not s:
        return ""
    try:
        p = urlparse(s if "://" in s else "https://" + s)
        base = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
        base = base.replace("://www.", "://")
        return base
    except Exception:
        return s.rstrip("/")


def to_num(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def within_days(ts: pd.Timestamp, ref: pd.Timestamp, days: int) -> bool:
    if pd.isna(ts):
        return False
    delta = ref - ts
    return (delta.days >= 0) and (delta.days < days)


def iso_week_key(ts: pd.Timestamp) -> str | None:
    if pd.isna(ts):
        return None
    iso = ts.isocalendar()  # (year, week, weekday)
    return f"{int(iso[0])}-W{int(iso[1])}"


def exp_decay_score(ages_days: np.ndarray, half_life: float) -> float:
    # weight = 0.5 ** (age_days / half_life)
    return float(np.sum(np.power(0.5, ages_days / half_life)))


# ----------------------------
# Load helpers
# ----------------------------
def read_category_dir(base_dir: Path, id_col: str) -> pd.DataFrame:
    """
    Read all category files under base_dir into a single dataframe with:
      - {id_col} present and normalized if URL-like
      - 'category' column set
      - 'event_time' from postTimestamp (UTC)
      - numeric engagement columns coerced
    """
    frames = []
    for fname, cat in CATEGORY_FILES.items():
        p = base_dir / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"⚠️ Could not read {p}: {e}")
            continue

        # Ensure id column exists (prefer what's there; otherwise try common fallbacks)
        if id_col not in df.columns:
            if id_col == "person_id":
                for alt in ["person_id", "profileUrl", "authorUrl", "linkedin_profile_url"]:
                    if alt in df.columns:
                        df[id_col] = df[alt]
                        break
            elif id_col == "company_id":
                for alt in ["company_id", "companyUrl", "authorUrl", "author"]:
                    if alt in df.columns:
                        df[id_col] = df[alt]
                        break

        # Drop if still missing
        if id_col not in df.columns:
            print(f"⚠️ Skipping {len(df)} rows in {p.name} because '{id_col}' is missing")
            continue

        # Normalize IDs if they look URL-like
        if df[id_col].astype(str).str.contains("linkedin", case=False, na=False).any():
            df[id_col] = df[id_col].astype(str).map(normalize_url)
        else:
            # still trim and lower for stability
            df[id_col] = df[id_col].astype(str).str.strip()

        # Use TRUE activity time only
        if "postTimestamp" not in df.columns:
            df["postTimestamp"] = pd.NaT
        df["event_time"] = pd.to_datetime(df["postTimestamp"], utc=True, errors="coerce")

        # Numeric engagements
        to_num(df, ["likeCount", "commentCount", "repostCount", "viewCount"])

        # Stamp category
        df["category"] = cat

        # Keep useful cols
        keep = [
            id_col, "category", "event_time",
            "postUrl", "postContent",
            "commentUrl", "commentContent",
            "likeCount", "commentCount", "repostCount", "viewCount"
        ]
        present = [c for c in keep if c in df.columns]
        frames.append(df[present])

    if not frames:
        return pd.DataFrame(columns=[id_col, "category", "event_time"])

    out = pd.concat(frames, ignore_index=True)
    # Remove rows with no id or no event_time (we cannot aggregate recency without a time)
    out = out.dropna(subset=[id_col, "event_time"])
    return out


def choose_anchor_ts(all_events: pd.DataFrame) -> pd.Timestamp:
    """Use the latest observed activity time as the anchor; fallback to UTC now."""
    latest = pd.to_datetime(all_events["event_time"], utc=True, errors="coerce").max()
    if pd.isna(latest):
        return pd.Timestamp.utcnow()
    return latest


# ----------------------------
# Aggregation core
# ----------------------------
def aggregate_side(all_events: pd.DataFrame, id_col: str, out_path: Path):
    if all_events.empty:
        print(f"⚠️ No events to aggregate for {id_col}. Skipping save.")
        return

    # Anchor time for windows/recency
    NOW = choose_anchor_ts(all_events)

    # Precompute helpers
    all_events["is_active"] = all_events["category"].isin(ACTIVE_CATS).astype(int)
    all_events["is_passive"] = all_events["category"].isin(PASSIVE_CATS).astype(int)

    # Fast overall counts per {id_col} via pivot on category
    cat_counts = (
        all_events
        .pivot_table(index=id_col, columns="category", values="event_time", aggfunc="count", fill_value=0)
        .rename_axis(None, axis=1)
    )
    # Ensure all expected categories exist
    for c in ["post", "repost", "comment_own", "comment_others", "like", "comment_reaction"]:
        if c not in cat_counts.columns:
            cat_counts[c] = 0

    # Last action ts
    last_ts = all_events.groupby(id_col)["event_time"].max().rename("last_action_ts")

    # Base numeric frame
    base = pd.DataFrame(index=cat_counts.index).assign(
        n_actions_total=cat_counts.sum(axis=1),
        n_posts=cat_counts["post"],
        n_reposts=cat_counts["repost"],
        n_comment_own=cat_counts["comment_own"],
        n_comment_others=cat_counts["comment_others"],
        n_likes=cat_counts["like"],
        n_comment_reactions=cat_counts["comment_reaction"],
    ).merge(last_ts, left_index=True, right_index=True, how="left")

    base["days_since_last_action"] = (NOW - base["last_action_ts"]).dt.days

    # Active/passive counts
    ap = all_events.groupby(id_col)[["is_active", "is_passive"]].sum().rename(
        columns={"is_active": "active_actions", "is_passive": "passive_actions"}
    )
    base = base.merge(ap, left_index=True, right_index=True, how="left")
    base["active_share"] = (base["active_actions"] / base["n_actions_total"].replace(0, np.nan)).fillna(0)

    # Rolling windows
    for d in WINDOWS:
        sub = all_events.loc[all_events["event_time"].apply(lambda t: within_days(t, NOW, d))]
        if sub.empty:
            # If no data in window, just zero-fill columns
            for key in ["posts", "reposts", "comment_own", "comment_others", "likes", "comment_reaction", "actions"]:
                base[f"{key}_{d}d"] = 0
            continue

        w_counts = (
            sub.pivot_table(index=id_col, columns="category", values="event_time", aggfunc="count", fill_value=0)
            .rename_axis(None, axis=1)
        )
        for c in ["post", "repost", "comment_own", "comment_others", "like", "comment_reaction"]:
            if c not in w_counts.columns:
                w_counts[c] = 0

        # Merge windowed counts
        base = base.merge(
            w_counts.rename(columns={
                "post": f"posts_{d}d",
                "repost": f"reposts_{d}d",
                "comment_own": f"comment_own_{d}d",
                "comment_others": f"comment_others_{d}d",
                "like": f"likes_{d}d",
                "comment_reaction": f"comment_reaction_{d}d",
            }),
            left_index=True, right_index=True, how="left"
        )
        base[f"actions_{d}d"] = (
            base.get(f"posts_{d}d", 0)
            + base.get(f"reposts_{d}d", 0)
            + base.get(f"comment_own_{d}d", 0)
            + base.get(f"comment_others_{d}d", 0)
            + base.get(f"likes_{d}d", 0)
            + base.get(f"comment_reaction_{d}d", 0)
        )
        # Fill possible NaNs from merge with 0
        for col in [c for c in base.columns if c.endswith(f"_{d}d")]:
            base[col] = base[col].fillna(0).astype(int)

    # Velocity & consistency
    # actions/week over last 90d
    base["actions_per_week_90d"] = base["actions_90d"] / (90.0 / 7.0)

    # weeks active in last 8 weeks (56 days)
    sub56 = all_events.loc[all_events["event_time"].apply(lambda t: within_days(t, NOW, 56))].copy()
    if not sub56.empty:
        sub56["iso_week"] = sub56["event_time"].apply(iso_week_key)
        weeks_active = (
            sub56.dropna(subset=["iso_week"])
            .groupby(id_col)["iso_week"].nunique()
            .rename("weeks_active_8")
        )
        base = base.merge(weeks_active, left_index=True, right_index=True, how="left")
    base["weeks_active_8"] = base["weeks_active_8"].fillna(0).astype(int)

    # Exponential decay recency scores
    # Compute per-id ages (in days) and sum weights for each half-life
    def ages_days(s: pd.Series) -> np.ndarray:
        return (NOW - s).dt.total_seconds().values / 86400.0

    grouped_times = all_events.groupby(id_col)["event_time"].apply(list)
    for hl in HALF_LIVES:
        scores = {}
        for k, times in grouped_times.items():
            if not times:
                scores[k] = 0.0
                continue
            arr = pd.to_datetime(pd.Series(times), utc=True, errors="coerce").dropna()
            if arr.empty:
                scores[k] = 0.0
            else:
                scores[k] = exp_decay_score(ages_days(arr), hl)
        base[f"recency_decay_hl{hl}d"] = pd.Series(scores)

    # Active-only decay (signals "doing" over "lurking")
    grouped_active = (
        all_events.loc[all_events["category"].isin(ACTIVE_CATS)]
        .groupby(id_col)["event_time"]
        .apply(list)
    )
    for hl in HALF_LIVES:
        scores = {}
        for k, times in grouped_active.items():
            if not times:
                scores[k] = 0.0
                continue
            arr = pd.to_datetime(pd.Series(times), utc=True, errors="coerce").dropna()
            scores[k] = exp_decay_score(ages_days(arr), hl) if not arr.empty else 0.0
        base[f"recency_decay_active_hl{hl}d"] = pd.Series(scores).reindex(base.index).fillna(0.0)

    # Burstiness over last 90d (std/mean of daily counts, clipped)
    sub90 = all_events.loc[all_events["event_time"].apply(lambda t: within_days(t, NOW, 90))].copy()
    if not sub90.empty:
        # floor to date
        sub90["date"] = sub90["event_time"].dt.tz_convert("UTC").dt.floor("D")
        daily = sub90.groupby([id_col, "date"]).size().rename("n").reset_index()
        # Build complete date index per id to avoid bias
        min_date = (NOW - timedelta(days=89)).floor("D")
        all_days = pd.date_range(min_date, NOW.floor("D"), freq="D", tz="UTC")
        bursties = {}
        for k, g in daily.groupby(id_col):
            gk = g.set_index("date").reindex(all_days, fill_value=0)
            mu = gk["n"].mean()
            sd = gk["n"].std(ddof=0)
            burst = 0.0 if mu == 0 else float(sd / max(mu, 1e-9))
            # cap to a reasonable range to avoid extreme noise
            bursties[k] = min(burst, 10.0)
        base["burstiness_90d"] = pd.Series(bursties).reindex(base.index).fillna(0.0)
    else:
        base["burstiness_90d"] = 0.0

    # Post engagement quality (overall + 30d/90d)
    def add_post_engagement_features(suffix: str, frame: pd.DataFrame):
        if frame.empty:
            base[f"post_avg_likes{suffix}"] = np.nan
            base[f"post_avg_comments{suffix}"] = np.nan
            base[f"post_avg_reposts{suffix}"] = np.nan
            base[f"post_med_likes{suffix}"] = np.nan
            base[f"post_med_comments{suffix}"] = np.nan
            base[f"post_med_reposts{suffix}"] = np.nan
            base[f"post_eng_rate{suffix}"] = np.nan
            return

        posts = frame.loc[frame["category"] == "post"].copy()
        if posts.empty:
            reidx = base.index
            for col in [
                f"post_avg_likes{suffix}", f"post_avg_comments{suffix}", f"post_avg_reposts{suffix}",
                f"post_med_likes{suffix}", f"post_med_comments{suffix}", f"post_med_reposts{suffix}",
                f"post_eng_rate{suffix}"
            ]:
                base[col] = np.nan
            return

        # aggregate per ID
        def agg_fun(g):
            likes = g.get("likeCount")
            coms  = g.get("commentCount")
            reps  = g.get("repostCount")
            views = g.get("viewCount")

            out = {}
            if likes is not None:
                out["post_avg_likes"] = likes.mean(skipna=True)
                out["post_med_likes"] = likes.median(skipna=True)
            if coms is not None:
                out["post_avg_comments"] = coms.mean(skipna=True)
                out["post_med_comments"] = coms.median(skipna=True)
            if reps is not None:
                out["post_avg_reposts"] = reps.mean(skipna=True)
                out["post_med_reposts"] = reps.median(skipna=True)

            # basic engagement rate ( (likes+comments+reposts)/views ), averaged per post
            if (views is not None) and (likes is not None or coms is not None or reps is not None):
                l = likes.fillna(0) if likes is not None else 0
                c = coms.fillna(0) if coms is not None else 0
                r = reps.fillna(0) if reps is not None else 0
                v = views.replace(0, np.nan) if views is not None else np.nan
                eng = (l + c + r) / v
                out["post_eng_rate"] = eng.replace([np.inf, -np.inf], np.nan).mean(skipna=True)
            return pd.Series(out)

        pe = posts.groupby(id_col).apply(agg_fun)

        # Merge with suffix
        for col in ["post_avg_likes", "post_avg_comments", "post_avg_reposts",
                    "post_med_likes", "post_med_comments", "post_med_reposts", "post_eng_rate"]:
            src = pe[col] if col in pe.columns else pd.Series(dtype=float)
            base[f"{col}{suffix}"] = src.reindex(base.index).astype(float)

    # Overall lifetime (all events)
    add_post_engagement_features("", all_events)

    # 30d and 90d windows for post quality
    for d in [30, 90]:
        sub = all_events.loc[all_events["event_time"].apply(lambda t: within_days(t, NOW, d))]
        add_post_engagement_features(f"_{d}d", sub)

    # Order columns nicely (optional)
    # Move id_col to the front
    base.reset_index(names=id_col, inplace=True)

    # Save
    base.to_csv(out_path, index=False)
    n_ids = base.shape[0]
    n_rows = all_events.shape[0]
    print(f"✅ Aggregated {n_rows:,} activity rows across {n_ids:,} unique {id_col}s")
    print(f"Saved features to: {out_path}")
    # quick peek
    peek_cols = [id_col, "n_actions_total", "actions_30d", "actions_7d", "days_since_last_action",
                 "actions_per_week_90d", "weeks_active_8", "recency_decay_hl7d", "burstiness_90d"]
    show = [c for c in peek_cols if c in base.columns]
    if show:
        print("\nSample (top 5 by activity):")
        print(base.sort_values("n_actions_total", ascending=False)[show].head(5).to_string(index=False))


# ----------------------------
# Main
# ----------------------------
def main():
    # Leads
    print("\n--- Processing Leads ---")
    leads_events = read_category_dir(LEADS_DIR, id_col="person_id")
    # collapse any accidental duplicates by keeping event rows as-is (normalization already handled)
    aggregate_side(leads_events, id_col="person_id", out_path=OUT_LEADS)

    # Companies
    print("\n--- Processing Companies ---")
    comps_events = read_category_dir(COMPS_DIR, id_col="company_id")
    aggregate_side(comps_events, id_col="company_id", out_path=OUT_COMPS)


if __name__ == "__main__":
    main()
