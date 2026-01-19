# s2aSplitActivity.py
import pandas as pd
from pathlib import Path

# --- Input files ---
INPUT_LEADS = Path("activity_clean.csv")
INPUT_COMPANIES = Path("company_activity_clean.csv")

# --- Output directories ---
OUTDIR = Path("split_activity")
OUTDIR_LEADS = OUTDIR / "leads"
OUTDIR_COMPANIES = OUTDIR / "companies"

for d in [OUTDIR, OUTDIR_LEADS, OUTDIR_COMPANIES]:
    d.mkdir(exist_ok=True, parents=True)

# --- Shared function: split & save ---
def split_and_save(df, outdir):
    # Normalize once
    df["action_norm"] = df.get("action", "").fillna("").str.strip().str.lower()
    df["type_norm"]   = df.get("type",   "").fillna("").str.strip().str.lower()

    post_like_types = {
        "post", "text", "image", "video (linkedin source)", "document", "poll", "celebration"
    }

    # Masks
    m_comment_reaction = (
        df["action_norm"].str.contains("comment", na=False)
        & df["action_norm"].str.contains(r"\blike|\blikes|\bliked|\bloves|\bsupports|\bcelebrates|\bfinds", regex=True, na=False)
    )
    m_comment_others = df["action_norm"].str.contains(r"\breplied to\b.*comment", regex=True, na=False)
    m_comment_own    = df["action_norm"].str.contains(r"\bcommented on\b", regex=True, na=False)
    m_repost = (
        df["action_norm"].str.contains(r"\breposted\b", regex=True, na=False)
        | (df["action_norm"] == "repost")
    )
    m_like_raw = df["action_norm"].str.contains(
        r"likes this|liked this|loves this|supports this|celebrates this|finds this insightful|finds this funny",
        regex=True, na=False
    )
    m_like = m_like_raw & ~df["action_norm"].str.contains("comment", na=False)
    m_post = (
        (df["action_norm"] == "post")
        | df["action_norm"].str.startswith("post", na=False)
        | (
            (df["action_norm"] == "")
            & (
                df["type_norm"].isin(post_like_types)
                | (df["type_norm"] == "article")
            )
        )
    )

    # Assign categories
    df["category"] = "uncategorized"
    order = [
        ("comment_reaction", m_comment_reaction),
        ("comment_others",   m_comment_others),
        ("comment_own",      m_comment_own),
        ("repost",           m_repost),
        ("like",             m_like),
        ("post",             m_post),
    ]
    for name, mask in order:
        df.loc[(df["category"] == "uncategorized") & mask, "category"] = name

    # Conflict check
    mask_df = pd.DataFrame({name: mask for name, mask in order}, index=df.index)
    conflicts = mask_df.sum(axis=1) > 1
    if conflicts.any():
        print(f"⚠️ Multi-match conflicts found in {outdir.name}: {conflicts.sum()}")
        print(df.loc[conflicts, ["action", "type"]].head(10))
    else:
        print(f"✅ No multi-match conflicts found in {outdir.name}!")

    # Save one CSV per category
    for cat, subset in df.groupby("category", sort=True):
        out_path = outdir / f"{cat}.csv"
        subset.drop(columns=["action_norm", "type_norm"], errors="ignore").to_csv(out_path, index=False)
        print(f"Saved {len(subset)} rows to {out_path}")

    # Summary
    total = len(df)
    per_split = df["category"].value_counts()
    print(f"\nCheck ({outdir.name}): {total} total rows vs {per_split.sum()} across splits")
    print("Counts per category:")
    print(per_split)

    if (df["category"] == "uncategorized").any():
        print("\n🔎 Sample of Uncategorized rows (action | type):")
        print(df.loc[df["category"] == "uncategorized", ["action", "type"]].head(15))


# --- Run for leads ---
if INPUT_LEADS.exists():
    df_leads = pd.read_csv(INPUT_LEADS)
    print("\n--- Processing Leads ---")
    split_and_save(df_leads, OUTDIR_LEADS)
else:
    print("⚠️ Leads input file not found:", INPUT_LEADS)

# --- Run for companies ---
if INPUT_COMPANIES.exists():
    df_companies = pd.read_csv(INPUT_COMPANIES)
    print("\n--- Processing Companies ---")
    split_and_save(df_companies, OUTDIR_COMPANIES)
else:
    print("⚠️ Companies input file not found:", INPUT_COMPANIES)
