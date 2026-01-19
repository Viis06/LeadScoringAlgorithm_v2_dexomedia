import pandas as pd

# === File paths (update if needed) ===
ACCOUNT_SCRAPER_FILE = "TMP - account_scraper.csv - account_scraper.csv.csv"
COMPANY_ACTIVITY_FILE = "tmp_company_activity - company_activity.csv"
OUTPUT_FILE = "company_activity_enriched.csv"

def normalize_url(url: str) -> str:
    """Normalize LinkedIn URLs so they can be joined reliably."""
    if pd.isna(url):
        return None
    url = url.strip().lower()
    if url.endswith("/"):
        url = url[:-1]
    return url

def main():
    # Load both datasets
    account_df = pd.read_csv(ACCOUNT_SCRAPER_FILE, dtype=str)
    activity_df = pd.read_csv(COMPANY_ACTIVITY_FILE, dtype=str)

    # Normalize URLs for join
    account_df["linkedInCompanyUrl_norm"] = account_df["linkedInCompanyUrl"].apply(normalize_url)
    activity_df["profileUrl_norm"] = activity_df["profileUrl"].apply(normalize_url)

    # Select only needed columns for joining
    account_lookup = account_df[["linkedInCompanyUrl_norm", "companyId"]].drop_duplicates()

    # Left join activity_df with account_df on normalized URLs
    enriched_df = activity_df.merge(
        account_lookup,
        how="left",
        left_on="profileUrl_norm",
        right_on="linkedInCompanyUrl_norm"
    )

    # Drop helper columns
    enriched_df.drop(columns=["profileUrl_norm", "linkedInCompanyUrl_norm"], inplace=True)

    # Save enriched file
    enriched_df.to_csv(OUTPUT_FILE, index=False)

    # Report stats
    total = len(enriched_df)
    matched = enriched_df["companyId"].notna().sum()
    print(f"✅ Enrichment complete: {matched}/{total} company activities matched with companyId")
    print(f"📂 Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
