import pandas as pd
import re

# ---------- UPDATE THESE WITH YOUR FILES ----------
ACTIVITY_CSV        = "tmp_linkedin_activity - linkedin_activity.csv"
COMPANY_CSV         = "TMP - account_scraper.csv - account_scraper.csv.csv"
COMPANY_ACTIVITY_CSV = "company_activity_enriched.csv"   # New company activity phantom
LEADS_CSV           = "TMP - sales_navigator_leads.csv - Filtered.csv"
# --------------------------------------------------

def normalize_profile_url(url: str) -> str:
    """Make LinkedIn profile URLs comparable (remove www, params, trailing slash)."""
    if pd.isna(url):
        return None
    u = str(url).strip()
    u = u.replace("www.linkedin.com", "linkedin.com")
    u = u.split("?")[0].split("#")[0].rstrip("/")
    return u

def extract_company_id_from_url(url: str) -> str | None:
    """Pull numeric company ID from Sales Navigator company URLs."""
    if pd.isna(url):
        return None
    m = re.search(r"/company/(\d+)", str(url))
    if m:
        return m.group(1)
    return None

# ---------------------------
# Load the four CSVs
# ---------------------------
leads = pd.read_csv(LEADS_CSV)
activity = pd.read_csv(ACTIVITY_CSV)
companies = pd.read_csv(COMPANY_CSV)
company_activity = pd.read_csv(COMPANY_ACTIVITY_CSV)

# ---------------------------
# LEADS: create keys
# ---------------------------
if "defaultProfileUrl" not in leads.columns:
    raise ValueError("Leads file must have 'defaultProfileUrl' column.")

leads["person_id"] = leads["defaultProfileUrl"].apply(normalize_profile_url)

if "companyId" in leads.columns:
    leads["company_id"] = leads["companyId"].astype(str)
else:
    leads["company_id"] = None
    for col in ["salesNavigatorCompanyUrl", "companyUrl", "regularCompanyUrl"]:
        if col in leads.columns:
            leads["company_id"] = leads["company_id"].fillna(
                leads[col].apply(extract_company_id_from_url)
            )

leads["company_id"] = leads["company_id"].astype(str)
leads = leads.dropna(subset=["person_id"]).drop_duplicates("person_id")

# ---------------------------
# ACTIVITY: create keys
# ---------------------------
if "profileUrl" not in activity.columns:
    raise ValueError("Activity file must have 'profileUrl' column.")

activity["person_id"] = activity["profileUrl"].apply(normalize_profile_url)
activity = activity.dropna(subset=["person_id"]).drop_duplicates(
    subset=["person_id", "postUrl"] if "postUrl" in activity.columns else ["person_id"]
)

# ---------------------------
# COMPANIES: create keys
# ---------------------------
if "companyId" in companies.columns:
    companies["company_id"] = companies["companyId"].astype(str)
else:
    for col in ["linkedInCompanyUrl", "salesNavigatorCompanyUrl", "companyUrl"]:
        if col in companies.columns:
            companies["company_id"] = companies.get("company_id", pd.Series([None]*len(companies)))
            companies["company_id"] = companies["company_id"].fillna(
                companies[col].apply(extract_company_id_from_url)
            )
    if "company_id" not in companies.columns:
        raise ValueError("Companies file must have 'companyId' or a company URL column.")

companies = companies.dropna(subset=["company_id"]).drop_duplicates("company_id")

# ---------------------------
# COMPANY ACTIVITY: create keys
# ---------------------------
if "companyId" in company_activity.columns:
    company_activity["company_id"] = company_activity["companyId"].astype(str)
else:
    for col in ["companyUrl", "salesNavigatorCompanyUrl", "linkedInCompanyUrl"]:
        if col in company_activity.columns:
            company_activity["company_id"] = company_activity.get("company_id", pd.Series([None]*len(company_activity)))
            company_activity["company_id"] = company_activity["company_id"].fillna(
                company_activity[col].apply(extract_company_id_from_url)
            )

if "company_id" not in company_activity.columns:
    raise ValueError("Company activity file must have 'companyId' or a company URL column.")

company_activity = company_activity.dropna(subset=["company_id"]).drop_duplicates(
    subset=["company_id", "postUrl"] if "postUrl" in company_activity.columns else ["company_id"]
)

# ---------------------------
# Save cleaned versions
# ---------------------------
leads.to_csv("leads_clean.csv", index=False, encoding="utf-8")
activity.to_csv("activity_clean.csv", index=False, encoding="utf-8")
companies.to_csv("companies_clean.csv", index=False, encoding="utf-8")
company_activity.to_csv("company_activity_clean.csv", index=False, encoding="utf-8")

print("✅ Saved: leads_clean.csv, activity_clean.csv, companies_clean.csv, company_activity_clean.csv")
print(f"Leads: {len(leads)} | Activity rows: {len(activity)} | Companies: {len(companies)} | Company Activity rows: {len(company_activity)}")
