"""
BUS 445 Final Project – FibVID Diffusion Analysis v2
Fixes:
    - Supports FibVID column name: claim_num
    - Creates cleaner visuals: bar charts, line charts, scatterplots, and readable box plots
    - Uses fake vs real labels from FibVID group codes:
        0 = COVID True
        1 = COVID Fake
        2 = Non-COVID True
        3 = Non-COVID Fake

Run in VS Code Terminal:
    cd "/Users/anthonymoran/Desktop/Bus 445 python code"
    python fibvid_diffusion_analysis_v2.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("fibvid_outputs_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

CLAIMS_FILE = "news_claim.csv"
PROP_FILE = "claim_propagation.csv"
ORIGIN_FILE = "origin_tweet.csv"
USER_FILE = "user_information.csv"


def save_chart(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def find_col(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            f"Could not find one of these columns: {candidates}\n"
            f"Available columns are: {df.columns.tolist()}"
        )
    return None


def map_truth_label(group_value):
    try:
        g = int(group_value)
    except Exception:
        return "Unknown"

    if g in [0, 2]:
        return "Real"
    if g in [1, 3]:
        return "Fake"
    return "Unknown"


def map_topic(group_value):
    try:
        g = int(group_value)
    except Exception:
        return "Unknown"

    if g in [0, 1]:
        return "COVID"
    if g in [2, 3]:
        return "Non-COVID"
    return "Unknown"


def require_file(filename):
    if not Path(filename).exists():
        raise FileNotFoundError(
            f"\nMissing file: {filename}\n"
            f"Make sure {filename} is in the same folder as this Python script.\n"
        )


print("=" * 70)
print("BUS 445 – FibVID Fake News Diffusion Analysis v2")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load files
# ------------------------------------------------------------

require_file(CLAIMS_FILE)
require_file(PROP_FILE)

claims = standardize_columns(pd.read_csv(CLAIMS_FILE))
prop = standardize_columns(pd.read_csv(PROP_FILE))

origin = None
users = None

if Path(ORIGIN_FILE).exists():
    origin = standardize_columns(pd.read_csv(ORIGIN_FILE))

if Path(USER_FILE).exists():
    users = standardize_columns(pd.read_csv(USER_FILE))

print("\n[1/8] Files loaded successfully")
print(f"Claims rows:       {len(claims):,}")
print(f"Propagation rows:  {len(prop):,}")
if origin is not None:
    print(f"Origin tweet rows: {len(origin):,}")
if users is not None:
    print(f"User rows:         {len(users):,}")

print("\nClaims columns:")
print(claims.columns.tolist())

print("\nPropagation columns:")
print(prop.columns.tolist())

# ------------------------------------------------------------
# 2. Detect columns
# ------------------------------------------------------------

claim_id_claims = find_col(
    claims,
    ["claim_num", "claim_number", "claim_id", "claim_no", "claim"],
    required=True
)

group_claims = find_col(
    claims,
    ["group", "label", "class"],
    required=True
)

claim_id_prop = find_col(
    prop,
    ["claim_num", "claim_number", "claim_id", "claim_no", "claim"],
    required=True
)

group_prop = find_col(
    prop,
    ["group", "label", "class"],
    required=False
)

parent_user = find_col(
    prop,
    ["parent_user", "source_user", "from_user"],
    required=True
)

tweet_user = find_col(
    prop,
    ["tweet_user", "user", "target_user", "to_user"],
    required=True
)

depth_col = find_col(
    prop,
    ["depth", "diffusion_depth"],
    required=True
)

retweet_col = find_col(
    prop,
    ["retweet_count", "retweets", "retweet"],
    required=False
)

like_col = find_col(
    prop,
    ["like_count", "likes", "favorite_count"],
    required=False
)

date_col = find_col(
    prop,
    ["create_date", "created_at", "date", "tweet_date"],
    required=False
)

print("\n[2/8] Detected columns")
print(f"Claim ID in claims:      {claim_id_claims}")
print(f"Claim ID in propagation: {claim_id_prop}")
print(f"Group column in claims:  {group_claims}")
print(f"Parent user column:      {parent_user}")
print(f"Tweet user column:       {tweet_user}")
print(f"Depth column:            {depth_col}")
print(f"Retweet column:          {retweet_col}")
print(f"Like column:             {like_col}")
print(f"Date column:             {date_col}")

# ------------------------------------------------------------
# 3. Clean and merge
# ------------------------------------------------------------

claims["claim_id_clean"] = claims[claim_id_claims].astype(str)
prop["claim_id_clean"] = prop[claim_id_prop].astype(str)

claims["truth_label"] = claims[group_claims].apply(map_truth_label)
claims["topic"] = claims[group_claims].apply(map_topic)

merged = prop.merge(
    claims[["claim_id_clean", "truth_label", "topic"]],
    on="claim_id_clean",
    how="left"
)

# Use propagation group if merge labels are missing
if group_prop is not None:
    merged["truth_from_prop"] = merged[group_prop].apply(map_truth_label)
    merged["topic_from_prop"] = merged[group_prop].apply(map_topic)
    merged["truth_label"] = merged["truth_label"].fillna(merged["truth_from_prop"])
    merged["topic"] = merged["topic"].fillna(merged["topic_from_prop"])

merged["truth_label"] = merged["truth_label"].fillna("Unknown")
merged["topic"] = merged["topic"].fillna("Unknown")

merged[depth_col] = pd.to_numeric(merged[depth_col], errors="coerce")

if retweet_col is not None:
    merged[retweet_col] = pd.to_numeric(merged[retweet_col], errors="coerce").fillna(0)
else:
    merged["retweet_count"] = 0
    retweet_col = "retweet_count"

if like_col is not None:
    merged[like_col] = pd.to_numeric(merged[like_col], errors="coerce").fillna(0)
else:
    merged["like_count"] = 0
    like_col = "like_count"

if date_col is not None:
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged["month"] = merged[date_col].dt.to_period("M").astype(str)
else:
    merged["month"] = np.nan

merged[parent_user] = merged[parent_user].astype(str)
merged[tweet_user] = merged[tweet_user].astype(str)

analysis = merged.dropna(subset=[depth_col, parent_user, tweet_user]).copy()
analysis = analysis[analysis["truth_label"].isin(["Fake", "Real"])].copy()

print("\n[3/8] Cleaned analysis data")
print(f"Usable rows: {len(analysis):,}")
print("\nFake/Real row counts:")
print(analysis["truth_label"].value_counts())

analysis.to_csv(OUTPUT_DIR / "cleaned_fibvid_diffusion_data.csv", index=False)

# ------------------------------------------------------------
# 4. Claim-level statistics
# ------------------------------------------------------------

claim_stats = (
    analysis.groupby(["claim_id_clean", "truth_label", "topic"], as_index=False)
    .agg(
        propagation_volume=(tweet_user, "count"),
        unique_users=(tweet_user, "nunique"),
        unique_parent_users=(parent_user, "nunique"),
        max_depth=(depth_col, "max"),
        avg_depth=(depth_col, "mean"),
        total_retweets=(retweet_col, "sum"),
        avg_retweets=(retweet_col, "mean"),
        total_likes=(like_col, "sum"),
        avg_likes=(like_col, "mean")
    )
)

claim_stats.to_csv(OUTPUT_DIR / "claim_level_diffusion_stats.csv", index=False)

summary = (
    claim_stats.groupby("truth_label")
    .agg(
        total_claims=("claim_id_clean", "count"),
        avg_volume=("propagation_volume", "mean"),
        median_volume=("propagation_volume", "median"),
        avg_unique_users=("unique_users", "mean"),
        median_unique_users=("unique_users", "median"),
        avg_max_depth=("max_depth", "mean"),
        median_max_depth=("max_depth", "median"),
        avg_total_retweets=("total_retweets", "mean"),
        avg_total_likes=("total_likes", "mean")
    )
    .round(3)
)

summary.to_csv(OUTPUT_DIR / "fake_vs_real_summary.csv")
print("\n[4/8] Claim-level fake vs real summary")
print(summary)

# ------------------------------------------------------------
# 5. Visual 1: Bar chart propagation events
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
counts = analysis["truth_label"].value_counts().reindex(["Real", "Fake"])
plt.bar(counts.index, counts.values)
plt.title("Propagation Events by Claim Type")
plt.xlabel("Claim Type")
plt.ylabel("Number of Propagation Events")
for i, v in enumerate(counts.values):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom")
save_chart("01_bar_propagation_events_by_claim_type.png")

# ------------------------------------------------------------
# 6. Visual 2: Column chart claims by topic and truth
# ------------------------------------------------------------

topic_counts = (
    claim_stats.groupby(["topic", "truth_label"])
    .size()
    .unstack(fill_value=0)
)

topic_counts = topic_counts.reindex(columns=["Real", "Fake"], fill_value=0)
topic_counts.plot(kind="bar", figsize=(10, 6))
plt.title("Number of Claims by Topic and Truth Label")
plt.xlabel("Topic")
plt.ylabel("Number of Claims")
plt.xticks(rotation=0)
save_chart("02_column_claims_by_topic_and_truth.png")

# ------------------------------------------------------------
# 7. Visual 3: Line chart diffusion depth
# ------------------------------------------------------------

depth_counts = (
    analysis.groupby([depth_col, "truth_label"])
    .size()
    .reset_index(name="count")
    .sort_values(depth_col)
)

plt.figure(figsize=(10, 6))
for label in ["Real", "Fake"]:
    temp = depth_counts[depth_counts["truth_label"] == label]
    plt.plot(temp[depth_col], temp["count"], marker="o", linewidth=2, label=label)

plt.title("Diffusion Depth Pattern: Fake vs Real Claims")
plt.xlabel("Diffusion Depth")
plt.ylabel("Number of Propagation Events")
plt.legend()
plt.grid(True, alpha=0.3)
save_chart("03_line_diffusion_depth_fake_vs_real.png")

# ------------------------------------------------------------
# 8. Visual 4: Scatterplot volume vs depth
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
for label in ["Real", "Fake"]:
    temp = claim_stats[claim_stats["truth_label"] == label]
    plt.scatter(
        temp["propagation_volume"],
        temp["max_depth"],
        alpha=0.7,
        label=label
    )

plt.title("Claim Volume vs Maximum Diffusion Depth")
plt.xlabel("Propagation Volume per Claim")
plt.ylabel("Maximum Diffusion Depth")
plt.xscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
save_chart("04_scatter_volume_vs_depth_logx.png")

# ------------------------------------------------------------
# 9. Visual 5: Scatterplot retweets vs likes
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
for label in ["Real", "Fake"]:
    temp = claim_stats[claim_stats["truth_label"] == label]
    plt.scatter(
        temp["total_retweets"] + 1,
        temp["total_likes"] + 1,
        alpha=0.7,
        label=label
    )

plt.title("Engagement Scatterplot: Retweets vs Likes")
plt.xlabel("Total Retweets per Claim + 1")
plt.ylabel("Total Likes per Claim + 1")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
save_chart("05_scatter_retweets_vs_likes_log.png")

# ------------------------------------------------------------
# 10. Visual 6: Median metrics bar chart
# ------------------------------------------------------------

median_metrics = (
    claim_stats.groupby("truth_label")[["propagation_volume", "unique_users", "max_depth"]]
    .median()
    .reindex(["Real", "Fake"])
)

median_metrics.plot(kind="bar", figsize=(10, 6))
plt.title("Median Diffusion Metrics by Claim Type")
plt.xlabel("Claim Type")
plt.ylabel("Median Value")
plt.xticks(rotation=0)
save_chart("06_bar_median_diffusion_metrics.png")

# ------------------------------------------------------------
# 11. Visual 7: Monthly line chart
# ------------------------------------------------------------

if date_col is not None and analysis["month"].notna().sum() > 0:
    monthly = (
        analysis.dropna(subset=["month"])
        .groupby(["month", "truth_label"])
        .size()
        .reset_index(name="events")
        .sort_values("month")
    )

    plt.figure(figsize=(12, 6))
    for label in ["Real", "Fake"]:
        temp = monthly[monthly["truth_label"] == label]
        plt.plot(temp["month"], temp["events"], marker="o", linewidth=2, label=label)

    plt.title("Monthly Propagation Events: Fake vs Real")
    plt.xlabel("Month")
    plt.ylabel("Number of Propagation Events")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_chart("07_line_monthly_propagation_events.png")
else:
    print("Skipped monthly chart because no usable date column was found.")

# ------------------------------------------------------------
# 12. Visual 8: Top super-spreaders
# ------------------------------------------------------------

super_spreaders = (
    analysis.groupby(parent_user)
    .agg(
        spread_events=(tweet_user, "count"),
        unique_users_reached=(tweet_user, "nunique"),
        avg_depth=(depth_col, "mean")
    )
    .reset_index()
    .sort_values("spread_events", ascending=False)
    .head(15)
)

super_spreaders.to_csv(OUTPUT_DIR / "top_15_super_spreaders.csv", index=False)

plt.figure(figsize=(10, 7))
plt.barh(super_spreaders[parent_user], super_spreaders["spread_events"])
plt.gca().invert_yaxis()
plt.title("Top 15 Super-Spreaders by Propagation Events")
plt.xlabel("Propagation Events")
plt.ylabel("Pseudonymized Parent User ID")
save_chart("08_bar_top_15_super_spreaders.png")

# ------------------------------------------------------------
# 13. Outlier-safe box plots
# ------------------------------------------------------------

trimmed = claim_stats.copy()
for metric in ["propagation_volume", "unique_users", "total_retweets", "total_likes"]:
    upper = trimmed[metric].quantile(0.95)
    trimmed = trimmed[trimmed[metric] <= upper]

for metric in ["propagation_volume", "unique_users", "total_retweets", "total_likes"]:
    plt.figure(figsize=(8, 6))
    data = [
        trimmed.loc[trimmed["truth_label"] == "Real", metric],
        trimmed.loc[trimmed["truth_label"] == "Fake", metric],
    ]
    plt.boxplot(data, labels=["Real", "Fake"], showfliers=False)
    plt.title(f"{metric.replace('_', ' ').title()} by Claim Type")
    plt.xlabel("Claim Type")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, axis="y", alpha=0.3)
    save_chart(f"09_box_{metric}_outliers_hidden.png")

# ------------------------------------------------------------
# 14. Write report notes
# ------------------------------------------------------------

notes = []
notes.append("BUS 445 – FibVID Diffusion Analysis Notes\n")
notes.append("The FibVID dataset is organized with group codes: 0 = COVID True, 1 = COVID Fake, 2 = Non-COVID True, and 3 = Non-COVID Fake.")
notes.append("For this project, groups 0 and 2 were recoded as Real, while groups 1 and 3 were recoded as Fake.\n")
notes.append("Key summary results:\n")
notes.append(summary.to_string())
notes.append("\n\nChart explanation:")
notes.append("- Bar charts compare the overall amount of fake vs real propagation.")
notes.append("- Line charts show how fake and real claims behave across diffusion depth and time.")
notes.append("- Scatterplots show claim-level relationships such as volume vs depth and retweets vs likes.")
notes.append("- Box plots hide extreme outliers so the main pattern is readable while the CSV files still preserve full values.")
notes.append("\nBusiness meaning:")
notes.append("The analysis helps identify whether misinformation spreads broadly, deeply, or through a small number of high-activity users. This supports brand risk monitoring, faster response strategies, and platform moderation decisions.")

with open(OUTPUT_DIR / "report_interpretation_notes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(notes))

print(f"Saved: {OUTPUT_DIR / 'report_interpretation_notes.txt'}")

print("\n" + "=" * 70)
print("DONE – FibVID diffusion analysis completed successfully.")
print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")
print("=" * 70)
