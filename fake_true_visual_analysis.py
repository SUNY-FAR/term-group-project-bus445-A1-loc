"""
BUS 445 Final Project – Fake vs True News Visual Analysis
Author: Anthony Moran / Group Project

Purpose:
    This script performs a professional fake vs real news analysis using:
    - Fake.csv
    - True.csv

    It creates accurate, visually appealing charts similar in style to the FibVID analysis:
    - bar charts
    - column charts
    - line charts
    - scatterplots
    - box-and-whisker plots
    - word frequency charts
    - model performance charts
    - confusion matrices
    - ROC curves

Important:
    Fake.csv and True.csv do NOT contain retweets, user IDs, reposts, or diffusion depth.
    Therefore, this script analyzes article-level patterns and machine learning classification,
    not true social network diffusion.

Required files in same folder:
    Fake.csv
    True.csv

Run in VS Code Terminal:
    cd "/Users/anthonymoran/Desktop/Bus 445 python code"
    python fake_true_visual_analysis.py

Install packages if needed:
    python -m pip install pandas numpy matplotlib scikit-learn
"""

import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0. Settings
# ------------------------------------------------------------

FAKE_FILE = "Fake.csv"
TRUE_FILE = "True.csv"

OUTPUT_DIR = Path("fake_true_visual_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def require_file(filename):
    if not Path(filename).exists():
        raise FileNotFoundError(
            f"\nMissing file: {filename}\n"
            f"Put {filename} in the same folder as this Python script.\n"
        )


def save_chart(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def clean_text(text):
    """Basic text cleaning for word counts and modeling."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())


def sentence_count(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0
    pieces = re.split(r"[.!?]+", str(text))
    return len([p for p in pieces if p.strip()])


def avg_word_length(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    if len(words) == 0:
        return 0
    return sum(len(w) for w in words) / len(words)


def top_words(text_series, n=20):
    stop_words = set(ENGLISH_STOP_WORDS)
    extra_stop_words = {
        "said", "will", "would", "could", "also", "one", "two", "new",
        "year", "years", "people", "time", "news", "image", "images",
        "video", "according", "told", "say", "says"
    }
    stop_words = stop_words.union(extra_stop_words)

    words = []
    for text in text_series.dropna():
        for word in clean_text(text).split():
            if len(word) > 2 and word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(n), columns=["word", "count"])


def add_bar_labels(values):
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("=" * 75)
print("BUS 445 – Fake vs True News Visual Analysis")
print("=" * 75)

require_file(FAKE_FILE)
require_file(TRUE_FILE)

fake = pd.read_csv(FAKE_FILE)
true = pd.read_csv(TRUE_FILE)

fake.columns = fake.columns.str.strip().str.lower()
true.columns = true.columns.str.strip().str.lower()

required_columns = {"title", "text", "subject", "date"}

if not required_columns.issubset(fake.columns):
    raise ValueError(f"Fake.csv must contain these columns: {required_columns}. Found: {fake.columns.tolist()}")

if not required_columns.issubset(true.columns):
    raise ValueError(f"True.csv must contain these columns: {required_columns}. Found: {true.columns.tolist()}")

fake["label"] = 1
fake["label_name"] = "Fake"

true["label"] = 0
true["label_name"] = "Real"

df = pd.concat([fake, true], ignore_index=True)

print("\n[1/12] Files loaded successfully")
print(f"Fake articles: {len(fake):,}")
print(f"Real articles: {len(true):,}")
print(f"Total articles: {len(df):,}")

# ------------------------------------------------------------
# 2. Clean and engineer article-level features
# ------------------------------------------------------------

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["subject"] = df["subject"].fillna("Unknown")
df["date"] = df["date"].fillna("")

df["content"] = df["title"] + " " + df["text"]
df["clean_content"] = df["content"].apply(clean_text)

df["title_word_count"] = df["title"].apply(word_count)
df["text_word_count"] = df["text"].apply(word_count)
df["content_word_count"] = df["content"].apply(word_count)
df["sentence_count"] = df["text"].apply(sentence_count)
df["avg_words_per_sentence"] = np.where(
    df["sentence_count"] > 0,
    df["text_word_count"] / df["sentence_count"],
    0
)
df["avg_word_length"] = df["content"].apply(avg_word_length)
df["char_count"] = df["content"].astype(str).str.len()

# Parse date carefully
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
df["year_month"] = df["date_parsed"].dt.to_period("M").astype(str)
df.loc[df["year_month"] == "NaT", "year_month"] = np.nan

before_dupes = len(df)
df = df.drop_duplicates(subset=["content", "label"], keep="first").copy()
after_dupes = len(df)

print("\n[2/12] Feature engineering complete")
print(f"Duplicates removed: {before_dupes - after_dupes:,}")
print(f"Final usable rows: {after_dupes:,}")

df.to_csv(OUTPUT_DIR / "cleaned_fake_true_article_data.csv", index=False)

# ------------------------------------------------------------
# 3. Summary tables
# ------------------------------------------------------------

summary_by_label = (
    df.groupby("label_name")
    .agg(
        articles=("content", "count"),
        avg_title_words=("title_word_count", "mean"),
        median_title_words=("title_word_count", "median"),
        avg_text_words=("text_word_count", "mean"),
        median_text_words=("text_word_count", "median"),
        avg_sentences=("sentence_count", "mean"),
        median_sentences=("sentence_count", "median"),
        avg_words_per_sentence=("avg_words_per_sentence", "mean"),
        avg_word_length=("avg_word_length", "mean"),
    )
    .round(2)
    .reindex(["Real", "Fake"])
)

summary_by_subject = (
    df.groupby(["subject", "label_name"])
    .size()
    .reset_index(name="article_count")
    .sort_values("article_count", ascending=False)
)

summary_by_label.to_csv(OUTPUT_DIR / "summary_by_label.csv")
summary_by_subject.to_csv(OUTPUT_DIR / "summary_by_subject.csv", index=False)

print("\n[3/12] Summary by label")
print(summary_by_label)

# ------------------------------------------------------------
# 4. Visual 1: Column chart - article count by label
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
label_counts = df["label_name"].value_counts().reindex(["Real", "Fake"])
plt.bar(label_counts.index, label_counts.values)
plt.title("Article Count by Label")
plt.xlabel("Article Label")
plt.ylabel("Number of Articles")
add_bar_labels(label_counts.values)
save_chart("01_column_article_count_by_label.png")

# ------------------------------------------------------------
# 5. Visual 2: Column chart - article count by subject and label
# ------------------------------------------------------------

subject_pivot = (
    df.groupby(["subject", "label_name"])
    .size()
    .unstack(fill_value=0)
)

subject_pivot = subject_pivot.reindex(columns=["Real", "Fake"], fill_value=0)
subject_pivot["total"] = subject_pivot.sum(axis=1)
subject_pivot = subject_pivot.sort_values("total", ascending=False).drop(columns="total")

subject_pivot.plot(kind="bar", figsize=(13, 7))
plt.title("Article Counts by Subject and Label")
plt.xlabel("Subject")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45, ha="right")
save_chart("02_column_subject_distribution.png")

# ------------------------------------------------------------
# 6. Visual 3: Bar chart - median article structure metrics
# ------------------------------------------------------------

structure_metrics = (
    df.groupby("label_name")[["title_word_count", "text_word_count", "sentence_count"]]
    .median()
    .reindex(["Real", "Fake"])
)

structure_metrics.plot(kind="bar", figsize=(11, 6))
plt.title("Median Article Structure Metrics by Label")
plt.xlabel("Article Label")
plt.ylabel("Median Value")
plt.xticks(rotation=0)
save_chart("03_bar_median_article_structure_metrics.png")

# ------------------------------------------------------------
# 7. Visual 4: Box-and-whisker - article word count
# ------------------------------------------------------------

# Hide extreme outliers so the plot is readable but keep all data in CSV.
box_data = []
for label in ["Real", "Fake"]:
    values = df.loc[df["label_name"] == label, "text_word_count"]
    upper = values.quantile(0.95)
    box_data.append(values[values <= upper])

plt.figure(figsize=(8, 6))
plt.boxplot(box_data, labels=["Real", "Fake"], showfliers=False)
plt.title("Article Text Word Count by Label")
plt.xlabel("Article Label")
plt.ylabel("Text Word Count")
plt.grid(True, axis="y", alpha=0.3)
save_chart("04_box_text_word_count_by_label.png")

# ------------------------------------------------------------
# 8. Visual 5: Box-and-whisker - average words per sentence
# ------------------------------------------------------------

box_data = []
for label in ["Real", "Fake"]:
    values = df.loc[df["label_name"] == label, "avg_words_per_sentence"]
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    upper = values.quantile(0.95)
    box_data.append(values[values <= upper])

plt.figure(figsize=(8, 6))
plt.boxplot(box_data, labels=["Real", "Fake"], showfliers=False)
plt.title("Average Words per Sentence by Label")
plt.xlabel("Article Label")
plt.ylabel("Average Words per Sentence")
plt.grid(True, axis="y", alpha=0.3)
save_chart("05_box_avg_words_per_sentence_by_label.png")

# ------------------------------------------------------------
# 9. Visual 6: Scatterplot - title length vs article length
# ------------------------------------------------------------

sample_df = df.sample(n=min(5000, len(df)), random_state=42)

plt.figure(figsize=(10, 6))
for label in ["Real", "Fake"]:
    temp = sample_df[sample_df["label_name"] == label]
    plt.scatter(
        temp["title_word_count"],
        temp["text_word_count"],
        alpha=0.45,
        label=label
    )

plt.title("Scatterplot: Title Length vs Article Length")
plt.xlabel("Title Word Count")
plt.ylabel("Text Word Count")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
save_chart("06_scatter_title_length_vs_text_length.png")

# ------------------------------------------------------------
# 10. Visual 7: Scatterplot - sentence count vs word count
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
for label in ["Real", "Fake"]:
    temp = sample_df[sample_df["label_name"] == label]
    plt.scatter(
        temp["sentence_count"] + 1,
        temp["text_word_count"] + 1,
        alpha=0.45,
        label=label
    )

plt.title("Scatterplot: Sentence Count vs Text Word Count")
plt.xlabel("Sentence Count + 1")
plt.ylabel("Text Word Count + 1")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
save_chart("07_scatter_sentence_count_vs_word_count.png")

# ------------------------------------------------------------
# 11. Visual 8: Line chart - monthly article counts
# ------------------------------------------------------------

monthly = (
    df.dropna(subset=["year_month"])
    .groupby(["year_month", "label_name"])
    .size()
    .reset_index(name="articles")
    .sort_values("year_month")
)

if len(monthly) > 0:
    plt.figure(figsize=(13, 6))
    for label in ["Real", "Fake"]:
        temp = monthly[monthly["label_name"] == label]
        plt.plot(temp["year_month"], temp["articles"], marker="o", linewidth=2, label=label)

    plt.title("Monthly Article Counts: Fake vs Real")
    plt.xlabel("Month")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_chart("08_line_monthly_article_counts.png")
else:
    print("Skipped monthly line chart because no usable dates were parsed.")

# ------------------------------------------------------------
# 12. Visual 9: Line chart - monthly average article length
# ------------------------------------------------------------

monthly_length = (
    df.dropna(subset=["year_month"])
    .groupby(["year_month", "label_name"])["text_word_count"]
    .mean()
    .reset_index(name="avg_text_words")
    .sort_values("year_month")
)

if len(monthly_length) > 0:
    plt.figure(figsize=(13, 6))
    for label in ["Real", "Fake"]:
        temp = monthly_length[monthly_length["label_name"] == label]
        plt.plot(temp["year_month"], temp["avg_text_words"], marker="o", linewidth=2, label=label)

    plt.title("Monthly Average Article Length: Fake vs Real")
    plt.xlabel("Month")
    plt.ylabel("Average Text Word Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_chart("09_line_monthly_average_article_length.png")
else:
    print("Skipped monthly length chart because no usable dates were parsed.")

# ------------------------------------------------------------
# 13. Visual 10 and 11: Top words fake vs real
# ------------------------------------------------------------

fake_words = top_words(df[df["label_name"] == "Fake"]["content"], n=20)
real_words = top_words(df[df["label_name"] == "Real"]["content"], n=20)

fake_words.to_csv(OUTPUT_DIR / "top_20_words_fake.csv", index=False)
real_words.to_csv(OUTPUT_DIR / "top_20_words_real.csv", index=False)

plt.figure(figsize=(10, 7))
plt.barh(fake_words["word"], fake_words["count"])
plt.gca().invert_yaxis()
plt.title("Top 20 Frequent Words in Fake News Articles")
plt.xlabel("Frequency")
plt.ylabel("Word")
save_chart("10_bar_top_words_fake.png")

plt.figure(figsize=(10, 7))
plt.barh(real_words["word"], real_words["count"])
plt.gca().invert_yaxis()
plt.title("Top 20 Frequent Words in Real News Articles")
plt.xlabel("Frequency")
plt.ylabel("Word")
save_chart("11_bar_top_words_real.png")

# ------------------------------------------------------------
# 14. Machine learning models
# ------------------------------------------------------------

print("\n[4/12] Preparing machine learning models")

X = df["clean_content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=40, random_state=42),
}

pipelines = {
    name: Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("model", model)
    ])
    for name, model in models.items()
}

# ------------------------------------------------------------
# 15. Cross-validation
# ------------------------------------------------------------

print("\n[5/12] Running 5-fold cross-validation")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

cv_results = []

for name, pipe in pipelines.items():
    print(f"Cross-validating {name}...")
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    cv_results.append({
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Precision": scores["test_precision"].mean(),
        "CV Recall": scores["test_recall"].mean(),
        "CV F1": scores["test_f1"].mean(),
    })

cv_df = pd.DataFrame(cv_results).round(4)
cv_df.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

print("\nCross-validation results:")
print(cv_df)

# ------------------------------------------------------------
# 16. Final model testing
# ------------------------------------------------------------

print("\n[6/12] Training final models and testing performance")

test_results = []
best_model = None
best_model_name = None
best_f1 = -1

for name, pipe in pipelines.items():
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    result = {
        "Model": name,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred),
        "Test Recall": recall_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred),
    }

    test_results.append(result)

    report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])
    with open(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(values_format="d")
    plt.title(f"{name} Confusion Matrix")
    save_chart(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")

    try:
        RocCurveDisplay.from_estimator(pipe, X_test, y_test)
        plt.title(f"{name} ROC Curve")
        save_chart(f"roc_curve_{name.lower().replace(' ', '_')}.png")
    except Exception:
        pass

    if result["Test F1"] > best_f1:
        best_f1 = result["Test F1"]
        best_model = pipe
        best_model_name = name

test_df = pd.DataFrame(test_results).round(4)
test_df.to_csv(OUTPUT_DIR / "final_test_results.csv", index=False)

print("\nFinal test results:")
print(test_df)
print(f"\nBest model by F1-score: {best_model_name} ({best_f1:.4f})")

# ------------------------------------------------------------
# 17. Visual 12: model performance grouped bar chart
# ------------------------------------------------------------

performance_plot = test_df.set_index("Model")[["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]]

performance_plot.plot(kind="bar", figsize=(12, 7))
plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
save_chart("12_bar_model_performance_comparison.png")

# ------------------------------------------------------------
# 18. Visual 13: model ranking column chart
# ------------------------------------------------------------

ranked = test_df.sort_values("Test F1", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(ranked["Model"], ranked["Test F1"])
plt.title("Final Model Ranking by F1-Score")
plt.xlabel("Model")
plt.ylabel("F1-Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
for i, v in enumerate(ranked["Test F1"]):
    plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
save_chart("13_column_model_ranking_by_f1.png")

# ------------------------------------------------------------
# 19. Visual 14 and 15: predictive TF-IDF terms
# ------------------------------------------------------------

print("\n[7/12] Creating predictive term charts")

log_pipe = pipelines["Logistic Regression"]
log_pipe.fit(X_train, y_train)

tfidf = log_pipe.named_steps["tfidf"]
log_model = log_pipe.named_steps["model"]

feature_names = np.array(tfidf.get_feature_names_out())
coefficients = log_model.coef_[0]

top_fake_idx = np.argsort(coefficients)[-20:]
top_real_idx = np.argsort(coefficients)[:20]

top_fake_terms = pd.DataFrame({
    "term": feature_names[top_fake_idx],
    "coefficient": coefficients[top_fake_idx]
}).sort_values("coefficient", ascending=True)

top_real_terms = pd.DataFrame({
    "term": feature_names[top_real_idx],
    "coefficient": coefficients[top_real_idx]
}).sort_values("coefficient", ascending=True)

top_fake_terms.to_csv(OUTPUT_DIR / "top_predictive_terms_fake.csv", index=False)
top_real_terms.to_csv(OUTPUT_DIR / "top_predictive_terms_real.csv", index=False)

plt.figure(figsize=(10, 7))
plt.barh(top_fake_terms["term"], top_fake_terms["coefficient"])
plt.title("Top Predictive Terms for Fake News")
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Term")
save_chart("14_bar_predictive_terms_fake.png")

plt.figure(figsize=(10, 7))
plt.barh(top_real_terms["term"], top_real_terms["coefficient"])
plt.title("Top Predictive Terms for Real News")
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Term")
save_chart("15_bar_predictive_terms_real.png")

# ------------------------------------------------------------
# 20. Write report notes
# ------------------------------------------------------------

print("\n[8/12] Writing report notes")

notes = []
notes.append("BUS 445 – Fake vs True News Visual Analysis Notes\n")
notes.append("Dataset Description:")
notes.append(
    "This analysis uses Fake.csv and True.csv. Fake.csv was labeled as Fake = 1, while True.csv was labeled as Real = 0. "
    "The files include title, text, subject, and date fields."
)
notes.append("\nImportant Limitation:")
notes.append(
    "This dataset supports text classification and article-level comparison. It does not include retweets, user IDs, repost chains, or diffusion depth. "
    "Therefore, it should be used for the machine learning classification portion of the project, while FibVID should be used for network diffusion analysis."
)

notes.append("\nSummary by Label:")
notes.append(summary_by_label.to_string())

notes.append("\n\nCross-Validation Results:")
notes.append(cv_df.to_string(index=False))

notes.append("\n\nFinal Test Results:")
notes.append(test_df.to_string(index=False))

notes.append(f"\n\nBest Model:")
notes.append(f"The best model by F1-score was {best_model_name}, with an F1-score of {best_f1:.4f}.")

notes.append("\n\nVisual Explanation:")
notes.append("- Column charts compare article counts by fake/real label and subject.")
notes.append("- Bar charts compare median article structure, top words, and model performance.")
notes.append("- Box-and-whisker plots compare article length and sentence structure while hiding extreme outliers for readability.")
notes.append("- Scatterplots compare relationships between title length, sentence count, and article length.")
notes.append("- Line charts show monthly article volume and monthly average article length.")
notes.append("- Predictive-term charts show words and phrases most associated with fake or real labels.")

notes.append("\n\nProfessional Interpretation:")
notes.append(
    "The visual analysis helps show that fake and real news can differ in subject distribution, article length, writing structure, and vocabulary patterns. "
    "The machine learning models use TF-IDF features to classify articles based on these textual patterns. "
    "High model performance should be interpreted carefully because the dataset may contain source-specific wording patterns that make classification easier than it would be in a real-world misinformation monitoring system."
)

with open(OUTPUT_DIR / "report_interpretation_notes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(notes))

# ------------------------------------------------------------
# 21. Done
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("DONE – Fake vs True visual analysis completed successfully.")
print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")
print("=" * 75)
