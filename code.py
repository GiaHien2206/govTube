# =========================
# BBC News - Top 10% View Prediction
# =========================

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
import json
import joblib
from pathlib import Path


# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS = PROJECT_ROOT / "outputs_bbc"
OUTPUTS.mkdir(parents=True, exist_ok=True)

FILE_PATH = PROJECT_ROOT / "data" / "bbc_news_videos.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_EXCEL = OUTPUTS / "bbc_top10_prediction_results.xlsx"

# Từ khóa trong title, m có thể sửa thêm/bớt
KEYWORDS = [
    "crisis", "war", "election", "uk", "trump",
    "israel", "russia", "bbc", "president", "attack"
]


# ========= HELPERS =========
def find_existing_column(df, candidates):
    """Tìm tên cột thật trong dataframe."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def iso8601_to_seconds(duration_str):
    """
    Convert ISO 8601 duration (vd: PT8M17S, PT30S, P0D, PT1H2M3S) -> seconds.
    """
    if pd.isna(duration_str):
        return np.nan

    s = str(duration_str).strip().upper()

    # xử lý vài case rỗng
    if s in ["", "NAN", "NONE"]:
        return np.nan

    # P0D hoặc P1D...
    pattern = re.compile(
        r"P"
        r"(?:(?P<days>\d+)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
        r")?$"
    )

    match = pattern.fullmatch(s)
    if not match:
        return np.nan

    parts = match.groupdict(default="0")
    days = int(parts["days"])
    hours = int(parts["hours"])
    minutes = int(parts["minutes"])
    seconds = int(parts["seconds"])

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def build_features(df):
    df = df.copy()

    # ---- Detect raw columns ----
    title_col = find_existing_column(df, ["title"])
    published_col = find_existing_column(df, ["published", "publishTime", "publishedAt"])
    view_col = find_existing_column(df, ["viewCount", "views", "view_count"])
    like_col = find_existing_column(df, ["likeCount", "likes", "like_count"])
    comment_col = find_existing_column(df, ["commentCount", "comments", "comment_count"])
    duration_col = find_existing_column(df, ["duration", "video_duration"])

    required = {
        "title": title_col,
        "published": published_col,
        "viewCount": view_col,
        "likeCount": like_col,
        "commentCount": comment_col,
        "duration": duration_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    # ---- Basic cleaning ----
    df[title_col] = df[title_col].fillna("").astype(str)
    df[published_col] = pd.to_datetime(df[published_col], errors="coerce", utc=True)

    for c in [view_col, like_col, comment_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ---- Duration ----
    df["duration_seconds"] = df[duration_col].apply(iso8601_to_seconds).fillna(0)

    # ---- Title features ----
    df["title_length"] = df[title_col].str.len()
    df["title_word_count"] = df[title_col].str.split().str.len().fillna(0).astype(int)
    df["has_number_in_title"] = df[title_col].str.contains(r"\d", regex=True).astype(int)
    df["has_question_mark"] = df[title_col].str.contains(r"\?", regex=True).astype(int)

    keyword_pattern = "|".join(map(re.escape, KEYWORDS))
    df["has_keyword"] = df[title_col].str.contains(keyword_pattern, case=False, regex=True).astype(int)

    # ---- Time features ----
    df["upload_hour"] = df[published_col].dt.hour.fillna(0).astype(int)
    df["upload_day"] = df[published_col].dt.day_name().fillna("Unknown")
    df["is_weekend"] = df["upload_day"].isin(["Saturday", "Sunday"]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["upload_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["upload_hour"] / 24)

    # ---- Interaction features ----
    df["engagement"] = df[like_col] + df[comment_col]
    df["like_comment_ratio"] = df[like_col] / (df[comment_col] + 1)

    # ---- Optional log features (nếu muốn thêm, mở comment ra) ----
    # df["log_likeCount"] = np.log1p(df[like_col])
    # df["log_commentCount"] = np.log1p(df[comment_col])
    # df["log_engagement"] = np.log1p(df["engagement"])

    # ---- Target: top 10% view ----
    top10_threshold = df[view_col].quantile(0.90)
    df["is_top10_view"] = (df[view_col] >= top10_threshold).astype(int)

    # ---- One-hot upload day ----
    upload_day_dummies = pd.get_dummies(df["upload_day"], prefix="upload_day", dtype=int)

    # ---- Final feature tables ----
    metadata_features = pd.concat(
        [
            df[
                [
                    "duration_seconds",
                    "title_length",
                    "title_word_count",
                    "has_keyword",
                    "has_number_in_title",
                    "has_question_mark",
                    "hour_sin",
                    "hour_cos",
                    "is_weekend",
                ]
            ],
            upload_day_dummies,
        ],
        axis=1,
    )

    full_features = pd.concat(
        [
            metadata_features,
            df[
                [
                    like_col,
                    comment_col,
                    "engagement",
                    "like_comment_ratio",
                ]
            ].rename(columns={
                like_col: "likeCount",
                comment_col: "commentCount",
            }),
        ],
        axis=1,
    )

    # export-friendly dataframe
    export_df = df.copy()
    export_df["top10_threshold"] = top10_threshold

    return df, metadata_features, full_features, top10_threshold


def evaluate_models(X_train, X_test, y_train, y_test, random_state=42):
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )
    }

    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append({
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

        fitted_models[model_name] = model

    results_df = pd.DataFrame(rows)
    return results_df, fitted_models


# ========= MAIN =========
df_raw = pd.read_csv(FILE_PATH)

# Tạo feature
df_processed, X_metadata, X_full, threshold_90 = build_features(df_raw)
y = df_processed["is_top10_view"]

print("=" * 60)
print(f"Top 10% threshold (90th percentile of views): {threshold_90:.0f}")
print("Class distribution:")
print(y.value_counts(dropna=False))
print(y.value_counts(normalize=True).round(4))
print("=" * 60)

# Tách train/test bằng cùng 1 index để so sánh công bằng
train_idx, test_idx = train_test_split(
    df_processed.index,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

X_meta_train = X_metadata.loc[train_idx]
X_meta_test = X_metadata.loc[test_idx]

X_full_train = X_full.loc[train_idx]
X_full_test = X_full.loc[test_idx]

y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

# ---- Experiment 1: Metadata-only ----
meta_results, meta_models = evaluate_models(
    X_meta_train, X_meta_test, y_train, y_test, random_state=RANDOM_STATE
)
meta_results.insert(0, "Feature Setting", "Metadata-only")

# ---- Experiment 2: Full-feature ----
full_results, full_models = evaluate_models(
    X_full_train, X_full_test, y_train, y_test, random_state=RANDOM_STATE
)
full_results.insert(0, "Feature Setting", "Full-feature")

# Gộp bảng kết quả
all_results = pd.concat([meta_results, full_results], ignore_index=True)

print("\nMODEL RESULTS")
print(all_results)

# ---- Feature importance của XGBoost full-feature ----
xgb_full = full_models["XGBoost"]
xgb_importance = pd.DataFrame({
    "Feature": X_full.columns,
    "Importance": xgb_full.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTOP 15 IMPORTANT FEATURES (XGBoost - Full-feature)")
print(xgb_importance.head(15))

# ---- Tạo bảng export giống kiểu m đang làm ----
export_base_cols = []
for c in ["video_id", "title", "published", "viewCount", "likeCount", "commentCount", "duration"]:
    real_col = find_existing_column(df_processed, [c])
    if real_col is not None:
        export_base_cols.append(real_col)

bbc_processed_dataset = df_processed[export_base_cols].copy()

bbc_model_features = pd.concat(
    [
        df_processed[export_base_cols].reset_index(drop=True),
        X_full.reset_index(drop=True),
        df_processed[["is_top10_view"]].reset_index(drop=True),
    ],
    axis=1
)

# ---- Export Excel ----
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    bbc_processed_dataset.to_excel(writer, sheet_name="bbc_processed_dataset", index=False)
    bbc_model_features.to_excel(writer, sheet_name="bbc_model_features", index=False)
    all_results.to_excel(writer, sheet_name="model_results", index=False)
    xgb_importance.to_excel(writer, sheet_name="xgb_full_importance", index=False)

print(f"\nĐã xuất file: {OUTPUT_EXCEL}")

# =========================
# VISUALIZATION: CONFUSION MATRIX + FEATURE IMPORTANCE
# =========================

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, model_name, feature_setting, output_prefix):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Top 10%", "Top 10%"])
    disp.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(f"Confusion Matrix - {model_name} ({feature_setting})")
    plt.tight_layout()

    filename = f"{output_prefix}_{feature_setting}_{model_name.lower().replace(' ', '_')}_cm.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename


def save_feature_importance(importance_df, top_n=15, output_file="xgb_feature_importance.png"):
    top_features = importance_df.head(top_n).iloc[::-1]  # đảo ngược để thanh lớn nhất ở trên

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["Feature"], top_features["Importance"])
    ax.set_title("Top Feature Importance - XGBoost")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


# ---- Save confusion matrices for Metadata-only ----
cm_files = []

for model_name, model in meta_models.items():
    y_pred = model.predict(X_meta_test)
    cm_file = save_confusion_matrix(
        y_test,
        y_pred,
        model_name=model_name,
        feature_setting="metadata_only",
        output_prefix="bbc"
    )
    cm_files.append(cm_file)

# ---- Save confusion matrices for Full-feature ----
for model_name, model in full_models.items():
    y_pred = model.predict(X_full_test)
    cm_file = save_confusion_matrix(
        y_test,
        y_pred,
        model_name=model_name,
        feature_setting="full_feature",
        output_prefix="bbc"
    )
    cm_files.append(cm_file)

# ---- Save XGBoost feature importance chart ----
importance_chart_file = save_feature_importance(
    xgb_importance,
    top_n=15,
    output_file="bbc_xgb_full_feature_importance.png"
)

print("\nĐã lưu các file confusion matrix:")
for f in cm_files:
    print("-", f)

print("\nĐã lưu file feature importance:")
print("-", importance_chart_file)

# =========================
# SAVE OUTPUTS FOR APP / CHATBOT
# =========================

# 1) Save best model for deployment
best_model = full_models["XGBoost"]
joblib.dump(best_model, OUTPUTS / "xgb_model.joblib")

# 2) Save model comparison CSV
all_results.to_csv(OUTPUTS / "model_comparison.csv", index=False)

# 3) Save XGBoost feature importance CSV
xgb_importance.to_csv(OUTPUTS / "xgb_feature_importance.csv", index=False)

# 4) Save processed dataset preview
bbc_model_features.to_csv(OUTPUTS / "bbc_model_features.csv", index=False)

# 5) Save training info JSON
training_info = {
    "project": "BBC News Top 10% View Prediction",
    "target_name": "is_top10_view",
    "target_threshold_90": float(threshold_90),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "metadata_feature_count": int(X_metadata.shape[1]),
    "full_feature_count": int(X_full.shape[1]),
    "rows_used": int(len(df_processed)),
    "best_model_for_app": "XGBoost",
    "best_model_setting": "Full-feature",
    "results": all_results.to_dict(orient="records"),
}
with open(OUTPUTS / "training_info.json", "w", encoding="utf-8") as f:
    json.dump(training_info, f, ensure_ascii=False, indent=2)

# 6) Save dataset info JSON
dataset_info = {
    "raw_shape": list(df_raw.shape),
    "processed_shape": list(df_processed.shape),
    "metadata_feature_shape": list(X_metadata.shape),
    "full_feature_shape": list(X_full.shape),
    "raw_columns": df_raw.columns.tolist(),
    "processed_columns": df_processed.columns.tolist(),
    "metadata_features": X_metadata.columns.tolist(),
    "full_features": X_full.columns.tolist(),
}
with open(OUTPUTS / "dataset_info.json", "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

# 7) Save summary JSON
best_rows = all_results[
    (all_results["Feature Setting"] == "Full-feature") &
    (all_results["Model"] == "XGBoost")
].to_dict(orient="records")

summary = {
    "project": "BBC News Top 10% View Prediction",
    "notes": "Predict whether a BBC News video belongs to the top 10% by views.",
    "top10_threshold_view_count": float(threshold_90),
    "best_model_summary": best_rows[0] if best_rows else {},
    "keyword_list": KEYWORDS,
}
with open(OUTPUTS / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nĐã lưu output cho BBC app vào:", OUTPUTS)