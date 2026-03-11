import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import json

# =========================
# 1 Load data
# =========================

meta = pd.read_csv("data/videos_metadata_latest.csv")
ts = pd.read_csv("data/engagement_timeseries.csv")

print("Meta:", meta.shape)
print("Timeseries:", ts.shape)

# =========================
# 2 Clean data
# =========================

meta = meta.drop_duplicates("video_id")

ts = ts.sort_values(["video_id", "snapshot_time_utc"])

# =========================
# 3 Engagement ratios
# =========================

meta["like_view_ratio"] = meta["likeCount"] / meta["viewCount"].replace(0,1)
meta["comment_view_ratio"] = meta["commentCount"] / meta["viewCount"].replace(0,1)

# =========================
# 4 Burst features
# =========================

ts["delta_views"] = ts.groupby("video_id")["viewCount"].diff()

burst = ts.groupby("video_id")["delta_views"].max().reset_index()
burst.columns = ["video_id","burst_score"]

# =========================
# 5 Growth rate
# =========================

growth = ts.groupby("video_id")["viewCount"].agg(["first","last"]).reset_index()

growth["growth_rate"] = (growth["last"] - growth["first"]) / growth["first"].replace(0,1)

growth = growth[["video_id","growth_rate"]]

# =========================
# 6 Merge all features
# =========================

df = meta.merge(burst, on="video_id", how="left")
df = df.merge(growth, on="video_id", how="left")

df = df.fillna(0)

# =========================
# 7 Create label
# =========================

threshold = df["viewCount"].quantile(0.90)

df["label"] = (df["viewCount"] >= threshold).astype(int)

print("Trending threshold:", threshold)

# =========================
# 8 Feature list
# =========================

features = [

    "viewCount",
    "likeCount",
    "commentCount",

    "like_view_ratio",
    "comment_view_ratio",

    "burst_score",
    "growth_rate"

]

X = df[features]
y = df["label"]

print("Features:", features)

# =========================
# 9 Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

# =========================
# 10 Handle imbalance
# =========================

pos = y_train.sum()
neg = len(y_train) - pos

scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)

# =========================
# 11 Train XGBoost
# =========================

model = XGBClassifier(

    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42

)

model.fit(X_train, y_train)

print("Model trained successfully")

# =========================
# 12 Evaluation
# =========================

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

print("\nClassification Report\n")

print(classification_report(y_test,pred))

print("ROC AUC:", roc_auc_score(y_test,prob))

# =========================
# 13 Feature importance
# =========================

importance = model.feature_importances_

imp_df = pd.DataFrame({

    "feature": features,
    "importance": importance

}).sort_values("importance", ascending=False)

print("\nFeature importance\n")
print(imp_df)

plt.figure(figsize=(8,4))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# =========================
# 14 Save model
# =========================

Path("outputs").mkdir(exist_ok=True)

joblib.dump(model, "outputs/xgb_model.joblib")

print("Model saved to outputs/xgb_model.joblib")

# save feature list

with open("outputs/baseline_feature_cols.json","w") as f:
    json.dump(features,f)

print("Feature list saved")