from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import re
import traceback
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from joblib import load

app = Flask(__name__)

# ================= PATHS =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ================= DEFAULT FEATURES =================
DEFAULT_FEATURES_6 = [
    "like_count",
    "comment_count",
    "hours_since_publish",
    "log_like_count",
    "log_comment_count",
    "log_hours_since_publish",
]

DEFAULT_FEATURES_3 = [
    "like_count",
    "comment_count",
    "hours_since_publish",
]


# ================= LOAD MODEL =================
def safe_load_joblib(path: Path):
    try:
        model = load(path)
        print("Loaded model from:", path)
        print("Model type:", type(model))
        return model
    except Exception as e:
        print("FAILED TO LOAD MODEL:", path)
        print("LOAD ERROR:", repr(e))
        return None


xgb_model = safe_load_joblib(OUTPUTS / "xgb_model.joblib")


# ================= MODEL FEATURE DETECTION =================
def get_model_feature_names(model):
    if model is None:
        return DEFAULT_FEATURES_6

    # 1) sklearn-style feature names
    feature_names_in = getattr(model, "feature_names_in_", None)
    if feature_names_in is not None:
        names = list(feature_names_in)
        print("Using model.feature_names_in_:", names)
        return names

    # 2) xgboost booster feature names
    try:
        booster_names = model.get_booster().feature_names
        if booster_names:
            print("Using booster feature names:", booster_names)
            return list(booster_names)
    except Exception as e:
        print("Cannot read booster feature names:", repr(e))

    # 3) fallback by n_features_in_
    n_features = getattr(model, "n_features_in_", None)
    if n_features == 3:
        print("Fallback to DEFAULT_FEATURES_3")
        return DEFAULT_FEATURES_3
    if n_features == 6:
        print("Fallback to DEFAULT_FEATURES_6")
        return DEFAULT_FEATURES_6

    # 4) final fallback
    print("Fallback to DEFAULT_FEATURES_6 (unknown model feature config)")
    return DEFAULT_FEATURES_6


MODEL_FEATURES = get_model_feature_names(xgb_model)
print("MODEL_FEATURES =", MODEL_FEATURES)


# ================= FEATURE BUILD =================
def build_feature_row(like, comment, hours):
    like = float(like)
    comment = float(comment)
    hours = float(hours)

    raw_values = {
        "like_count": like,
        "comment_count": comment,
        "hours_since_publish": hours,
        "log_like_count": float(np.log1p(like)),
        "log_comment_count": float(np.log1p(comment)),
        "log_hours_since_publish": float(np.log1p(hours)),
    }

    # tạo row theo đúng thứ tự feature model yêu cầu
    row = {}
    for col in MODEL_FEATURES:
        if col in raw_values:
            row[col] = raw_values[col]
        else:
            # cột lạ thì tạm cho 0.0 để không crash
            row[col] = 0.0
            print(f"WARNING: Unknown feature '{col}' -> filled with 0.0")

    X = pd.DataFrame([row], columns=MODEL_FEATURES).astype("float64")

    print("X columns:", X.columns.tolist())
    print("X shape:", X.shape)
    print("X dtypes:", {k: str(v) for k, v in X.dtypes.items()})
    print("X row:", X.to_dict(orient="records")[0])

    return X


# ================= PREDICT =================
def predict(like, comment, hours):
    if xgb_model is None:
        raise RuntimeError("xgb_model is None. Không load được file outputs/xgb_model.joblib")

    X = build_feature_row(like, comment, hours)

    print("model n_features_in_:", getattr(xgb_model, "n_features_in_", None))
    print("model feature_names_in_:", getattr(xgb_model, "feature_names_in_", None))

    try:
        print("booster feature names:", xgb_model.get_booster().feature_names)
    except Exception as e:
        print("cannot read booster feature names:", repr(e))

    proba = xgb_model.predict_proba(X)
    print("predict_proba raw:", proba)

    prob = float(proba[:, 1][0])
    pred = int(prob >= 0.5)

    return prob, pred


# ================= YOUTUBE HELPERS =================
def extract_video_id(url):
    if not url:
        return None

    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    return None


def get_video_stats(video_id):
    if not YOUTUBE_API_KEY:
        raise RuntimeError("Thiếu biến môi trường YOUTUBE_API_KEY")

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "statistics,snippet",
        "id": video_id,
        "key": YOUTUBE_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    if not items:
        return None

    item = items[0]
    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})

    return {
        "title": snippet.get("title", ""),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
    }


# ================= TIME SERIES SIMULATION =================
def build_timeseries(views, likes, comments):
    hours = [1, 6, 12, 24, 48]

    views = float(views)
    likes = float(likes)
    comments = float(comments)

    views_series = [views * 0.05, views * 0.15, views * 0.30, views * 0.60, views]
    likes_series = [likes * 0.10, likes * 0.30, likes * 0.50, likes * 0.70, likes]
    comments_series = [comments * 0.10, comments * 0.30, comments * 0.60, comments * 0.80, comments]

    return {
        "hours": hours,
        "views": views_series,
        "likes": likes_series,
        "comments": comments_series,
    }


# ================= ROUTES =================
@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict_api():
    try:
        data = request.get_json(force=True)
        print("Received /api/predict data:", data)

        like = float(data["like"])
        comment = float(data["comment"])
        hours = float(data["hours"])

        prob, pred = predict(like, comment, hours)

        return jsonify({
            "ok": True,
            "prob": prob,
            "pred": pred,
            "model_features": MODEL_FEATURES,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.post("/api/youtube_predict")
def youtube_predict():
    try:
        data = request.get_json(force=True)
        print("Received /api/youtube_predict data:", data)

        url = data.get("url", "").strip()
        vid = extract_video_id(url)

        if not vid:
            return jsonify({
                "ok": False,
                "error": "Invalid URL"
            }), 400

        stats = get_video_stats(vid)
        if not stats:
            return jsonify({
                "ok": False,
                "error": "Video not found"
            }), 404

        # tạm giữ hours = 100 như code cũ của bạn
        prob, pred = predict(stats["likes"], stats["comments"], 100)
        ts = build_timeseries(stats["views"], stats["likes"], stats["comments"])

        return jsonify({
            "ok": True,
            "stats": stats,
            "prob": prob,
            "pred": pred,
            "timeseries": ts,
            "model_features": MODEL_FEATURES,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.get("/outputs/<path:filename>")
def outputs_files(filename):
    return send_from_directory(OUTPUTS, filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, debug=True)