from flask import Flask, request, jsonify, render_template, send_from_directory
import os, re
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from joblib import load

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ================= LOAD MODEL =================
def safe_load_joblib(path):
    try:
        return load(path)
    except:
        return None

xgb_model = safe_load_joblib(OUTPUTS / "xgb_model.joblib")

baseline_features = [
"like_count","comment_count","hours_since_publish",
"log_like_count","log_comment_count","log_hours_since_publish"
]

# ================= FEATURE BUILD =================
def build_feature_row(like,comment,hours):

    row={
    "like_count":like,
    "comment_count":comment,
    "hours_since_publish":hours,
    "log_like_count":float(np.log1p(like)),
    "log_comment_count":float(np.log1p(comment)),
    "log_hours_since_publish":float(np.log1p(hours))
    }

    return pd.DataFrame([[row.get(c,0.0) for c in baseline_features]],columns=baseline_features)

# ================= PREDICT =================
def predict(like,comment,hours):

    X=build_feature_row(like,comment,hours)

    prob=float(xgb_model.predict_proba(X)[:,1][0])
    pred=int(prob>=0.5)

    return prob,pred

# ================= YOUTUBE =================
def extract_video_id(url):

    match=re.search(r"(?:v=|youtu.be/)([A-Za-z0-9_-]{11})",url)

    if match:
        return match.group(1)

    return None

def get_video_stats(video_id):

    url="https://www.googleapis.com/youtube/v3/videos"

    params={
    "part":"statistics,snippet",
    "id":video_id,
    "key":YOUTUBE_API_KEY
    }

    r=requests.get(url,params=params).json()

    if not r["items"]:
        return None

    stats=r["items"][0]["statistics"]
    snippet=r["items"][0]["snippet"]

    return {
    "title":snippet["title"],
    "views":int(stats.get("viewCount",0)),
    "likes":int(stats.get("likeCount",0)),
    "comments":int(stats.get("commentCount",0))
    }

# ================= TIME SERIES SIMULATION =================
def build_timeseries(views,likes,comments):

    hours=[1,6,12,24,48]

    views_series=[views*0.05,views*0.15,views*0.30,views*0.60,views]
    likes_series=[likes*0.1,likes*0.3,likes*0.5,likes*0.7,likes]
    comments_series=[comments*0.1,comments*0.3,comments*0.6,comments*0.8,comments]

    return {
    "hours":hours,
    "views":views_series,
    "likes":likes_series,
    "comments":comments_series
    }

# ================= ROUTES =================
@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/predict")
def predict_api():

    data=request.json

    like=float(data["like"])
    comment=float(data["comment"])
    hours=float(data["hours"])

    prob,pred=predict(like,comment,hours)

    return jsonify({
    "prob":prob,
    "pred":pred
    })

@app.post("/api/youtube_predict")
def youtube_predict():

    url=request.json.get("url")

    vid=extract_video_id(url)

    if not vid:
        return jsonify({"error":"Invalid URL"})

    stats=get_video_stats(vid)

    if not stats:
        return jsonify({"error":"Video not found"})

    prob,pred=predict(stats["likes"],stats["comments"],100)

    ts=build_timeseries(stats["views"],stats["likes"],stats["comments"])

    return jsonify({
    "stats":stats,
    "prob":prob,
    "pred":pred,
    "timeseries":ts
    })

@app.get("/outputs/<path:filename>")
def outputs_files(filename):
    return send_from_directory(OUTPUTS, filename)

if __name__=="__main__":

    port=int(os.getenv("PORT","3000"))
    app.run(host="0.0.0.0",port=port)