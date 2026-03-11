from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os, re, json, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from flask import send_from_directory

# ===================== Paths & ENV =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # root project (parent of web_chatbot/)
OUTPUTS = PROJECT_ROOT / "outputs"

ENV_PATH = Path(__file__).resolve().parent / ".env" 


load_dotenv(dotenv_path=ENV_PATH)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
from flask import send_from_directory



@app.get("/favicon.ico")
def favicon():
    return ("", 204)

print("APP VERSION: OFFLINE_CHAT_V2")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUTS:", OUTPUTS)

# ===================== Load artifacts =====================
def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_load_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_load_joblib(path: Path):
    try:
        return load(path)
    except Exception:
        return None

burst_stats = safe_load_json(OUTPUTS / "burst_stats.json")
print("Loaded burst_stats.json:", burst_stats is not None)
dataset_info = safe_load_json(OUTPUTS / "dataset_info.json")
comparison_df = safe_load_csv(OUTPUTS / "model_comparison.csv")
week2_df = safe_load_csv(OUTPUTS / "week2_model_table.csv")
gov_counts_df = safe_load_csv(OUTPUTS / "gov_channels_video_counts.csv")
print("Loaded gov_channels_video_counts.csv:", gov_counts_df is not None)

baseline_features = safe_load_json(OUTPUTS / "baseline_feature_cols.json") or [
    "like_count","comment_count","hours_since_publish",
    "log_like_count","log_comment_count","log_hours_since_publish"
]

xgb_model = safe_load_joblib(OUTPUTS / "xgb_model.joblib")

print("Loaded dataset_info:", bool(dataset_info))
print("Loaded model_comparison.csv:", comparison_df is not None)
print("Loaded week2_model_table.csv:", week2_df is not None)
print("Loaded baseline_feature_cols.json:", bool(baseline_features))
print("Loaded xgb_model.joblib:", xgb_model is not None)

# ===================== Text normalize (Vietnamese accents) =====================
def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # remove accents
    s = re.sub(r"\s+", " ", s)
    return s

# ===================== Replies formatters =====================
HELP = (
    "Bạn có thể hỏi:\n"
    "- 'GovTrendTube là gì'\n"
    "- 'dataset' (bao nhiêu video + cột)\n"
    "- 'proxy label' / 'threshold'\n"
    "- 'bảng so sánh model' / 'kết quả'\n"
    "- 'ablation'\n"
    "- 'shap' / 'feature importance'\n"
    "- 'limitations' / 'hạn chế'\n"
    "- 'predict like=500 comment=20 hours=100'\n"
)

def fmt_burst():
    if not burst_stats:
        return "Chưa có outputs/burst_stats.json. Hãy export burst stats từ notebook."
    thr = burst_stats.get("burst_threshold_top10_max_delta_views")
    c1 = burst_stats.get("count_has_burst_1")
    c0 = burst_stats.get("count_has_burst_0")
    tot = burst_stats.get("total_videos")
    return (
        "Burst stats :\n"
        f"- Burst threshold (top 10% max Δviews): {thr:.4f}\n"
        f"- Videos with burst (has_burst=1): {c1}/{tot}\n"
        f"- Videos without burst (has_burst=0): {c0}/{tot}\n"
        "Ghi chú: burst_score = max(delta_views) trong K snapshots đầu."
    )

def fmt_gov_channels():
    if gov_counts_df is None:
        return "Chưa có outputs/gov_channels_video_counts.csv. Hãy chạy cell export trong notebook trước."
    lines = ["Gov channels + số video:"]
    for _, r in gov_counts_df.iterrows():
        agency = r.get("agency")
        num = int(r.get("num_videos", 0))
        handle = r.get("channel_handle", "")
        url = r.get("channel_url", "")
        extra = ""
        if isinstance(handle, str) and handle.strip():
            extra += f" {handle.strip()}"
        if isinstance(url, str) and url.strip():
            extra += f" | {url.strip()}"
        lines.append(f"- {agency}: {num} videos{extra}")
    lines.append(f"\nTổng video: {int(gov_counts_df['num_videos'].sum())} ")
    return "\n".join(lines)

def fmt_dataset():
    if not dataset_info:
        return "Chưa có outputs/dataset_info.json. Hãy chạy cell export dataset_info trong notebook."
    lines = ["Dataset info :"]
    for k in ["gov","channel_videos","eng","mostpop","meta"]:
        if k in dataset_info:
            lines.append(f"- {k}: shape={dataset_info[k]['shape']}")
            lines.append(f"  columns={dataset_info[k]['columns']}")
    lines.append("")
    lines.append(f"Unique videos (meta): {dataset_info.get('meta',{}).get('unique_videos')}")
    lines.append(f"Unique videos (eng): {dataset_info.get('eng',{}).get('unique_videos')}")
    return "\n".join(lines)

def fmt_comparison():
    if comparison_df is None:
        return {"type":"text", "text":"Chưa có outputs/model_comparison.csv."}
    df = comparison_df.copy()
    # làm tròn cho đẹp
    for c in ["Accuracy","Precision","Recall","F1","ROC_AUC","AUC"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(4)
    return {"type":"table", "title":"Bảng so sánh model (baseline)", "columns":df.columns.tolist(), "rows":df.values.tolist()}

def fmt_week2():
    if week2_df is None:
        return {"type":"text", "text":"Chưa có outputs/week2_model_table.csv."}
    df = week2_df.copy()
    for c in ["AUC","F1","Precision","Recall","Accuracy","ROC_AUC"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(4)
    return {"type":"table", "title":"Ablation", "columns":df.columns.tolist(), "rows":df.values.tolist()}

def fmt_proxy_label():
    # bạn có thể sửa nội dung này theo đúng label bạn dùng trong report
    return (
        "Proxy label :\n"
        "- label=1 nếu video thuộc top 10% popularity\n"
        "- label=0 nếu còn lại\n"
    )

def fmt_shap():
    return (
        "XAI (SHAP/Feature importance):\n"
        "- like_count là feature quan trọng nhất\n"
        "- hours_since_publish cũng ảnh hưởng\n"
        "- comment_count thường yếu hơn trong dataset hiện tại\n"
        "\nẢnh SHAP: /outputs/shap_summary.png"
        
    )

def fmt_limitations():
    return (
        "Limitations:\n"
        "1) Không overlap với mostPopular → không có nhãn trending thật.\n"
        "2) Snapshot chủ yếu theo thời điểm crawl → chưa đủ early prediction 24h đầu.\n"
        "3) Bài toán hiện tại là proxy-based popularity prediction."
    )


# ===================== Prediction helper =====================
def parse_predict(text: str):
    t = text.lower()
    def grab(key):
        m = re.search(rf"{key}\s*=\s*([0-9]+(\.[0-9]+)?)", t)
        return float(m.group(1)) if m else None
    like = grab("like")
    comment = grab("comment")
    hours = grab("hours") or grab("hour")
    return like, comment, hours

def build_feature_row(like_count: float, comment_count: float, hours: float):
    like_count = float(like_count)
    comment_count = float(comment_count)
    hours = float(hours)

    row = {
        "like_count": like_count,
        "comment_count": comment_count,
        "hours_since_publish": hours,
        "log_like_count": float(np.log1p(like_count)),
        "log_comment_count": float(np.log1p(comment_count)),
        "log_hours_since_publish": float(np.log1p(hours)),
    }
    return pd.DataFrame([[row.get(c, 0.0) for c in baseline_features]], columns=baseline_features)

def predict_reply(like, comment, hours):
    if xgb_model is None:
        return "Chưa load được outputs/xgb_model.joblib. Hãy export model XGBoost từ notebook."
    if like is None or comment is None or hours is None:
        return "Sai format. Dùng: predict like=500 comment=20 hours=100"

    X = build_feature_row(like, comment, hours)
    prob = float(xgb_model.predict_proba(X)[:, 1][0])
    pred = int(prob >= 0.5)

    return (
        "Prediction (XGBoost baseline):\n"
        f"- input: like={like}, comment={comment}, hours={hours}\n"
        f"- prob(high-popularity) = {prob:.4f}\n"
        f"- pred(label) = {pred}  (1=phổ biến cao / 0=không)\n"
    )

# ===================== Main offline chat =====================
def offline_chat(message: str):
    m = norm_text(message)

    if any(k in m for k in ["burst", "burst stats", "event", "delta views", "max delta"]):
        return fmt_burst()
    
    if any(k in m for k in ["timestep", "timestep xai", "time xai", "xai time", "time-step", "occlusion"]):
        return {"type":"text", "text":"Time-step XAI (Occlusion): /outputs/time.png"}
    
    if any(k in m for k in ["kenh", "gov channel", "gov channels", "agency", "bao nhieu video", "channel"]):
        return fmt_gov_channels()

    if any(k in m for k in ["shap", "importance", "feature"]):
        return fmt_shap()
       
    
    if not m:
        return HELP

    # predict
    if m.startswith("predict"):
        like, comment, hours = parse_predict(m)
        return predict_reply(like, comment, hours)

    # help/menu
    if any(k in m for k in ["help", "menu", "giup", "huong dan"]):
        return HELP

    # intro
    if any(k in m for k in ["govtrendtube", "gioi thieu", "de tai", "project", "la gi"]):
        return (
            "Nghiên cứu này đề xuất GovTrendTube, một khung phương pháp dựa trên dữ liệu nhằm dự đoán sớm và giải thích các video thịnh hành trên YouTube, với trọng tâm ban đầu là các kênh YouTube chính thức của chính phủ. Thay vì sử dụng các danh sách kênh chưa được xác minh hoặc có nhiều nhiễu, nghiên cứu sử dụng U.S. Social Media Registry / U.S. Digital Registry, cung cấp danh sách chính thức các tài khoản mạng xã hội của chính phủ Hoa Kỳ. Điều này giúp đảm bảo độ tin cậy cao của dữ liệu và giảm sự mơ hồ trong quá trình thu thập dữ liệu.\n\n"

            "Mục tiêu chính của nghiên cứu là dự đoán ở giai đoạn sớm liệu một video mới được đăng tải có khả năng lọt vào Top-K video thịnh hành (được xấp xỉ bằng danh sách mostPopular của YouTube) trong một khoảng thời gian tương lai hay không. Ngoài ra, nghiên cứu còn nhằm giải thích lý do tại sao một số video có sự tăng trưởng đột ngột (burst behavior) bằng cách phân tích các mẫu tương tác theo thời gian và các tín hiệu bên ngoài.\n\n"

            "Phương pháp được đề xuất kết hợp dự báo chuỗi thời gian (time-series forecasting) và phân loại xu hướng (trending classification) trong một mô hình học đa nhiệm có nhận biết burst (burst-aware multi-task learning), sử dụng các tín hiệu tương tác sớm như lượt xem (views), lượt thích (likes), và bình luận (comments). Để cải thiện khả năng diễn giải, nghiên cứu áp dụng Timeshap-based explainability, cho phép xác định những khoảng thời gian và đặc trưng nào đóng góp nhiều nhất vào dự đoán xu hướng.\n\n"
            
            
        )

    # dataset
    if any(k in m for k in ["dataset", "du lieu", "data", "bao nhieu", "cot", "columns"]):
        return fmt_dataset()

    # proxy label
    if any(k in m for k in ["proxy", "label", "threshold", "top 10", "gan nhan"]):
        return fmt_proxy_label()

    # results / comparison
    if any(k in m for k in ["ket qua", "results", "metric", "auc", "f1", "roc", "so sanh", "comparison", "bang so sanh"]):
        return fmt_comparison()

    # week2
    if any(k in m for k in ["week2", "week 2", "ablation", "weighted", "burst"]):
        return fmt_week2()

    # shap / importance
    if any(k in m for k in ["shap", "importance", "feature"]):
        return fmt_shap()

    # limitations
    if any(k in m for k in ["limitation", "han che", "gioi han"]):
        return fmt_limitations()

    # next steps


    return "Mình chưa hiểu câu này.\n\n" + HELP

# ===================== Routes =====================
@app.get("/outputs/<path:filename>")
def outputs_files(filename):
    return send_from_directory(OUTPUTS, filename)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/chat")
def chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    reply = offline_chat(message)

    # reply có thể là string hoặc dict
    if isinstance(reply, dict):
        return jsonify(reply)
    return jsonify({"type":"text", "text": reply})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, debug=False)