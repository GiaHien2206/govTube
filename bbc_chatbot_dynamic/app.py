from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_bbc"
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR.resolve())
)

BLOCKED_TERMS = [
    "danh sach file", "list file", "output file", "sheet", "csv", "xlsx",
    "folder", "thu muc", "file output", "trong file output", "co nhung file nao",
    "file nao", "ten file", "show files", "liet ke file", "sheets nao", "workbook"
]

QUICK_QUESTIONS = [
    {"label": "Chủ đề nhóm đang làm là gì?", "intent": "topic"},
    {"label": "Dataset là gì?", "intent": "dataset"},
    {"label": "Target top 10% được định nghĩa như thế nào?", "intent": "target_definition"},
    {"label": "Metadata-only là gì?", "intent": "metadata_definition"},
    {"label": "Full-feature là gì?", "intent": "full_feature_definition"},
    {"label": "Dữ liệu mất cân bằng như thế nào?", "intent": "class_imbalance"},
    {"label": "Các feature đang dùng là gì?", "intent": "features_used"},
    {"label": "Kết quả metadata-only là gì?", "intent": "metadata_results"},
    {"label": "Kết quả full-feature là gì?", "intent": "full_results"},
    {"label": "So sánh model ra sao?", "intent": "model_comparison"},
    {"label": "Model tốt nhất là gì?", "intent": "best_model"},
    {"label": "Feature importance là gì?", "intent": "feature_importance"},
    {"label": "Top feature quan trọng nhất là gì?", "intent": "top_feature"},
    {"label": "Confusion matrix cho biết gì?", "intent": "confusion_matrix"},
    {"label": "Sự khác nhau giữa metadata-only và full-feature là gì?", "intent": "compare_settings"},
    {"label": "Kết luận của bài là gì?", "intent": "conclusion"},
]
VALID_INTENTS = {
    "topic",
    "dataset",
    "target_definition",
    "metadata_definition",
    "full_feature_definition",
    "class_imbalance",
    "features_used",
    "metadata_results",
    "full_results",
    "model_comparison",
    "best_model",
    "feature_importance",
    "top_feature",
    "confusion_matrix",
    "compare_settings",
    "conclusion",
    "blocked",
    "unknown",
}

INTENT_PATTERNS = {
    "topic": [
        "chu de nhom dang lam la gi",
        "chu de nhom dang lam",
        "de tai la gi",
        "topic la gi",
        "nhom dang lam gi",
    ],
    "dataset": [
        "dataset la gi",
        "bo du lieu la gi",
        "du lieu la gi",
        "data la gi",
    ],
    "target_definition": [
        "target top 10 duoc dinh nghia nhu the nao",
        "target top 10 la gi",
        "top 10 duoc dinh nghia nhu the nao",
        "90th percentile la gi",
        "target la gi",
    ],
    "metadata_definition": [
        "metadata only la gi",
        "metadata-only la gi",
        "metadata only la sao",
        "metadata-only la sao",
        "dinh nghia metadata only",
        "metadata only co nghia la gi",
        "metadata la gi",
    ],
    "full_feature_definition": [
        "full feature la gi",
        "full-feature la gi",
        "full feature la sao",
        "full-feature la sao",
        "dinh nghia full feature",
        "full feature co nghia la gi",
    ],
    "class_imbalance": [
        "du lieu mat can bang nhu the nao",
        "du lieu mat can bang",
        "mat can bang nhu the nao",
        "class imbalance",
        "imbalanced data",
        "class distribution",
    ],
    "features_used": [
        "cac feature dang dung la gi",
        "feature dang dung la gi",
        "cac bien dang dung la gi",
        "features dang dung la gi",
        "features la gi",
    ],
    "metadata_results": [
        "ket qua metadata only la gi",
        "ket qua metadata-only la gi",
        "metadata only result",
        "metadata-only result",
        "ket qua metadata only",
        "ket qua metadata-only",
    ],
    "full_results": [
        "ket qua full feature la gi",
        "ket qua full-feature la gi",
        "full feature result",
        "full-feature result",
        "ket qua full feature",
        "ket qua full-feature",
    ],
    "model_comparison": [
        "so sanh model ra sao",
        "so sanh model",
        "model comparison",
        "compare model",
        "cac model ra sao",
    ],
    "best_model": [
        "model tot nhat la gi",
        "best model la gi",
        "mo hinh tot nhat la gi",
        "model nao tot nhat",
    ],
    "feature_importance": [
        "feature importance la gi",
        "importance la gi",
        "do quan trong feature la gi",
        "feature importance",
    ],
    "top_feature": [
        "top feature quan trong nhat la gi",
        "top feature la gi",
        "feature quan trong nhat",
        "most important feature",
        "feature nao quan trong nhat",
    ],
    "confusion_matrix": [
        "confusion matrix cho biet gi",
        "confusion matrix la gi",
        "ma tran nham lan la gi",
        "confusion matrix",
    ],
    "compare_settings": [
        "su khac nhau giua metadata only va full feature la gi",
        "khac nhau giua metadata only va full feature",
        "metadata only va full feature khac nhau nhu the nao",
        "compare metadata only and full feature",
    ],
    "conclusion": [
        "ket luan cua bai la gi",
        "ket luan la gi",
        "conclusion la gi",
        "conclusion",
        "tong ket bai",
    ],
}
RUNTIME: Dict[str, Any] = {}


class DataLoadError(Exception):
    pass


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    text = strip_accents(str(text)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def has_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def answer_card(title: str, body_html: str) -> str:
    return f"""
    <div class="answer-card">
      <div class="answer-title">{html.escape(title)}</div>
      {body_html}
    </div>
    """


def clean_feature_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\.\d+$", "", name)
    return name


def prettify_image_name(filename: str) -> str:
    stem = Path(filename).stem.lower()

    if "full_feature_random_forest_cm" in stem:
        return "Confusion Matrix - Random Forest (Full-feature)"
    if "full_feature_xgboost_cm" in stem:
        return "Confusion Matrix - XGBoost (Full-feature)"
    if "full_feature_logistic_regression_cm" in stem:
        return "Confusion Matrix - Logistic Regression (Full-feature)"
    if "metadata_only_random_forest_cm" in stem:
        return "Confusion Matrix - Random Forest (Metadata-only)"
    if "metadata_only_xgboost_cm" in stem:
        return "Confusion Matrix - XGBoost (Metadata-only)"
    if "metadata_only_logistic_regression_cm" in stem:
        return "Confusion Matrix - Logistic Regression (Metadata-only)"
    if "xgb_full_feature_importance" in stem:
        return "Top Feature Importance - XGBoost (Full-feature)"

    return stem.replace("_", " ").replace("-", " ").title()


def image_sort_key(filename: str) -> tuple[int, str]:
    stem = filename.lower()
    order = 99

    if "full_feature_random_forest_cm" in stem:
        order = 1
    elif "xgb_full_feature_importance" in stem:
        order = 2
    elif "full_feature_xgboost_cm" in stem:
        order = 3
    elif "full_feature_logistic_regression_cm" in stem:
        order = 4
    elif "metadata_only_xgboost_cm" in stem:
        order = 5
    elif "metadata_only_logistic_regression_cm" in stem:
        order = 6
    elif "metadata_only_random_forest_cm" in stem:
        order = 7

    return (order, stem)


def get_gallery_images() -> List[Dict[str, str]]:
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    images = sorted(images, key=lambda p: image_sort_key(p.name))

    result = []
    for img in images:
        stem = img.stem.lower()
        tag = "other"
        if "full_feature" in stem:
            tag = "full-feature"
        elif "metadata_only" in stem:
            tag = "metadata-only"
        elif "importance" in stem:
            tag = "importance"

        result.append({
            "filename": img.name,
            "title": prettify_image_name(img.name),
            "tag": tag,
            "url": f"/outputs-image/{img.name}"
        })
    return result


def find_excel_with_results() -> Path:
    xlsx_files = sorted(OUTPUT_DIR.glob("*.xlsx"))
    for fp in xlsx_files:
        try:
            xl = pd.ExcelFile(fp)
            if "model_results" in xl.sheet_names:
                return fp
        except Exception:
            continue
    raise DataLoadError("Không tìm thấy file Excel có sheet model_results trong outputs_bbc.")


def find_feature_source() -> tuple[str, Path]:
    csv_files = sorted(OUTPUT_DIR.glob("*.csv"))
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, nrows=5)
            cols = {normalize_key(c): c for c in df.columns}
            if "istop10view" in cols and ("viewcount" in cols or "views" in cols):
                return ("csv", fp)
        except Exception:
            continue

    xlsx_files = sorted(OUTPUT_DIR.glob("*.xlsx"))
    for fp in xlsx_files:
        try:
            xl = pd.ExcelFile(fp)
            if "bbc_model_features" in xl.sheet_names:
                return ("xlsx_sheet", fp)
        except Exception:
            continue

    raise DataLoadError("Không tìm thấy CSV hoặc sheet bbc_model_features phù hợp trong outputs_bbc.")


def load_runtime_data() -> Dict[str, Any]:
    if not OUTPUT_DIR.exists():
        raise DataLoadError("Không tìm thấy thư mục outputs_bbc. Hãy đặt outputs_bbc cùng cấp với app.py.")

    excel_path = find_excel_with_results()
    feature_source_type, feature_source_path = find_feature_source()

    model_results = pd.read_excel(excel_path, sheet_name="model_results")
    required_cols = {"Feature Setting", "Model", "Accuracy", "Precision", "Recall", "F1"}
    if not required_cols.issubset(model_results.columns):
        raise DataLoadError("Sheet model_results thiếu các cột bắt buộc.")

    model_results = model_results[["Feature Setting", "Model", "Accuracy", "Precision", "Recall", "F1"]].copy()

    feature_importance = None
    xl = pd.ExcelFile(excel_path)
    if "xgb_full_importance" in xl.sheet_names:
        fi = pd.read_excel(excel_path, sheet_name="xgb_full_importance")
        if {"Feature", "Importance"}.issubset(fi.columns):
            feature_importance = fi[["Feature", "Importance"]].copy()
            feature_importance = feature_importance.sort_values("Importance", ascending=False).reset_index(drop=True)

    if feature_source_type == "csv":
        model_features = pd.read_csv(feature_source_path)
    else:
        model_features = pd.read_excel(feature_source_path, sheet_name="bbc_model_features")

    norm_cols = {normalize_key(c): c for c in model_features.columns}
    target_col = norm_cols.get("istop10view") or norm_cols.get("performance")
    view_col = norm_cols.get("viewcount") or norm_cols.get("views")

    if target_col is None or view_col is None:
        raise DataLoadError("Không tìm thấy cột is_top10_view/performance hoặc viewCount/views trong dữ liệu feature.")

    model_features[target_col] = pd.to_numeric(model_features[target_col], errors="coerce").fillna(0).astype(int)
    model_features[view_col] = pd.to_numeric(model_features[view_col], errors="coerce")

    threshold = int(model_features[view_col].quantile(0.90))
    class_counts = model_features[target_col].value_counts().sort_index().to_dict()
    total_rows = len(model_features)

    raw_like_cols = {
        "videoid", "title", "published", "publishtime", "publishedat", "publish_time",
        "viewcount", "views", "likecount", "commentcount", "duration", "performance",
        "istop10view", "top10threshold", "uploadday", "uploadhour"
    }

    cleaned_features = []
    seen = set()
    for c in model_features.columns:
        cleaned = clean_feature_name(c)
        if normalize_key(cleaned) in raw_like_cols:
            continue
        if cleaned not in seen:
            cleaned_features.append(cleaned)
            seen.add(cleaned)

    interaction_names = {"likeCount", "commentCount", "engagement", "like_comment_ratio"}
    metadata_features = [f for f in cleaned_features if f not in interaction_names]
    full_features = cleaned_features

    meta_df = model_results[model_results["Feature Setting"].astype(str).str.lower().str.contains("metadata")].copy()
    full_df = model_results[model_results["Feature Setting"].astype(str).str.lower().str.contains("full")].copy()

    meta_best = meta_df.sort_values("F1", ascending=False).iloc[0].to_dict() if not meta_df.empty else None
    full_best = full_df.sort_values("F1", ascending=False).iloc[0].to_dict() if not full_df.empty else None

    gallery_images = get_gallery_images()

    return {
        "excel_path": excel_path,
        "feature_source_type": feature_source_type,
        "feature_source_path": feature_source_path,
        "model_results": model_results,
        "feature_importance": feature_importance,
        "model_features": model_features,
        "threshold": threshold,
        "total_rows": total_rows,
        "class_counts": class_counts,
        "metadata_features": metadata_features,
        "full_features": full_features,
        "meta_df": meta_df,
        "full_df": full_df,
        "meta_best": meta_best,
        "full_best": full_best,
        "gallery_images": gallery_images,
    }


def reload_data() -> None:
    global RUNTIME
    RUNTIME = load_runtime_data()


def chips_html(items: List[str]) -> str:
    return "<div class='chips'>" + "".join(
        f"<span class='chip'>{html.escape(str(x))}</span>" for x in items
    ) + "</div>"


def model_table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>Không có dữ liệu.</p>"

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = ""
    for _, r in df.iterrows():
        rows += (
            "<tr>"
            f"<td>{html.escape(str(r['Model']))}</td>"
            f"<td>{float(r['Accuracy']):.4f}</td>"
            f"<td>{float(r['Precision']):.4f}</td>"
            f"<td>{float(r['Recall']):.4f}</td>"
            f"<td>{float(r['F1']):.4f}</td>"
            "</tr>"
        )
    return f"<table class='answer-table'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def detect_intent(question: str) -> str:
    q = normalize_text(question)

    blocked_terms = [normalize_text(x) for x in BLOCKED_TERMS]
    if has_any(q, blocked_terms):
        return "blocked"

    # chấm điểm để tránh match nhầm
    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        best_score = 0
        for raw_p in patterns:
            p = normalize_text(raw_p)

            # match tuyệt đối
            if q == p:
                best_score = max(best_score, 100)
                continue

            # match nguyên cụm
            if f" {p} " in f" {q} ":
                best_score = max(best_score, 85)
                continue

            # q chứa pattern hoặc pattern chứa q
            if p in q:
                best_score = max(best_score, 70)
                continue
            if len(q) >= 8 and q in p:
                best_score = max(best_score, 55)
                continue

        if best_score > 0:
            scores[intent] = best_score

    if not scores:
        return "unknown"

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # ngưỡng an toàn, dưới mức này thì coi như không hiểu rõ
    if best_score < 55:
        return "unknown"

    return best_intent

def answer_query(question: str, forced_intent: str | None = None) -> str:
    intent = forced_intent if forced_intent in VALID_INTENTS else detect_intent(question)

    if intent == "unknown":
        return answer_card(
            "Mình chưa hiểu rõ câu hỏi",
            "<p>Hãy chọn một câu hỏi nhanh bên trên hoặc hỏi theo các mẫu rõ hơn như:</p>"
            + chips_html([
                "Metadata-only là gì?",
                "Full-feature là gì?",
                "Kết quả metadata-only là gì?",
                "Kết quả full-feature là gì?",
                "Top feature quan trọng nhất là gì?",
                "Dữ liệu mất cân bằng như thế nào?",
                "Kết luận của bài là gì?"
            ])
        )

    if intent == "blocked":
        return answer_card(
            "Không hỗ trợ câu hỏi này",
            "<p>Ứng dụng này không hiển thị danh sách file, tên file, sheet hay cấu trúc thư mục trong outputs_bbc.</p>"
        )

    if intent == "topic":
        return answer_card(
            "Chủ đề nhóm",
            "<p>Nhóm đang làm bài toán <b>classifying top 10% most-viewed BBC News YouTube videos using machine learning</b>.</p>"
            "<p>Mục tiêu là phân loại video có thuộc nhóm top 10% lượt xem hay không, dựa trên metadata và interaction-based features.</p>"
        )

    if intent == "dataset":
        total_rows = RUNTIME["total_rows"]
        threshold = RUNTIME["threshold"]
        c0 = RUNTIME["class_counts"].get(0, 0)
        c1 = RUNTIME["class_counts"].get(1, 0)

        body = f"""
        <div class="two-col">
          <div><h4>Tổng số dòng</h4><p><b>{total_rows:,}</b></p></div>
          <div><h4>Ngưỡng top 10%</h4><p><b>{threshold:,}</b> views</p></div>
          <div><h4>Class 0</h4><p><b>{c0:,}</b></p></div>
          <div><h4>Class 1</h4><p><b>{c1:,}</b></p></div>
        </div>
        <p>Dataset gồm các video BBC News với các biến như title, published time, duration, viewCount, likeCount, commentCount và các feature đã được xử lý để huấn luyện model.</p>
        """
        return answer_card("Dataset", body)

    if intent == "target_definition":
        threshold = RUNTIME["threshold"]
        return answer_card(
            "Target top 10%",
            f"<p>Target được định nghĩa bằng <b>90th percentile</b> của view counts.</p>"
            f"<p>Ngưỡng hiện tại là <b>{threshold:,}</b> views.</p>"
            "<p>Video có views lớn hơn hoặc bằng ngưỡng này được gán nhãn <b>1</b>, còn lại gán nhãn <b>0</b>.</p>"
        )

    if intent == "metadata_definition":
        metadata_features = RUNTIME["metadata_features"]
        body = (
            "<p><b>Metadata-only</b> là setting chỉ dùng các biến có sẵn tại hoặc gần thời điểm đăng video, "
            "không dùng tín hiệu tương tác như likes hay comments.</p>"
            "<p>Nó phù hợp hơn cho <b>early-stage classification</b> vì không phụ thuộc vào phản hồi của người xem sau khi video đã đăng.</p>"
            "<h4>Các feature thuộc metadata-only</h4>"
            + chips_html(metadata_features)
        )
        return answer_card("Metadata-only là gì?", body)

    if intent == "full_feature_definition":
        full_features = RUNTIME["full_features"]
        body = (
            "<p><b>Full-feature</b> là setting dùng toàn bộ feature, bao gồm cả metadata features và interaction-based features.</p>"
            "<p>Nó thường cho kết quả mạnh hơn vì có thêm các biến phản ánh phản hồi của người xem như "
            "<b>likeCount</b>, <b>commentCount</b>, <b>engagement</b> và <b>like_comment_ratio</b>.</p>"
            "<h4>Các feature thuộc full-feature</h4>"
            + chips_html(full_features)
        )
        return answer_card("Full-feature là gì?", body)

    if intent == "class_imbalance":
        c0 = RUNTIME["class_counts"].get(0, 0)
        c1 = RUNTIME["class_counts"].get(1, 0)
        total = max(c0 + c1, 1)
        p0 = c0 / total * 100
        p1 = c1 / total * 100

        body = f"""
        <table class="mini-table">
          <thead><tr><th>Class</th><th>Count</th><th>Percentage</th></tr></thead>
          <tbody>
            <tr><td>0 (Not top 10%)</td><td>{c0:,}</td><td>{p0:.1f}%</td></tr>
            <tr><td>1 (Top 10%)</td><td>{c1:,}</td><td>{p1:.1f}%</td></tr>
          </tbody>
        </table>
        <p>Dataset bị lệch lớp nên không nên chỉ nhìn accuracy, mà phải xem thêm precision, recall và F1-score.</p>
        """
        return answer_card("Class imbalance", body)

    if intent == "features_used":
        metadata_features = RUNTIME["metadata_features"]
        full_features = RUNTIME["full_features"]

        body = (
            "<h4>Metadata-only features</h4>"
            + chips_html(metadata_features)
            + "<h4 style='margin-top:14px'>Full-feature setting</h4>"
            + chips_html(full_features)
            + "<p style='margin-top:10px'>Metadata-only chỉ dùng các biến có sẵn gần thời điểm đăng. "
              "Full-feature thêm cả interaction-based variables để phân loại mạnh hơn.</p>"
        )
        return answer_card("Các feature đang dùng", body)

    if intent == "metadata_results":
        return answer_card("Kết quả metadata-only", model_table_html(RUNTIME["meta_df"]))

    if intent == "full_results":
        return answer_card("Kết quả full-feature", model_table_html(RUNTIME["full_df"]))

    if intent == "model_comparison":
        return answer_card("So sánh model", model_table_html(RUNTIME["model_results"]))

    if intent == "best_model":
        best = RUNTIME["full_best"]
        body = f"""
        <div class="two-col">
          <div><h4>Model</h4><p><b>{html.escape(str(best['Model']))}</b></p></div>
          <div><h4>Accuracy</h4><p><b>{float(best['Accuracy']):.4f}</b></p></div>
          <div><h4>Precision</h4><p><b>{float(best['Precision']):.4f}</b></p></div>
          <div><h4>Recall</h4><p><b>{float(best['Recall']):.4f}</b></p></div>
          <div><h4>F1</h4><p><b>{float(best['F1']):.4f}</b></p></div>
        </div>
        <p>Model tốt nhất overall hiện tại là model có F1-score cao nhất trong <b>full-feature setting</b>.</p>
        """
        return answer_card("Best model", body)

    if intent == "feature_importance":
        fi = RUNTIME["feature_importance"]
        if fi is None or fi.empty:
            return answer_card("Feature importance", "<p>Không tìm thấy sheet xgb_full_importance trong file Excel.</p>")

        top = fi.head(10)
        rows = "".join(
            f"<tr><td>{html.escape(str(r['Feature']))}</td><td>{float(r['Importance']):.6f}</td></tr>"
            for _, r in top.iterrows()
        )
        return answer_card(
            "Feature importance",
            f"<table class='answer-table'><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if intent == "top_feature":
        fi = RUNTIME["feature_importance"]
        if fi is None or fi.empty:
            return answer_card("Top feature", "<p>Không tìm thấy dữ liệu feature importance.</p>")

        top = fi.head(5)
        rows = "".join(
            f"<tr><td>{html.escape(str(r['Feature']))}</td><td>{float(r['Importance']):.6f}</td></tr>"
            for _, r in top.iterrows()
        )
        return answer_card(
            "Top feature quan trọng nhất",
            f"<table class='answer-table'><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if intent == "confusion_matrix":
        best = RUNTIME["full_best"]
        body = (
            "<p>Confusion matrix cho biết số lượng dự đoán đúng và sai theo từng nhóm:</p>"
            "<table class='mini-table'>"
            "<thead><tr><th>Thành phần</th><th>Ý nghĩa</th></tr></thead>"
            "<tbody>"
            "<tr><td>True Positive</td><td>Dự đoán đúng video thuộc top 10%</td></tr>"
            "<tr><td>True Negative</td><td>Dự đoán đúng video không thuộc top 10%</td></tr>"
            "<tr><td>False Positive</td><td>Dự đoán nhầm video vào top 10%</td></tr>"
            "<tr><td>False Negative</td><td>Bỏ sót video thật sự thuộc top 10%</td></tr>"
            "</tbody></table>"
            f"<p>Trong paper nên dùng confusion matrix của <b>{html.escape(str(best['Model']))}</b> ở <b>full-feature</b>.</p>"
        )
        return answer_card("Confusion matrix", body)

    if intent == "compare_settings":
        meta_best = RUNTIME["meta_best"]
        full_best = RUNTIME["full_best"]
        body = f"""
        <table class="mini-table">
          <thead>
            <tr>
              <th>Setting</th>
              <th>Best Model</th>
              <th>F1-score</th>
              <th>Ý nghĩa</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Metadata-only</td>
              <td>{html.escape(str(meta_best['Model']))}</td>
              <td>{float(meta_best['F1']):.4f}</td>
              <td>Dùng metadata và title-related features, phù hợp hơn với early-stage classification.</td>
            </tr>
            <tr>
              <td>Full-feature</td>
              <td>{html.escape(str(full_best['Model']))}</td>
              <td>{float(full_best['F1']):.4f}</td>
              <td>Thêm likeCount, commentCount, engagement và like_comment_ratio nên phân loại mạnh hơn rõ rệt.</td>
            </tr>
          </tbody>
        </table>
        """
        return answer_card("So sánh metadata-only và full-feature", body)

    if intent == "conclusion":
        meta_best = RUNTIME["meta_best"]
        full_best = RUNTIME["full_best"]
        body = (
            f"<p>Ở metadata-only, <b>{html.escape(str(meta_best['Model']))}</b> có F1 tốt nhất là <b>{float(meta_best['F1']):.4f}</b>.</p>"
            f"<p>Ở full-feature, <b>{html.escape(str(full_best['Model']))}</b> là model tốt nhất overall với F1 = <b>{float(full_best['F1']):.4f}</b>.</p>"
            "<p>Khi thêm likeCount, commentCount, engagement và like_comment_ratio, khả năng phân loại video top 10% tăng lên rõ rệt.</p>"
        )
        return answer_card("Kết luận", body)

    return answer_card(
        "Bạn có thể hỏi gì?",
        "<p>Mình đang đọc dữ liệu trực tiếp từ <b>outputs_bbc</b> và chỉ trả lời về nội dung nghiên cứu, dataset, feature và kết quả mô hình.</p>"
        + chips_html([
            "Metadata-only là gì?",
            "Full-feature là gì?",
            "Kết quả metadata-only là gì?",
            "Kết quả full-feature là gì?",
            "Top feature quan trọng nhất là gì?",
            "Kết luận của bài là gì?"
        ])
        + "<p style='margin-top:10px'>Hãy bấm một câu hỏi nhanh hoặc nhập câu hỏi cụ thể hơn.</p>"
    )


@app.route("/")
def index():
    reload_data()
    status = {
        "rows": RUNTIME.get("total_rows", 0),
        "threshold": RUNTIME.get("threshold", 0),
        "images": len(RUNTIME.get("gallery_images", [])),
    }
    return render_template(
        "index.html",
        quick_questions=QUICK_QUESTIONS,
        status=status,
        gallery_images=RUNTIME.get("gallery_images", [])
    )


@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    forced_intent = str(payload.get("intent", "")).strip()

    if not question and not forced_intent:
        return jsonify({"ok": False, "error": "Vui lòng nhập câu hỏi."}), 400

    try:
        reload_data()
        answer_html = answer_query(question or forced_intent, forced_intent=forced_intent)
        return jsonify({"ok": True, "answer_html": answer_html})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Không đọc được dữ liệu từ outputs_bbc: {e}"}), 500


@app.route("/outputs-image/<path:filename>")
def outputs_image(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    print(f"[INFO] BASE_DIR      = {BASE_DIR}")
    print(f"[INFO] TEMPLATE_DIR = {TEMPLATE_DIR}")
    print(f"[INFO] OUTPUT_DIR   = {OUTPUT_DIR}")

    try:
        reload_data()
        print("[INFO] Đã đọc dữ liệu outputs_bbc thành công.")
        print(f"[INFO] Tổng ảnh tìm thấy: {len(RUNTIME.get('gallery_images', []))}")
    except Exception as e:
        print(f"[ERROR] {e}")

    app.run(host="127.0.0.1", port=3000, debug=True)