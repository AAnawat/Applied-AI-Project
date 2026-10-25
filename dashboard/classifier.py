"""Saved-model sentiment and temporary keyword topics for the Upload page."""

import pandas as pd
import streamlit as st

from sentiment.model import load_model, predict_sentiment

CATEGORY_KEYWORDS = {
    "Welfare & Benefits": ["สวัสดิการ", "ประกัน", "ลาพักร้อน", "วันลา", "อาหาร"],
    "Management": ["หัวหน้า", "ผู้บริหาร", "บริหาร", "นโยบาย", "ตัดสินใจ"],
    "Work Environment": ["ออฟฟิศ", "แอร์", "อุปกรณ์", "สถานที่", "บรรยากาศ", "เพื่อนร่วมงาน", "ที่จอดรถ"],
    "Workload": ["งานเยอะ", "โอที", "ภาระงาน", "เหนื่อย", "ปริมาณงาน"],
    "Career Growth": ["อบรม", "เลื่อนตำแหน่ง", "พัฒนา", "เติบโต", "ประเมิน"],
    "Compensation": ["เงินเดือน", "โบนัส", "ค่าตอบแทน", "ค่าจ้าง"],
    "Communication": ["สื่อสาร", "ประชุม", "ข่าวสาร", "แจ้ง", "ประกาศ"],
}


def _category(text: str) -> str:
    hits = {cat: sum(k in text for k in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "Other"


@st.cache_resource(show_spinner="Loading sentiment model...")
def cached_sentiment_model():
    return load_model()


def classify(texts: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with `sentiment`, `category` and `confidence` for each text."""
    result = predict_sentiment(texts, cached_sentiment_model())
    result["category"] = texts.astype(str).map(_category)
    return result
