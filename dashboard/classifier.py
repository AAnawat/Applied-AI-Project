"""Placeholder classifier used by the Upload page.

Keyword matching only, so the upload flow works end to end. Replace `classify`
with the real TF-IDF + Logistic Regression sentiment model and the LLM topic
categorizer; keep the same return columns.
"""

import pandas as pd

POSITIVE_WORDS = ["ดี", "ชอบ", "ขอบคุณ", "เหมาะสม", "ราบรื่น", "ประทับใจ", "สะอาด", "ยุติธรรม", "โปร่งใส", "ช่วยเหลือ"]
NEGATIVE_WORDS = ["ไม่", "แย่", "เหนื่อย", "ช้า", "น้อยเกินไป", "เยอะเกินไป", "ต่ำ", "เสีย", "ยุ่งยาก", "ลดลง", "ผิดพลาด"]

CATEGORY_KEYWORDS = {
    "Welfare & Benefits": ["สวัสดิการ", "ประกัน", "ลาพักร้อน", "วันลา", "อาหาร"],
    "Management": ["หัวหน้า", "ผู้บริหาร", "บริหาร", "นโยบาย", "ตัดสินใจ"],
    "Work Environment": ["ออฟฟิศ", "แอร์", "อุปกรณ์", "สถานที่", "บรรยากาศ", "เพื่อนร่วมงาน", "ที่จอดรถ"],
    "Workload": ["งานเยอะ", "โอที", "ภาระงาน", "เหนื่อย", "ปริมาณงาน"],
    "Career Growth": ["อบรม", "เลื่อนตำแหน่ง", "พัฒนา", "เติบโต", "ประเมิน"],
    "Compensation": ["เงินเดือน", "โบนัส", "ค่าตอบแทน", "ค่าจ้าง"],
    "Communication": ["สื่อสาร", "ประชุม", "ข่าวสาร", "แจ้ง", "ประกาศ"],
}


def _sentiment(text: str) -> tuple[str, float]:
    score = sum(w in text for w in POSITIVE_WORDS) - 1.5 * sum(w in text for w in NEGATIVE_WORDS)
    if score > 0:
        return "Positive", min(0.6 + 0.1 * score, 0.95)
    if score < 0:
        return "Negative", min(0.6 + 0.1 * -score, 0.95)
    return "Neutral", 0.5


def _category(text: str) -> str:
    hits = {cat: sum(k in text for k in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "Other"


def classify(texts: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with `sentiment`, `category` and `confidence` for each text."""
    results = []
    for text in texts.fillna("").astype(str):
        sentiment, confidence = _sentiment(text)
        results.append({"sentiment": sentiment, "category": _category(text), "confidence": round(confidence, 2)})
    return pd.DataFrame(results, index=texts.index)
