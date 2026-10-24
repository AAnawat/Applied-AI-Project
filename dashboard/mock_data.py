"""Mock employee survey data for developing the dashboard before the real models are ready."""

import numpy as np
import pandas as pd

# Department -> headcount
DEPARTMENTS = {
    "Human Resources": 25,
    "IT": 40,
    "Sales": 55,
    "Marketing": 30,
    "Finance": 25,
    "Operations": 45,
    "Production": 80,
}

CATEGORIES = [
    "Welfare & Benefits",
    "Management",
    "Work Environment",
    "Workload",
    "Career Growth",
    "Compensation",
    "Communication",
]

MONTHS = pd.period_range("2026-01", "2026-10", freq="M")

# Thai feedback templates: category -> sentiment -> examples
TEMPLATES = {
    "Welfare & Benefits": {
        "Positive": [
            "สวัสดิการประกันสุขภาพดีมาก ครอบคลุมครอบครัวด้วย",
            "ชอบที่บริษัทมีวันลาพักร้อนเยอะ",
            "อาหารกลางวันฟรีช่วยประหยัดค่าใช้จ่ายได้มาก",
        ],
        "Neutral": [
            "สวัสดิการโดยรวมก็พอใช้ได้ เหมือนบริษัทอื่นทั่วไป",
            "อยากทราบรายละเอียดสวัสดิการใหม่ปีหน้า",
            "ยังไม่ค่อยได้ใช้สวัสดิการเท่าไหร่",
        ],
        "Negative": [
            "ประกันสุขภาพวงเงินน้อยเกินไป ไม่พอค่ารักษา",
            "สวัสดิการถูกตัดลงทุกปี รู้สึกไม่มั่นคง",
            "ขั้นตอนเบิกสวัสดิการยุ่งยากและช้ามาก",
        ],
    },
    "Management": {
        "Positive": [
            "หัวหน้ารับฟังความคิดเห็นและให้คำแนะนำดี",
            "ผู้บริหารมีวิสัยทัศน์ชัดเจน ทำให้มั่นใจในอนาคต",
            "หัวหน้าให้อิสระในการทำงานและไว้ใจทีม",
        ],
        "Neutral": [
            "การบริหารงานก็เป็นไปตามปกติ",
            "มีการเปลี่ยนหัวหน้าทีมเมื่อเดือนที่แล้ว ยังต้องรอดู",
            "นโยบายใหม่ของฝ่ายบริหารยังไม่เห็นผลชัดเจน",
        ],
        "Negative": [
            "หัวหน้าสั่งงานไม่ชัดเจน เปลี่ยนใจบ่อย",
            "ผู้บริหารไม่ค่อยรับฟังพนักงานระดับล่าง",
            "การตัดสินใจล่าช้า ทำให้งานติดขัด",
        ],
    },
    "Work Environment": {
        "Positive": [
            "ออฟฟิศสะอาด บรรยากาศน่าทำงาน",
            "เพื่อนร่วมงานเป็นกันเองและช่วยเหลือกันดี",
            "ชอบที่มีพื้นที่พักผ่อนและมุมกาแฟ",
        ],
        "Neutral": [
            "สภาพแวดล้อมในการทำงานก็โอเค",
            "ที่จอดรถพอมีบ้างแต่ต้องมาเช้า",
            "ห้องประชุมพอใช้ได้ แต่บางครั้งต้องจองล่วงหน้านาน",
        ],
        "Negative": [
            "แอร์เสียบ่อยมาก ร้อนจนทำงานไม่ได้",
            "อุปกรณ์คอมพิวเตอร์เก่าและช้า",
            "ในไลน์ผลิตเสียงดังและฝุ่นเยอะ",
        ],
    },
    "Workload": {
        "Positive": [
            "ปริมาณงานเหมาะสม มีเวลาพักผ่อน",
            "ทีมช่วยกระจายงานกันได้ดี",
            "ไม่ค่อยต้องทำโอทีแล้ว รู้สึกสมดุลขึ้น",
        ],
        "Neutral": [
            "ช่วงปลายเดือนงานเยอะตามปกติ",
            "งานบางช่วงมาก บางช่วงน้อย",
            "ภาระงานพอ ๆ กับปีที่แล้ว",
        ],
        "Negative": [
            "งานเยอะเกินไป ต้องทำโอทีทุกวัน",
            "คนลาออกแต่ไม่รับคนเพิ่ม งานตกมาที่คนเหลือ",
            "เหนื่อยมาก ไม่มีเวลาให้ครอบครัว",
        ],
    },
    "Career Growth": {
        "Positive": [
            "บริษัทส่งไปอบรมบ่อย ได้พัฒนาทักษะใหม่",
            "มีเส้นทางการเติบโตในสายงานชัดเจน",
            "ได้รับมอบหมายโปรเจกต์ที่ท้าทายและได้เรียนรู้",
        ],
        "Neutral": [
            "ยังไม่แน่ใจเรื่องโอกาสเลื่อนตำแหน่ง",
            "มีคอร์สออนไลน์ให้เรียนแต่ยังไม่ได้ลอง",
            "การประเมินผลงานเป็นไปตามรอบปกติ",
        ],
        "Negative": [
            "ทำงานมาหลายปีแต่ไม่เคยได้เลื่อนตำแหน่ง",
            "ไม่มีงบอบรมหรือพัฒนาทักษะเลย",
            "เกณฑ์การประเมินผลงานไม่โปร่งใส",
        ],
    },
    "Compensation": {
        "Positive": [
            "เงินเดือนเหมาะสมกับงานที่ทำ",
            "โบนัสปีนี้ดีกว่าที่คาดไว้",
            "ขึ้นเงินเดือนตรงตามผลงาน รู้สึกยุติธรรม",
        ],
        "Neutral": [
            "ค่าตอบแทนอยู่ในระดับตลาด",
            "รอดูผลการปรับเงินเดือนรอบหน้า",
            "ค่าโอทีคิดตามระเบียบปกติ",
        ],
        "Negative": [
            "เงินเดือนต่ำกว่าตลาด เมื่อเทียบกับบริษัทอื่น",
            "ไม่ได้ขึ้นเงินเดือนมาสองปีแล้ว",
            "โบนัสลดลงทั้งที่ทำงานหนักขึ้น",
        ],
    },
    "Communication": {
        "Positive": [
            "มีประชุมอัปเดตข่าวสารบริษัททุกเดือน ดีมาก",
            "สื่อสารระหว่างทีมราบรื่น",
            "ผู้บริหารแจ้งทิศทางบริษัทอย่างโปร่งใส",
        ],
        "Neutral": [
            "ได้รับข่าวสารผ่านอีเมลเป็นหลัก",
            "บางเรื่องรู้จากเพื่อนก่อนประกาศทางการ",
            "การสื่อสารระหว่างแผนกพอใช้ได้",
        ],
        "Negative": [
            "ข้อมูลสำคัญไม่ถูกส่งต่อ ทำให้ทำงานผิดพลาด",
            "แต่ละแผนกไม่คุยกัน ทำงานซ้ำซ้อน",
            "ประกาศนโยบายกะทันหันโดยไม่อธิบายเหตุผล",
        ],
    },
}

CATEGORY_WEIGHTS = [0.15, 0.18, 0.14, 0.16, 0.12, 0.13, 0.12]

# Built-in "stories" so the dashboard has something to find
CATEGORY_BIAS = {"Compensation": -0.3, "Welfare & Benefits": 0.2}
DEPT_CATEGORY_BIAS = {
    ("Production", "Workload"): -1.2,
    ("Production", "Work Environment"): -1.0,
    ("Production", "Compensation"): -0.5,
    ("Sales", "Management"): -0.4,
    ("Sales", "Compensation"): 0.4,
    ("IT", "Career Growth"): 0.9,
    ("IT", "Work Environment"): 0.5,
    ("IT", "Workload"): -0.4,
    ("Human Resources", "Communication"): 0.6,
    ("Finance", "Career Growth"): -0.5,
    ("Operations", "Communication"): -0.7,
}
# Per-month drift of the sentiment score (Sales getting worse, Marketing improving)
DEPT_TREND = {"Sales": -0.2, "Marketing": 0.10}


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def generate_mock_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for month_idx, month in enumerate(MONTHS):
        for dept, headcount in DEPARTMENTS.items():
            n = rng.binomial(headcount, 0.7)
            for _ in range(n):
                category = rng.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
                z = (
                    0.2
                    + CATEGORY_BIAS.get(category, 0)
                    + DEPT_CATEGORY_BIAS.get((dept, category), 0)
                    + DEPT_TREND.get(dept, 0) * month_idx
                    + rng.normal(0, 0.3)
                )
                p_pos = 0.7 * _sigmoid(z)
                p_neg = 0.7 * _sigmoid(-z)
                sentiment = rng.choice(["Positive", "Neutral", "Negative"], p=[p_pos, 0.3, p_neg])
                day = rng.integers(1, month.days_in_month + 1)
                rows.append(
                    {
                        "date": pd.Timestamp(month.year, month.month, day),
                        "department": dept,
                        "feedback": rng.choice(TEMPLATES[category][sentiment]),
                        "sentiment": sentiment,
                        "category": category,
                        "confidence": round(rng.uniform(0.55, 0.98), 2),
                    }
                )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.insert(0, "response_id", [f"R{i:05d}" for i in range(1, len(df) + 1)])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df
