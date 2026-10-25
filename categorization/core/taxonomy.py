"""Single-label topic taxonomy. Codes are what category_predictions stores."""

CATEGORIES = {
    "welfare": {
        "name": "Welfare & Benefits",
        "description": "สวัสดิการ ประกันสุขภาพ ค่ารักษาพยาบาล วันลา อาหาร รถรับส่ง กิจกรรมพนักงาน",
    },
    "compensation": {
        "name": "Compensation",
        "description": "เงินเดือน โบนัส ค่าตอบแทน ค่าล่วงเวลา การขึ้นเงินเดือน",
    },
    "management": {
        "name": "Management",
        "description": "หัวหน้างาน ผู้บริหาร การตัดสินใจ การมอบหมายงาน นโยบาย ความยุติธรรมของผู้บังคับบัญชา",
    },
    "workload": {
        "name": "Workload",
        "description": "ปริมาณงาน ภาระงาน ชั่วโมงทำงาน โอที ความเร่งรีบ คนไม่พอ",
    },
    "environment": {
        "name": "Work Environment",
        "description": "สถานที่ทำงาน อุปกรณ์ ความสะอาด ความปลอดภัย บรรยากาศ ความสัมพันธ์กับเพื่อนร่วมงาน",
    },
    "career": {
        "name": "Career Growth",
        "description": "การอบรม การพัฒนาทักษะ การเลื่อนตำแหน่ง การประเมินผลงาน โอกาสเติบโต",
    },
    "communication": {
        "name": "Communication",
        "description": "การสื่อสารภายในองค์กร การประชุม การแจ้งข่าวสารหรือประกาศ ความชัดเจนของข้อมูล",
    },
    "other": {
        "name": "Other",
        "description": "ไม่เข้าหมวดใดข้างต้น หรือไม่มีเนื้อหาเกี่ยวกับการทำงาน",
    },
}

CODES = list(CATEGORIES)

# Tie-breaking rules for feedback that touches more than one topic
RULES = [
    "Choose the one topic the respondent most wants the organization to address.",
    "Overtime pay or OT rates -> compensation; overtime hours or too much work -> workload.",
    "A specific manager's behaviour -> management; company-wide announcements or meetings -> communication.",
    "Performance reviews or promotion criteria -> career, unless the complaint is that a manager is unfair -> management.",
    "Tone does not decide the topic: neutral or positive feedback about a topic still gets that topic.",
    "Use other only when the feedback names no topic above, for example no comment or something unrelated to work.",
]
