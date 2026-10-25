"""Few-shot examples, one per category.

These sentences are synthetic (written for the prompt, not taken from any survey).
Keep them out of exams; evaluate refuses to score overlapping text.
"""

FEW_SHOT_EXAMPLES = [
    ("อยากให้บริษัทเพิ่มวงเงินประกันสุขภาพ ตอนนี้ไม่พอค่ารักษาจริง", "welfare"),
    ("ทำงานมาสามปีแล้ว เงินเดือนแทบไม่ขยับเลย", "compensation"),
    ("หัวหน้าตัดสินใจเองทุกเรื่องโดยไม่ถามความเห็นทีมก่อน", "management"),
    ("คนในแผนกน้อยเกินไป ต้องทำงานแทนกันจนกลับบ้านดึกทุกวัน", "workload"),
    ("เก้าอี้กับคอมพิวเตอร์ในออฟฟิศเก่ามาก ใช้งานไม่สะดวก", "environment"),
    ("อยากให้มีหลักสูตรอบรมทักษะใหม่ ๆ ให้พนักงานบ้าง", "career"),
    ("เรื่องเปลี่ยนตารางกะแจ้งกระชั้นชิดมาก ไม่มีใครรู้ล่วงหน้า", "communication"),
    ("ไม่มีความคิดเห็นเพิ่มเติม", "other"),
]
