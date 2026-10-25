# ทดลอง fine-tuning Typhoon สำหรับ ClimateLens

ชุดนี้เพิ่มแยกจากเว็บบน branch categorization สำหรับทดลอง **QLoRA จัดหมวดหมู่**
ใช้ `typhoon-ai/typhoon2.5-qwen3-4b` และ prompt/taxonomy เดิมของโปรเจกต์
ไม่มีการฝึกเมื่ออัปโหลดข้อมูลเข้า Streamlit และไม่แก้โมเดลที่เว็บใช้อยู่

**สถานะ:** ตรวจเครื่องมือเตรียมข้อมูลและโค้ดบน CPU ได้ แต่ยังไม่ได้รัน GPU training
ยังไม่มีข้อมูลฝึกใหม่ที่คนตรวจแล้ว และไม่มีคะแนนหลัง fine-tuning ให้รายงาน

## ไฟล์ที่เริ่มดู

- `baseline_audit.json`: ค่าที่ตรวจจากผลทดลองเดิม รวม 91/105 เทียบ 92/105
- `development_review.csv`: 13 ข้อที่ few-shot ตอบผิด ให้คนทบทวนกติกา/เฉลย
- `labelled_template.csv`: หัวตารางสำหรับข้อมูลใหม่ (ตั้งใจไม่มีตัวอย่างฝึกปลอม)
- `ClimateLens_Typhoon_QLoRA.ipynb`: ขั้นตอนสำหรับ Colab CUDA GPU
- `prepare.py`: ตรวจ/แบ่งข้อมูลพร้อม manifest และ SHA256
- `train.py`: ฝึกเฉพาะ LoRA บน frozen 4-bit NF4 base
- `evaluate.py`: เทียบ base กับ adapter ด้วย runtime/ข้อมูล/prompt เดียวกัน
- `merge.py`: ขั้น export ตัวเลือก สำหรับเครื่องที่มี VRAM เพียงพอ

## 1. ทบทวนเฉลยก่อน

ให้สมาชิกอย่างน้อยสองคนตรวจข้อกำกวมใน development_review.csv และสรุปกติกา
เช่น career vs communication และ welfare vs environment
ไม่จำเป็นต้องเปลี่ยนคำตอบให้ตรงโมเดล และห้ามถือว่าทุกข้อผิดเกิดจากโมเดลเสมอ
worksheet นี้ใช้พัฒนาเท่านั้น ไม่ใช่ข้อมูลใหม่สำหรับฝึกหรือ final test

ผลเดิม Macro F1 เฉลี่ยเฉพาะ 7 หมวดที่มีเฉลย; ไม่มี other
few-shot แก้ได้ 6 ข้อ แต่ทำข้อที่เดิมถูกให้ผิด 5 ข้อ ดีขึ้นสุทธิหนึ่งข้อ
0.55/0.62 วินาทีเป็น p95; ค่าเฉลี่ยจริงอยู่ประมาณ 0.479/0.455 วินาที
หน่วยความจำตอน inference ไม่ใช่งบ VRAM สำหรับฝึก

## 2. เตรียม CSV ใหม่ที่คนตรวจแล้ว

คอลัมน์จำเป็น: `id,feedback,category,group_id`

รหัสหมวดใช้ตาม taxonomy เท่านั้น:
`welfare, compensation, management, workload, environment, career, communication, other`

- group_id คือกลุ่มประโยคต้นแบบเดียวกัน รวมประโยคดัดแปลง/สลับคำ/เปลี่ยน sentiment
- ประโยคแต่ละกลุ่มต้องมีหมวดที่ตกลงกันชัดเจน; ไม่ตั้ง group_id ตามหมวดทั้งหมวด
- ถ้าแต่ละข้อความเป็นอิสระจริง ให้ใช้ group_id เท่ากับ id
- รวมตัวอย่าง other จริง เช่น ไม่มีความคิดเห็นหรือเรื่องนอกประเด็น
- ใช้ข้อมูลที่ได้รับอนุญาตและลบข้อมูลระบุตัวตน ไม่ใช้ผลโมเดลเป็นเฉลยโดยไม่ตรวจ
- ชุด 105 ข้อและ few-shot เดิมถูกกันออกด้วย normalized exact matching
- ระบบตรวจความคล้ายเชิงความหมายไม่ได้ คนต้องจัด group_id ให้ถูก
- ต้องมีอย่างน้อย 3 กลุ่มอิสระต่อหมวดเพื่อแบ่งได้ครบ; เป็นขั้นต่ำทางเทคนิค
  ไม่ใช่คำแนะนำว่าข้อมูลเท่านี้พอฝึกให้ดี ควรเพิ่มความหลากหลายและรายงานขนาดจริง
- สำหรับข้อมูลจริงให้เก็บใน outputs/ หรือเส้นทางที่ไม่ถูก commit

จาก repository root:

```bash
python -m categorization.finetuning.prepare --input /path/to/reviewed.csv
```

ได้ `categorization/data/finetuning/{train,validation,test}.jsonl` และ manifest.json
แบ่งแบบ stratified ตามกลุ่ม ประมาณ 80/10/10 จึงไม่ใช่สัดส่วนจำนวนแถวเป๊ะเมื่อกลุ่มใหญ่ไม่เท่ากัน
output เดิมจะไม่ถูกทับ ใช้โฟลเดอร์ใหม่เมื่อทำรุ่นข้อมูลใหม่
manifest ตรวจ SHA256 ของไฟล์และ prompt; ห้ามแก้ split หลังจากเริ่มทดลอง
training script ไม่ใช้ test สำหรับเลือก checkpoint

## 3. เปิด Colab notebook

ใช้ code bundle `outputs/climatelens_finetuning_colab.zip` ที่เตรียมไว้
ไม่มี Supabase keys, secrets.toml, model sentiment หรือข้อมูลฝึกใหม่ใน bundle
อัปโหลด CSV ที่ตรวจแล้วแยกอีกครั้งด้วยตัวเอง
การอัปโหลดเข้า Colab คือส่งข้อมูลไปยัง Google จึงใช้เฉพาะข้อมูลที่อนุญาตให้ใช้ที่นั่น

เลือก Runtime > Change runtime type > NVIDIA GPU และใช้ PyTorch CUDA >= 2.6
dependency ชุด GPU แยกจาก requirements.txt ของเว็บ ห้ามติดตั้งปน environment เว็บ
ยังไม่ได้รับรองว่า GPU ทุกรุ่นหรือ Colab ทุก runtime จะรันผ่าน และไม่รับประกันเวลาหรือ VRAM
ค่าเริ่มต้น: context training 1024, microbatch 1, accumulation 8, rank 16, LR 1e-4, 2 epochs
ทั้งหมดเป็นจุดเริ่มทดลอง ไม่ใช่ค่าที่พิสูจน์ว่าดีที่สุด
บันทึก base revision, seed, GPU, package versions และ data/prompt hashes ใน run.json
ฉบับนี้เพิ่ม `--learning-rate` และบันทึกค่าตั้งต้นใน `parameters.learning_rate` สำหรับ run ใหม่
ผลเก่าที่ไม่มีฟิลด์นี้ยังอ่านได้ และไม่ได้แก้ metadata ย้อนหลัง

Notebook รวมค่าฝึก 4 run ไว้ใน `RUNS` และใช้ `MODE="restore"` เป็นค่าเริ่มต้น
เพื่ออัปโหลด artifacts เดิมและอ่านผลโดยไม่ฝึกซ้ำ; เปลี่ยนเป็น `"train"` เมื่อต้องการทำซ้ำ
Final test และชุดท้าทายเปิดแยกด้วย `RUN_FINAL_TEST` / `RUN_CHALLENGE`
หลังแก้โค้ดให้สร้าง code bundle ใหม่ (ใช้ชื่อ output ใหม่เพื่อไม่ทับ ZIP เดิม):

```bash
python -m categorization.finetuning.package_colab --output outputs/climatelens_finetuning_colab_clean.zip
```
ฝึก loss เฉพาะคำตอบ JSON และ EOS; prompt ถูก mask ด้วย -100
ข้อความยาวเกินจะหยุดพร้อมแจ้ง ID ไม่ตัดทิ้งเงียบ ๆ

## 4. Smoke test แล้วจึงฝึกจริง

คำสั่งเหล่านี้รันบน GPU runtime ที่ติดตั้ง requirements-gpu.txt เท่านั้น:

```bash
python -m categorization.finetuning.train --output outputs/typhoon_smoke --max-steps 20
python -m categorization.finetuning.train --output outputs/typhoon_run_01
```

ต้องใช้ข้อมูลใหม่ก่อน คำสั่งจะไม่สร้างข้อมูลฝึกให้อัตโนมัติ
model download หลาย GB เริ่มหลังผ่านการตรวจข้อมูลและ GPU
smoke adapter มีไว้ตรวจ pipeline ไม่ใช้รายงาน final-test score

## 5. ประเมินอย่างยุติธรรม

```bash
python -m categorization.finetuning.evaluate --run outputs/typhoon_run_01 --output outputs/val_zero --method zero-shot
python -m categorization.finetuning.evaluate --run outputs/typhoon_run_01 --output outputs/val_few --method few-shot
```

ทั้ง base/adapter ใช้ frozen NF4 checkpoint เดียวกัน เปิด/ปิด adapter เพื่อเทียบ
ใช้ greedy generation และไม่ได้บังคับ JSON Schema; JSON ที่ผิดนับเป็น invalid
รายงาน Accuracy, Macro F1, รายหมวด, mean/p95 warm latency และ raw output
ใหม่มี 8 หมวดรวม other จึงห้ามเทียบคะแนนตรงกับข้อสอบเก่า 7 หมวด/คนละข้อมูล
Ollama เดิมเป็น Q4_K_M + constrained JSON ส่วนชุดนี้เป็น NF4 + unconstrained JSON
ให้เปรียบเทียบก่อน/หลังภายใน runtime เดียวกันก่อน

เมื่อเลือกวิธีและหยุดปรับค่าแล้วจึงเปิด final test:

```bash
python -m categorization.finetuning.evaluate --run outputs/typhoon_run_01 --split test --confirm-final-test --method zero-shot --output outputs/final_test
```

หากปรับอีกหลังดู final test ต้องมีชุดทดสอบใหม่ที่ยังไม่ใช้ตัดสินใจ

### CSV ชุดท้าทายเพิ่มเติม

```bash
python -m categorization.finetuning.evaluate --run outputs/typhoon_run_04 --input-csv outputs/challenge.csv --method zero-shot --output outputs/challenge_run_04
```

ใช้ id, feedback และ reviewed_category (หรือ category ถ้าไม่มี reviewed_category)
ไม่แบ่งข้อมูลใหม่ ไม่ฝึกใหม่ และไม่ต้องมี prepared splits เมื่อประเมิน CSV ภายนอก
ยังตรวจ prompt hash กับ run เดิม และใช้ base revision เดียวกับตอนฝึก
บันทึก SHA256 ของ CSV และสถานะการตรวจเฉลยใน metrics.json
reviewed_ai ยังหมายถึงเฉลยเสนอจาก AI ไม่ใช่การตรวจโดยมนุษย์
ผลเป็น external_development และใช้ร่วมกับ --split test / --confirm-final-test ไม่ได้
predictions.csv ของชุดภายนอกมี feedback; notebook รวมไฟล์นี้ในการดาวน์โหลดด้วย

## 6. นำกลับไป Ollama (ทำเมื่อผลดีขึ้นแล้ว)

```bash
python -m categorization.finetuning.merge --run outputs/typhoon_run_01 --output outputs/typhoon_merged
```

ขั้น merge โหลด original weights ที่ไม่ใช่ 4-bit จึงอาจต้องใช้ VRAM มากกว่าขั้นฝึก
จากนั้นใช้รูปแบบ import ที่ Ollama รุ่นที่ใช้อยู่รองรับ หรือแปลง Safetensors เป็น GGUF
ด้วย llama.cpp และ quantize เป็น Q4_K_M ก่อนสร้าง Ollama model tag ใหม่
ทดสอบ original กับ fine-tuned บน Ollama ด้วย settings และข้อสอบเดียวกันอีกครั้ง
อย่าทับ model tag เดิมหรือเปลี่ยนเว็บก่อนผ่านการทดสอบ

## ตรวจบนเครื่องนี้โดยไม่ใช้ GPU

```bash
python -m unittest discover -s tests -v
python -m categorization.finetuning.train --help
```

การผ่าน unit tests ยืนยันการเตรียมข้อมูล/การป้องกันข้อมูลรั่ว/การ mask labels
ไม่ใช่การยืนยันว่า GPU training หรือการ export สำเร็จแล้ว

แหล่งอ้างอิง:
[Typhoon model](https://huggingface.co/typhoon-ai/typhoon2.5-qwen3-4b),
[PEFT quantization](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization),
[Transformers Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer),
[Ollama import](https://docs.ollama.com/import)
