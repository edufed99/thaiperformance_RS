# -*- coding: utf-8 -*-
import os
import re
import time
import json
import hashlib
import random
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
API_URL = "https://api.modelharbor.com/v1/chat/completions"
API_KEY = "sk-wkujsX0ljqnir_57UhF4Fg"
MODEL_NAME = "qwen/qwen3-235b-a22b-instruct-2507"  # Using Qwen model

INPUT_CSV_FILE = "cluster_results.csv"
STOPWORDS_OUTPUT_CSV = "stopwords_analysis_QWEN.csv"
NON_STOPWORDS_OUTPUT_CSV = "non_stopwords_QWEN.csv"
PARTIAL_OUTPUT_CSV = "stopwords_analysis_QWEN.partial.csv"

# Runlog + metadata for resuming/retrying
RUNLOG_CSV = "stopwords_runlog.csv"
RUNMETA_JSON = "stopwords_runmeta.json"

REQUEST_TIMEOUT = 120
TEMPERATURE = 0.1
MAX_TOKENS = 2048
CHUNK_SIZE = 20              # initial batch size
CHUNK_SIZE_RETRY = 12        # smaller size for failed-batch retries
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
BACKOFF_JITTER = (0.2, 0.8)

SEED_STOPWORDS = {
    "พิธี", "สาวงาม", "ท้องทะเล", "ริมแม่น้ำ", "สายน้ำ",
    "ท่วงทำนอง", "แต่งกาย", "อาภรณ์", "กสิกรรม", "ออกศึก", "กลอุบาย",
    "แปลงตัว", "จำแลง", "กลายร่าง", "วิรุณจำบัง", "ศรพรหมาสตร์", "เขาไกลลาส", "กรุงเทพ",
    "พลวานร", "ไปมา", "มือเปล่า", "ที่อยู่", "ที่นั่ง", "วิชา", "มโนห์รา",
    "พื้นฐาน", "ประดุจ", "เนื้อเรื่อง", "ทำนองเพลง", "ต้นขา", "แห่งชาติ", "ขึ้นทะเบียน",
    "ลักษณะพิเศษ", "ต้นแบบ", "แบบอย่าง", "แนวคิด", "หลักปฏิบัติ", "วิถี", "หน้าที่",
    "บรรดา", "ตอนนี้", "โดยรวม", "หลากหลาย", "บริเวณ", "ที่รวม", "ทางการ", "มากมาย",
    "การขับ", "ขับลำ", "การร้อง", "ร่ายรำ", "การเคลื่อนไหว", "กระลวน", "ความไพเราะ", "เสียงดนตรี", "บทละคร",
    "ชื่อเรื่อง", "คำร้อง", "บทร้อง", "ลำนำ", "ที่มา", "ศิลปิน", "ผู้ประดิษฐ์",
    "ตัวละคร", "ผู้แสดงฝ่ายหญิง", "ผู้แสดงฝ่ายชาย", "ติดมือ", "ว่องไว", "คล่องแคล่ว",
    "จัดแสดง", "เวที", "รามเกียรติ์", "นางกินรี", "จนตาย", "สิ้นชีพ", "ฤทธิ์เดช",
    "ลงกา", "ตามเพลง", "ได้ยิน", "อาวุธ", "ทำพิธี", "มงคล", "ศรัทธา",
    "การทอ", "ประกอบการ", "กิจการงาน", "สะเดาะ", "กีฬา", "กตเวที", "พบปะ",
    "บันเทิงใจ", "เป็นบ้า", "กิริยา", "อากัปกิริยา", "บรรพต", "ตะวันออกเฉียงเหนือ", "ตามคำสั่ง",
    "รับสั่ง", "บัญชา", "ประทับ", "มรดก", "การอนุรักษ์", "คุณูปการ",
    "นวัตกรรม", "ทรงคุณค่า", "คำสั่ง", "ก้มกราบ", "กราบทูล", "เข้าเฝ้า", "พระองค์",
    "ไตรรงค์", "ระดับชาติ", "ลีลา", "กระบวน", "ธิเบศร์", "กลุ่มชาติพันธุ์",
    "รักสนุก", "ทางวัฒนธรรม", "มีวัฒนธรรม", "การสอน", "ว่าการ", "ประกาศ", "ประทาน",
    "ให้การ", "หาทาง", "บอกทาง", "คืนหนึ่ง", "วันวาน", "ครั้งสุดท้าย", "เชิงบันได",
    "ชั้นใน", "ข้างใน", "มือซ้าย", "มือขวา", "ผู้พบเห็น", "ผู้ใช้", "ตัวจริง",
    "เรื่องราว", "วาจา", "ผลงาน", "สากล", "พื้นผิว", "บุรี",
    "ตัวตน", "สัญลักษณ์", "รัชกาล", "พระราชบัญชา", "แผ่อิทธิพล", "กระบวนท่า", "กรรมวิธี", "นำเสนอ",
    "ตัวจริง", "ผู้พบเห็น", "ประชากร", "การดำเนินชีวิต", "กิจวัตร", "ของป่า", "ความเป็นอยู่", "ป้องกันตัว",
    "กิจจะ", "คู่ใจ", "ทางธรรมชาติ", "พื้นผิว", "แต่งตัว", "นึกถึง", "ประจำปี",
    "การดำรงชีวิต", "แกล้งทำ", "การแสดง", "ขึ้นไป", "มาจาก", "ผสมผสาน", "เคลื่อนไหว", "พื้นที่", "ความงดงาม",
    "โบราณ", "คนไทย", "มีความหมาย", "การโปรย", "มีอายุ", "ยมะ",
    "เดินทาง", "วัตถุประสงค์", "เสรี", "กตัญญู", "พระคุณ", "แผ่นดิน",
    "ประกอบอาชีพ", "บรรยากาศ", "พื้นบ้าน", "จังหวะ", "สอดแทรก", "ชีวิต",
    "ระยะเวลา", "คู่ต่อสู้", "อิริยาบถ", "บุหงา", "ชื่อดอกไม้", "ร่วมใน",
    "ทำนอง", "ผู้เชี่ยวชาญ", "สำเนียง", "ประเทศไทย", "ชั้นเดียว", "วิธี",
    "ความหมาย", "พี่เลี้ยง", "กลับบ้าน", "ออกเดินทาง", "ขับร้อง", "การออกไป",
    "เลียนแบบ", "โอกาส", "กระบวนการ", "กำเริบ", "มีฤทธิ์", "เข้าพบ",
    "เอนก", "ออกอุบาย", "สมทบ", "นางเอก", "พระเอก",
    "เปรียบเสมือน", "การรำ", "ผู้ชม", "เป็นต้นแบบ", "ความร้อน"
}

# ------------------------------------------------------------------------------
# PROMPT TEMPLATE
# ------------------------------------------------------------------------------
PROMPT_TEMPLATE = """
# MISSION BRIEFING

คุณจะรับบทเป็น **ผู้เชี่ยวชาญด้านภาษาศาสตร์เชิงคำนวณ (Computational Linguist)** ที่มีความแม่นยำสูง

**PRIMARY OBJECTIVE (เป้าหมายหลัก):**
วิเคราะห์ข้อมูลคำศัพท์ที่ได้จากการแบ่งกลุ่ม (Clustering) และทำการจำแนกคำที่ไม่สำคัญ (Stopword) ออกมาอย่างเป็นระบบ

**CONTEXT (บริบทของข้อมูล):**
ข้อมูลทั้งหมดในส่วน `Full Dataset for Analysis` มาจากคลังข้อมูลเกี่ยวกับ **"ศิลปวัฒนธรรม นาฏศิลป์ และการแสดงของไทย"**

คุณต้องปฏิบัติตามเกณฑ์, ตัวอย่าง, และรูปแบบผลลัพธ์ที่กำหนดให้อย่างเคร่งครัด  
**ห้ามเพิ่มหมวดใหม่, ห้ามเปลี่ยนชื่อหมวด, และห้ามตอบภาษาอื่นนอกจากภาษาไทย**

---

# SECTION 1: CRITERIA FOR JUDGEMENT (เกณฑ์การตัดสินใจ)

คุณต้องใช้เกณฑ์ 4 ข้อต่อไปนี้เป็นหลักในการตัดสินใจ โดย **คำหนึ่งต้องถูกจัดให้อยู่เพียงหมวดเดียว**  
และใช้ลำดับความสำคัญดังนี้:  
1) คำที่มีความหมายกว้าง (Umbrella Term)  
2) คำเชิงนามธรรม หรือ คำเชิงประเมินค่า (Abstract/Evaluative Term)  
3) คำเฉพาะโดเมนที่พบบ่อยเกินไป (Domain-Specific Overuse)  
4) คำซ้ำซ้อน หรือ คำพ้องความหมาย (Redundant/Synonym)  

**นิยามเกณฑ์:**
1. **คำที่มีความหมายกว้าง (Umbrella Term / Low Semantic Weight):**
   - คำทั่วไปที่ใช้เป็นหมวดหมู่, ไม่ชี้เฉพาะเจาะจง, ใช้ได้หลายบริบท
   - ตัวอย่าง: `ลักษณะ`, `บุคคล`, `สถานที่`, `กิจกรรม`, `รูปแบบ`, `ประเทศ`

2. **คำเชิงนามธรรม หรือ คำเชิงประเมินค่า (Abstract or Evaluative Term):**
   - แสดงแนวคิด ความรู้สึก ความเชื่อ หรือความคิดเห็น/คุณค่า
   - ตัวอย่าง: `ความสุข`, `ความเชื่อ`, `ความสำคัญ`, `งดงาม`, `สวยงาม`, `ยอดเยี่ยม`

3. **คำเฉพาะโดเมนที่พบบ่อยเกินไป (Domain-Specific Overuse):**
   - คำเทคนิคในโดเมนนี้ที่พบถี่มากจนไม่ช่วยจำแนก
   - ตัวอย่าง: `ศิลปะ`, `วัฒนธรรม`, `การแสดง`, `ประเพณี`, `นาฏศิลป์`

4. **คำซ้ำซ้อน หรือ คำพ้องความหมาย (Redundant or Synonym):**
   - ความหมายซ้ำ/ใกล้เคียงจนเก็บไว้เพียงคำเดียวได้
   - ตัวอย่าง: `สวยงาม` และ `งดงาม`; `พิธี` และ `พิธีกรรม`

---

# SECTION 2: EXAMPLE OF EXECUTION (ตัวอย่างการทำงาน)

**Input ตัวอย่าง (CSV):**
word,cluster_label
ประเทศ,2
สนุกสนาน,2
การแสดง,1
งดงาม,2

**Output ที่คาดหวัง:**
# ผลการวิเคราะห์ Stopword
## Cluster 1 (จำนวน 1 คำ)
1. คำที่มีความหมายกว้าง (Umbrella Terms):
-
2. คำเชิงนามธรรม หรือ คำเชิงประเมินค่า (Abstract or Evaluative Terms):
-
3. คำเฉพาะโดเมนที่พบบ่อยเกินไป (Domain-Specific Overuse):
* `การแสดง`: คำพื้นฐานในโดเมน ทำให้จำแนกต่ำ
4. คำซ้ำซ้อน หรือ คำพ้องความหมาย (Redundant or Synonym):
-

## Cluster 2 (จำนวน 3 คำ)
1. คำที่มีความหมายกว้าง (Umbrella Terms):
* `ประเทศ`: นามทั่วไป ใช้เป็นหมวดหมู่
2. คำเชิงนามธรรม หรือ คำเชิงประเมินค่า (Abstract or Evaluative Terms):
* `สนุกสนาน`: แสดงความรู้สึก/คุณค่า
* `งดงาม`: แสดงคุณค่า/ความคิดเห็น
3. คำเฉพาะโดเมนที่พบบ่อยเกินไป (Domain-Specific Overuse):
-
4. คำซ้ำซ้อน หรือ คำพ้องความหมาย (Redundant or Synonym):
-

---

# SECTION 3: FULL DATASET FOR ANALYSIS (ข้อมูลทั้งหมดสำหรับวิเคราะห์)

```csv
{csv_data}
```

# SECTION 4: OUTPUT REQUIREMENTS (รูปแบบผลลัพธ์ที่ต้องการ)

วิเคราะห์ทีละ Cluster (เรียงจากใหญ่ไปเล็กถ้าเป็นไปได้)  
สำหรับแต่ละ Cluster ให้แยกหัวข้อ 4 ประเภทตามเกณฑ์ด้านบน

จัดรูปแบบให้เป็นหัวข้อและ bullet ดังนี้:

# ผลการวิเคราะห์ Stopword
## Cluster [หมายเลข] (จำนวน [X] คำ)
1. คำที่มีความหมายกว้าง (Umbrella Terms):
* `คำ`: เหตุผลย่อ หรือ "-" ถ้าไม่มี
2. คำเชิงนามธรรม หรือ คำเชิงประเมินค่า (Abstract or Evaluative Terms):
* `คำ`: เหตุผลย่อ หรือ "-" ถ้าไม่มี
3. คำเฉพาะโดเมนที่พบบ่อยเกินไป (Domain-Specific Overuse):
* `คำ`: เหตุผลย่อ หรือ "-" ถ้าไม่มี
4. คำซ้ำซ้อน หรือ คำพ้องความหมาย (Redundant or Synonym):
* `คำ`: เหตุผลย่อ หรือ "-" ถ้าไม่มี

**ข้อห้าม:**
- ห้ามเพิ่มหมวดใหม่
- ห้ามเปลี่ยนเลขลำดับหัวข้อ
- ห้ามใช้ภาษาที่ไม่ใช่ภาษาไทย
- ห้ามเพิ่มข้อมูลที่ไม่มีใน CSV
"""

# ------------------------------------------------------------------------------
# TIME HELPERS (UTC, timezone-aware)
# ------------------------------------------------------------------------------
def now_utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ------------------------------------------------------------------------------
# UTILS: signature, runlog, metadata
# ------------------------------------------------------------------------------
def dataset_signature(df: pd.DataFrame) -> str:
    concat = "\n".join(f"{w}|{c}" for w, c in zip(df["word"].astype(str), df["cluster_label"].astype(str)))
    return hashlib.md5(concat.encode("utf-8")).hexdigest()

def init_runmeta(sig: str, chunk_size: int, model: str):
    meta = {"dataset_signature": sig, "chunk_size": chunk_size, "model": model, "created_at": now_utc_iso_z()}
    with open(RUNMETA_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_runmeta():
    if not os.path.exists(RUNMETA_JSON):
        return None
    with open(RUNMETA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def init_runlog(batches):
    df = pd.DataFrame(batches, columns=["batch_id","start_idx","end_idx","size"])
    df["status"] = "pending"
    df["attempts"] = 0
    df["last_error"] = ""
    df["updated_at"] = now_utc_iso_z()
    # enforce stable dtypes (avoid float64-inferred text cols)
    df = df.astype({
        "batch_id": "int64",
        "start_idx": "int64",
        "end_idx": "int64",
        "size": "int64",
        "status": "string",
        "attempts": "Int64",
        "last_error": "string",
        "updated_at": "string",
    })
    df.to_csv(RUNLOG_CSV, index=False, encoding="utf-8-sig")

def load_runlog():
    if not os.path.exists(RUNLOG_CSV):
        return None
    # read with explicit dtypes so empty text columns aren't float64
    return pd.read_csv(
        RUNLOG_CSV,
        dtype={
            "batch_id": "int64",
            "start_idx": "int64",
            "end_idx": "int64",
            "size": "int64",
            "status": "string",
            "attempts": "Int64",
            "last_error": "string",
            "updated_at": "string",
        }
    )

def update_runlog(batch_id: int, status: str, attempts: int, last_error: str = ""):
    df = load_runlog()
    if df is None:
        return
    # ensure dtypes before assigning
    try:
        df = df.astype({
            "status": "string",
            "attempts": "Int64",
            "last_error": "string",
            "updated_at": "string",
        })
    except Exception:
        pass

    mask = df["batch_id"] == batch_id
    df.loc[mask, "status"] = str(status)
    df.loc[mask, "attempts"] = int(attempts)
    df.loc[mask, "last_error"] = "" if last_error is None else str(last_error)
    df.loc[mask, "updated_at"] = now_utc_iso_z()

    df.to_csv(RUNLOG_CSV, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------------------
# API CALL WITH RETRIES
# ------------------------------------------------------------------------------
def call_api(prompt_content: str) -> (str, str):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    delay = 1.0
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content, None
        except Exception as e:
            last_err = str(e)
            print(f"   ⚠️  Attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
            time.sleep(delay + random.uniform(*BACKOFF_JITTER))
            delay *= BACKOFF_BASE
    return None, last_err or "Unknown error"

# ------------------------------------------------------------------------------
# PARSE OUTPUT
# ------------------------------------------------------------------------------
def parse_analysis_to_dataframe(analysis_text: str, original_df: pd.DataFrame) -> pd.DataFrame:
    stopwords_data = []
    current_cluster = None
    current_type = None

    word_to_cluster = pd.Series(
        original_df["cluster_label"].values,
        index=original_df["word"].astype(str)
    ).to_dict()

    cluster_re = re.compile(r"(?:##\s*)?cluster\s*(\d+)", re.IGNORECASE)
    type_re    = re.compile(r"^\s*\d+\.\s*(.+)", re.IGNORECASE)
    item_re    = re.compile(r"^[\*\-\u2022]\s*`?([^`:\n]+?)`?\s*:\s*(.+)$", re.IGNORECASE)

    for raw in analysis_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m_cluster = cluster_re.search(line)
        if m_cluster:
            try:
                current_cluster = int(m_cluster.group(1))
            except ValueError:
                current_cluster = None
            current_type = None
            continue

        m_type = type_re.search(line)
        if m_type:
            current_type = m_type.group(1).strip().rstrip(":")
            continue

        m_item = item_re.search(line)
        if m_item and current_cluster is not None and current_type:
            word = m_item.group(1).strip()
            reason = m_item.group(2).strip()
            cluster_label = word_to_cluster.get(word, current_cluster)
            stopwords_data.append({
                "word": word,
                "cluster_label": cluster_label,
                "stopword_type": current_type,
                "reason": reason
            })

    return pd.DataFrame(stopwords_data)

# ------------------------------------------------------------------------------
# BATCHING
# ------------------------------------------------------------------------------
def make_batches(df: pd.DataFrame, chunk_size: int):
    batches = []
    n = len(df)
    batch_id = 1
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        size = end - start
        batches.append((batch_id, start, end, size))
        batch_id += 1
    return batches

def chunk_dataframe(df, start_idx, end_idx):
    return df.iloc[start_idx:end_idx]

# ------------------------------------------------------------------------------
# SAFE SAVE
# ------------------------------------------------------------------------------
def safe_save_partial(df: pd.DataFrame):
    if not df.empty:
        df.to_csv(PARTIAL_OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"🧾 บันทึกชั่วคราว → {PARTIAL_OUTPUT_CSV} ({len(df)} แถว)")

# ------------------------------------------------------------------------------
# PROCESS ONE SPAN (supports sub-batching if needed)
# ------------------------------------------------------------------------------
def process_span(span_df: pd.DataFrame, original_df: pd.DataFrame, sub_chunk_size: int = None) -> (pd.DataFrame, str):
    if sub_chunk_size is None:
        csv_data_string = span_df.to_csv(index=False)
        prompt = PROMPT_TEMPLATE.format(csv_data=csv_data_string)
        content, err = call_api(prompt)
        if err:
            return pd.DataFrame(), err
        return parse_analysis_to_dataframe(content, original_df), None

    merged = pd.DataFrame()
    last_err = None
    for sub_start in range(0, len(span_df), sub_chunk_size):
        sub_end = min(sub_start + sub_chunk_size, len(span_df))
        sub_df = span_df.iloc[sub_start:sub_end]
        csv_sub = sub_df.to_csv(index=False)
        sub_prompt = PROMPT_TEMPLATE.format(csv_data=csv_sub)
        sub_content, sub_err = call_api(sub_prompt)
        if sub_err:
            last_err = sub_err
            print(f"   ❌ Sub-batch {sub_start}-{sub_end} failed: {sub_err}")
            continue
        merged = pd.concat([merged, parse_analysis_to_dataframe(sub_content, original_df)], ignore_index=True)
    return merged, last_err

# ------------------------------------------------------------------------------
# MAIN MODES
# ------------------------------------------------------------------------------
def run_all():
    print("🚀 เริ่มกระบวนการจำแนก Stopword แบบ Batch (Qwen + Runlog + RetryFailed + dtypefix)")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"❌ ไม่พบไฟล์: {INPUT_CSV_FILE}")
        return

    original_df = pd.read_csv(INPUT_CSV_FILE)
    filtered_df = original_df[~original_df["word"].astype(str).isin(SEED_STOPWORDS)].reset_index(drop=True)

    sig = dataset_signature(filtered_df)
    init_runmeta(sig, CHUNK_SIZE, MODEL_NAME)

    batches = make_batches(filtered_df, CHUNK_SIZE)
    init_runlog(batches)

    all_stopwords_df = pd.DataFrame()

    for (batch_id, start_idx, end_idx, size) in batches:
        span_df = chunk_dataframe(filtered_df, start_idx, end_idx)
        print(f"\n📦 Batch {batch_id}: index {start_idx}-{end_idx} ({size} คำ)")

        df, err = process_span(span_df, original_df, sub_chunk_size=None)
        attempts = 1

        if df.empty and err:
            print("   🔁 Falling back to sub-batching...")
            df, err = process_span(span_df, original_df, sub_chunk_size=max(8, min(CHUNK_SIZE_RETRY, size)))
            attempts += 1

        if df.empty and err:
            update_runlog(batch_id, "failed", attempts, err)
            print(f"   ❌ Batch {batch_id} failed: {err}")
            continue

        all_stopwords_df = pd.concat([all_stopwords_df, df], ignore_index=True)
        all_stopwords_df.drop_duplicates(subset=["word","cluster_label","stopword_type"], inplace=True)
        update_runlog(batch_id, "success", attempts, "")
        print(f"   ✅ Batch {batch_id} success → {len(df)} รายการ (สะสม {len(all_stopwords_df)})")
        safe_save_partial(all_stopwords_df)

    finalize_outputs(original_df, all_stopwords_df)

def retry_failed_only():
    print("🔁 โหมดรันซ้ำเฉพาะ batch ที่ล้มเหลว (dtypefix)")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"❌ ไม่พบไฟล์: {INPUT_CSV_FILE}")
        return

    meta = load_runmeta()
    if meta is None:
        print("❌ ไม่พบไฟล์ metadata (runmeta). โปรดรันโหมดปกติก่อนเพื่อสร้าง runlog/meta")
        return

    original_df = pd.read_csv(INPUT_CSV_FILE)
    filtered_df = original_df[~original_df["word"].astype(str).isin(SEED_STOPWORDS)].reset_index(drop=True)

    current_sig = dataset_signature(filtered_df)
    if current_sig != meta.get("dataset_signature"):
        print("❌ ข้อมูลปัจจุบันไม่ตรงกับรอบก่อน (signature mismatch). หยุดเพื่อความปลอดภัย")
        return

    log = load_runlog()
    if log is None:
        print("❌ ไม่พบ runlog. โปรดรันโหมดปกติก่อน")
        return

    pending = log[log["status"] != "success"]
    if pending.empty:
        print("ℹ️ ไม่มี batch ที่ล้มเหลว/ค้างคา")
        return

    print(f"📋 พบ {len(pending)} batch ที่ต้องรันซ้ำ")

    all_stopwords_df = pd.DataFrame()
    if os.path.exists(PARTIAL_OUTPUT_CSV):
        try:
            all_stopwords_df = pd.read_csv(PARTIAL_OUTPUT_CSV)
        except Exception:
            all_stopwords_df = pd.DataFrame()

    for _, row in pending.iterrows():
        batch_id = int(row["batch_id"])
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        size = int(row["size"])
        attempts = int(row.get("attempts", 0))

        span_df = chunk_dataframe(filtered_df, start_idx, end_idx)
        print(f"\n🔁 Retrying Batch {batch_id}: index {start_idx}-{end_idx} ({size} คำ)")

        df, err = process_span(span_df, original_df, sub_chunk_size=max(8, min(CHUNK_SIZE_RETRY, size)))
        attempts += 1

        if df.empty and err:
            update_runlog(batch_id, "failed", attempts, err)
            print(f"   ❌ Still failing: {err}")
            continue

        all_stopwords_df = pd.concat([all_stopwords_df, df], ignore_index=True)
        all_stopwords_df.drop_duplicates(subset=["word","cluster_label","stopword_type"], inplace=True)
        update_runlog(batch_id, "success", attempts, "")
        print(f"   ✅ Batch {batch_id} success (retry) → {len(df)} รายการ (สะสม {len(all_stopwords_df)})")
        safe_save_partial(all_stopwords_df)

    finalize_outputs(original_df, all_stopwords_df)

# ------------------------------------------------------------------------------
# FINALIZE
# ------------------------------------------------------------------------------
def finalize_outputs(original_df: pd.DataFrame, all_stopwords_df: pd.DataFrame):
    if all_stopwords_df.empty:
        print("ℹ️ ไม่มี stopwords ที่ตรวจพบ")
        return

    all_stopwords_df.to_csv(STOPWORDS_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 บันทึก stopwords ทั้งหมด → {STOPWORDS_OUTPUT_CSV}")

    stop_set = set(all_stopwords_df["word"].astype(str)) | SEED_STOPWORDS
    non_stopwords_df = original_df[~original_df["word"].astype(str).isin(stop_set)]
    non_stopwords_df.to_csv(NON_STOPWORDS_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 บันทึก non-stopwords → {NON_STOPWORDS_OUTPUT_CSV}")
    print(f"📊 สรุป: เดิม {len(original_df)} | stopwords {len(all_stopwords_df) + len(SEED_STOPWORDS)} | เหลือ {len(non_stopwords_df)}")

# ------------------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stopword batch analyzer with retry-failed support (dtypefix)")
    parser.add_argument("--retry-failed", action="store_true", help="รันซ้ำเฉพาะ batch ที่ล้มเหลว (ต้องมี runlog/meta จากรอบก่อน)")
    args = parser.parse_args()

    if args.retry_failed:
        retry_failed_only()
    else:
        run_all()

if __name__ == "__main__":
    main()
