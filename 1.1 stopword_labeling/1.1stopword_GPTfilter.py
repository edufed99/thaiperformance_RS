# -*- coding: utf-8 -*-
import os
import re
import requests
import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
API_URL = "https://api.modelharbor.com/v1/chat/completions"
API_KEY = "sk-wkujsX0ljqnir_57UhF4Fg"
MODEL_NAME = "openai/gpt-4.1"

INPUT_CSV_FILE = "cluster_results.csv"
STOPWORDS_OUTPUT_CSV = "stopwords_analysis_GPT.csv"
NON_STOPWORDS_OUTPUT_CSV = "non_stopwords_GPT.csv"

REQUEST_TIMEOUT = 600
TEMPERATURE = 0.1
MAX_TOKENS = 8192
CHUNK_SIZE = 30  # จำนวนคำต่อ batch

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
# 2) PROMPT TEMPLATE (หัวใจของระบบ)
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
# API CALL
# ------------------------------------------------------------------------------
def get_stopword_analysis(prompt_content: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return f"ERROR: {e}"

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

    # รองรับ "## Cluster N (จำนวน ...)" และกรณีเขียนคลาดเคลื่อน
    cluster_re = re.compile(r"(?:##\s*)?cluster\s*(\d+)", re.IGNORECASE)
    # จับหัวข้อย่อยแบบ "1. ....", "2. ...." ฯลฯ
    type_re    = re.compile(r"^\s*\d+\.\s*(.+)", re.IGNORECASE)
    # จับ bullet ที่ขึ้นต้นด้วย * หรือ - และมี "คำ: เหตุผล"
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
        # ข้ามกรณีหัวข้อว่างที่เป็น "-" (จะไม่ match กับ item_re อยู่แล้ว)
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
# SPLIT DATAFRAME INTO CHUNKS
# ------------------------------------------------------------------------------
def chunk_dataframe(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    print("🚀 เริ่มกระบวนการจำแนก Stopword แบบ Batch")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"❌ ไม่พบไฟล์: {INPUT_CSV_FILE}")
        return

    original_df = pd.read_csv(INPUT_CSV_FILE)
    print(f"📖 โหลดข้อมูล {len(original_df)} แถว")

    filtered_df = original_df[~original_df["word"].astype(str).isin(SEED_STOPWORDS)]
    print(f"✂️ เหลือ {len(filtered_df)} คำหลังกรอง SEED_STOPWORDS")

    all_stopwords_df = pd.DataFrame()

    for idx, chunk_df in enumerate(chunk_dataframe(filtered_df, CHUNK_SIZE), start=1):
        print(f"\n📦 Batch {idx}: {len(chunk_df)} คำ")
        csv_data_string = chunk_df.to_csv(index=False)
        prompt = PROMPT_TEMPLATE.format(csv_data=csv_data_string)

        analysis_text = get_stopword_analysis(prompt)
        if str(analysis_text).startswith("ERROR:"):
            print(f"❌ Batch {idx} ล้มเหลว: {analysis_text}")
            continue

        stopwords_df = parse_analysis_to_dataframe(analysis_text, original_df)
        all_stopwords_df = pd.concat([all_stopwords_df, stopwords_df], ignore_index=True)
        print(f"✅ Batch {idx} ได้ stopwords {len(stopwords_df)} คำ")

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

if __name__ == "__main__":
    main()
