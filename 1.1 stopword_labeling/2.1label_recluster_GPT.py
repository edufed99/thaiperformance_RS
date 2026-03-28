# -*- coding: utf-8 -*-
import os
import re
import json
import time
import random
import pandas as pd
import requests
from typing import List, Iterable

# ======================
# CONFIG
# ======================
API_URL = "https://api.modelharbor.com/v1/chat/completions"
MODEL_NAME = "openai/gpt-4.1"

# ⚠️ ฝัง API key ตรงๆ ตามที่ร้องขอ (ไม่ปลอดภัยหากแชร์ไฟล์/ขึ้น Git)
API_KEY = "sk-wkujsX0ljqnir_57UhF4Fg"

INPUT_CSV = "non_stopwords_GPT.csv"       # ต้องมีคอลัมน์ 'word' หรือ 'Words'
OUTPUT_CSV = "classified_nonstopwords_GPT_output.csv"
MAX_WORDS_PER_PROMPT = 30               # ลดโอกาส timeout
REQUEST_TIMEOUT = 60                    # วินาที
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0

# ======================
# REGEX DEFINITIONS
# ======================
HEADER_RE = re.compile(r'^\s*\|\s*Main Label\s*\|\s*Sub Label\s*\|\s*Sub Group\s*\|\s*Words\s*\|\s*$', re.I)
SEP_RE    = re.compile(r'^\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*$')
ROW_RE    = re.compile(r'^\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|\s*$')
THAI_BLOCK_RE = re.compile(r'[\u0E00-\u0E7F]')

# ======================
# MASTER PROMPT TEMPLATE
# ======================
MASTER_PROMPT_TEMPLATE = r'''**หัวข้อ: การจัดระเบียบคำศัพท์ด้านศิลปวัฒนธรรมไทยตามโครงสร้างที่กำหนด**

**คำสั่งหลัก:**
คุณคือผู้เชี่ยวชาญด้านการจัดหมวดหมู่ข้อมูลศิลปวัฒนธรรมไทย (Thai Arts and Culture Classification Specialist) ภารกิจของคุณคือการนำคำศัพท์แต่ละคำจาก **"รายการคำศัพท์ที่ต้องจัดหมวดหมู่"** ที่ให้มาด้านล่าง ไปใส่ในตารางโครงสร้างที่กำหนดให้ถูกต้องและครบถ้วน

**หลักการสำคัญที่ต้องยึดถือ:**
1. **ยึดมั่นในโครงสร้างที่กำหนด (Strict Adherence to Structure):** **ห้ามสร้าง** Main Label, Sub Label, หรือ Sub Group ใหม่โดยเด็ดขาด ให้ใช้โครงสร้างจาก **"ตารางโครงสร้างหลักสำหรับชี้นำ"** ที่ให้มาเป็นพิมพ์เขียวเท่านั้น
2. **ความครบถ้วน (Completeness):** ต้องนำคำศัพท์ **ทุกคำ** จาก **"รายการคำศัพท์ที่ต้องจัดหมวดหมู่"** มาใส่ในตารางผลลัพธ์ให้ครบถ้วน ห้ามมีคำตกหล่น
3. **การจัดกลุ่มเชิงความหมาย (Semantic Placement):** สำหรับคำศัพท์แต่ละคำ ให้พิจารณาความหมายและจัดวางลงใน Sub Group ที่มีความหมายสอดคล้องกันมากที่สุด
4. **การรวมคำพ้องความหมาย (Synonym Consolidation):** ภายใน Sub Group หากมีคำที่เป็นคำพ้องความหมาย ให้แสดงไว้ในวงเล็บต่อท้ายกัน (เช่น มโนห์รา, มโนราห์)
5. **ข้อกำกับภาษา (สำคัญมาก):**
   - ค่าในคอลัมน์ **Main Label, Sub Label, Sub Group** ต้องเป็น **ภาษาไทยเท่านั้น** ห้ามมีคำทับศัพท์/ภาษาอังกฤษ/อักษรละตินปะปน
   - คอลัมน์ **Words** เป็นภาษาไทยตามรายการคำศัพท์
6. **รูปแบบผลลัพธ์ (Output Format):** ส่งคืนเป็น “ตาราง Markdown 4 คอลัมน์” เท่านั้น โดยมีหัวตารางดังนี้ (อย่าเพิ่มข้อความอื่นก่อน/หลังตาราง)
| Main Label | Sub Label | Sub Group | Words |
|---|---|---|---|

**ตารางโครงสร้างหลักสำหรับชี้นำ (Master Structure Blueprint + ตัวอย่าง 2–3 คำ/กลุ่มย่อย):**
---
**หมวดหมู่ที่ 1: ศิลปะการแสดงและมหรสพ (Performing Arts and Entertainment)**
* **Sub Label: ประเภทการแสดงและบทเพลง (Performance Types and Music)**
  * **Sub Group: นาฏศิลป์และละครรำ** (ตัวอย่าง: มโนราห์, ละครดึกดำบรรพ์)
  * **Sub Group: ดนตรีและบทเพลง** (ตัวอย่าง: เพลงหน้าพาทย์, เพลงคืนเดือนหงาย)
* **Sub Label: ท่วงท่าและดนตรี (Movement and Music Elements)**
  * **Sub Group: ลีลาและการเคลื่อนไหว** (ตัวอย่าง: แม่ไม้, การร่ายรำ)
  * **Sub Group: เครื่องดนตรีและอุปกรณ์ประกอบ** (ตัวอย่าง: สะล้อ, กลองสะบัดชัย)
* **Sub Label: องค์ประกอบละครเวที (Theatrical Elements)**
  * **Sub Group: บุคลากรและบทบาท** (ตัวอย่าง: ตัวพระ, นางเอก)
  * **Sub Group: เทคนิคและฉาก** (ตัวอย่าง: เบิกโรง, แม่ท่า)

**หมวดหมู่ที่ 2: วรรณกรรม ตัวละคร และโลกในตำนาน (Literature, Characters, and Legendary Worlds)**
* **Sub Label: ตัวละครและอมนุษย์ (Characters and Beings)**
  * **Sub Group: เทพ เทวดา และสิ่งศักดิ์สิทธิ์** (ตัวอย่าง: พระอินทร์, พระนารายณ์)
  * **Sub Group: กษัตริย์และตัวละครหลัก** (ตัวอย่าง: พระราม, ขุนแผน)
  * **Sub Group: ยักษ์และอสูร** (ตัวอย่าง: ทศกัณฐ์, กุมภกรรณ)
  * **Sub Group: วานรและทหารเอก** (ตัวอย่าง: หนุมาน, สุครีพ)
  * **Sub Group: ตัวละครฝ่ายหญิงและบริวาร** (ตัวอย่าง: นางสีดา, รจนา)
* **Sub Label: เหตุการณ์และกลวิธีในเรื่องเล่า (Plot and Narrative Devices)**
  * **Sub Group: สงครามและการต่อสู้** (ตัวอย่าง: สงคราม, ฟาดฟัน)
  * **Sub Group: เวทมนตร์และอุบาย** (ตัวอย่าง: แปลงกาย, อุบาย)
  * **Sub Group: การเดินทางและสถานที่** (ตัวอย่าง: กลับบ้าน, เขาไกรลาส)
  * **Sub Group: ฉากและเหตุการณ์สำคัญ** (ตัวอย่าง: การเกี้ยวพาราสี, ไฟไหม้)
  * **Sub Group: อาวุธและของวิเศษ** (ตัวอย่าง: ศรนาคบาศ, ข่ายเพชร)

**หมวดหมู่ที่ 3: สังคม ประเพณี และวิถีชีวิต (Society, Traditions, and Way of Life)**
* **Sub Label: ประเพณี พิธีกรรม และความเชื่อ (Traditions, Rituals, and Beliefs)**
  * **Sub Group: พิธีการและเทศกาล** (ตัวอย่าง: สงกรานต์, ลอยกระทง)
  * **Sub Group: ความเชื่อและค่านิยม** (ตัวอย่าง: จารีตประเพณี, สิ่งศักดิ์สิทธิ์)
* **Sub Label: กลุ่มชาติพันธุ์และการดำรงชีพ (Ethnic Groups and Livelihoods)**
  * **Sub Group: กลุ่มชนและเชื้อชาติ** (ตัวอย่าง: ชาวภูไท, ไทยจีน)
  * **Sub Group: การทำมาหากินและหัตถกรรม** (ตัวอย่าง: ทอผ้า, เกษตรกรรม)
  * **Sub Group: สันทนาการและการละเล่น** (ตัวอย่าง: มวยไทย, มวยโบราณ)
  * **Sub Group: วัฒนธรรมและภาษา** (ตัวอย่าง: การหลอมรวม, ภาษามลายู)
* **Sub Label: ปฏิสัมพันธ์และธรรมชาติ (Interaction and Nature)**
  * **Sub Group: คุณธรรมและความสัมพันธ์** (ตัวอย่าง: ผู้มีพระคุณ, ความจงรักภักดี)
  * **Sub Group: อารมณ์และสภาวะ** (ตัวอย่าง: ความบันเทิง, รำลึก)
  * **Sub Group: ธรรมชาติและสิ่งแวดล้อม** (ตัวอย่าง: แสงจันทร์, มหาสมุทร)
  * **Sub Group: วัตถุและของมีค่า** (ตัวอย่าง: อัญมณี, พระพุทธรูป)
  * **Sub Group: สีและรูปลักษณ์** (ตัวอย่าง: สีทอง, ฟันปลา)

**หมวดหมู่ที่ 4: ประวัติศาสตร์ ราชสำนัก และมรดกชาติ (History, Royal Court, and National Heritage)**
* **Sub Label: สถาบันและบุคคลสำคัญ (Institutions and Key Figures)**
  * **Sub Group: พระมหากษัตริย์และราชวงศ์** (ตัวอย่าง: พระบาทสมเด็จพระจุลจอมเกล้าเจ้าอยู่หัว, พระเจ้าชัยวรมัน)
  * **Sub Group: ราชสำนักและขุนนาง** (ตัวอย่าง: เจ้าพระยา, อุปราช)
* **Sub Label: ร่องรอยและยุคสมัย (Traces and Eras)**
  * **Sub Group: ยุคสมัยและอาณาจักร** (ตัวอย่าง: สมัยสุโขทัย, กรุงรัตนโกสินทร์)
  * **Sub Group: เมืองและโบราณสถาน** (ตัวอย่าง: ปราสาทพนมรุ้ง, ปราสาทหินพิมาย)
* **Sub Label: มรดกและสัญลักษณ์ประจำชาติ (Heritage and National Symbols)**
  * **Sub Group: มรดกทางวัฒนธรรม** (ตัวอย่าง: จารึก, ปูนปั้น)
  * **Sub Group: ธรรมเนียมและเครื่องยศ** (ตัวอย่าง: ฉลองพระองค์, ทับหลัง)
---

**ขั้นตอนการดำเนินงาน:**
1. อ่านและทำความเข้าใจโครงสร้าง
2. จัดวางคำศัพท์แต่ละคำจาก **"รายการคำศัพท์ที่ต้องจัดหมวดหมู่"** ลงใน Sub Group ที่เหมาะสมที่สุด
3. สร้าง “ตาราง Markdown 4 คอลัมน์” ตามหัวตารางที่กำหนด (ห้ามมีข้อความอื่นนอกตาราง)
4. ทวนสอบความครบถ้วนว่าทุกคำถูกจัดหมวดแล้ว

**รายการคำศัพท์ที่ต้องจัดหมวดหมู่ (ชุดย่อย):**
---
{word_list}
---'''

# ======================
# HELPERS
# ======================
def read_words(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"[WARN] ไม่พบไฟล์ {path}")
        return []
    df = pd.read_csv(path)
    candidates = []
    for col in ["word", "Word", "Words"]:
        if col in df.columns:
            candidates = df[col].dropna().astype(str).tolist()
            break
    if not candidates and df.shape[1] == 1:
        candidates = df.iloc[:, 0].dropna().astype(str).tolist()
    words = []
    for w in candidates:
        for x in str(w).split(","):
            x = x.strip()
            if x:
                words.append(x)
    return words

def chunk_list(seq: List[str], k: int) -> List[List[str]]:
    return [seq[i:i+k] for i in range(0, len(seq), k)]

def backoff_sleep(attempt: int):
    time.sleep((RETRY_BASE_SLEEP * (2 ** (attempt-1))) + random.uniform(0, 0.5))

def ensure_markdown_table_only(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, started = [], False
    for ln in lines:
        if HEADER_RE.match(ln):
            started = True
            out.append(ln)
            continue
        if not started:
            continue
        if SEP_RE.match(ln) or ROW_RE.match(ln):
            out.append(ln)
        else:
            break
    return "\n".join(out).strip()

# ======================
# PARSER
# ======================
def parse_markdown_table(md: str) -> pd.DataFrame:
    md = ensure_markdown_table_only(md)
    if not md:
        return pd.DataFrame(columns=["Main Label", "Sub Label", "Sub Group", "Words"])
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    rows, header_found = [], False
    for ln in lines:
        if HEADER_RE.match(ln):
            header_found = True
            continue
        if not header_found:
            continue
        if SEP_RE.match(ln):
            continue
        m = ROW_RE.match(ln)
        if m:
            cols = [c.strip().strip("*_`") for c in m.groups()]
            rows.append(cols)
    df = pd.DataFrame(rows, columns=["Main Label","Sub Label","Sub Group","Words"])
    for c in df.columns:
        df[c] = df[c].map(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

# ======================
# API CALLS
# ======================
def classify_words(words: Iterable[str]) -> pd.DataFrame:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": MASTER_PROMPT_TEMPLATE.format(word_list=", ".join(words))}
        ],
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            df = parse_markdown_table(content)
            if not df.empty:
                return df
            else:
                last_err = RuntimeError("ได้ผลลัพธ์แต่ตารางว่าง")
        except Exception as e:
            last_err = e
        if attempt < MAX_RETRIES:
            print(f"[RETRY] ครั้งที่ {attempt} ล้มเหลว: {last_err}. ลองใหม่…")
            backoff_sleep(attempt)
    raise RuntimeError(f"จัดหมวดคำไม่สำเร็จหลัง retry {MAX_RETRIES} ครั้ง: {last_err}")

# ======================
# VALIDATION / REPORT
# ======================
def is_thai_text(s: str) -> bool:
    return bool(THAI_BLOCK_RE.search(s or ""))

def report_non_thai_rows(df: pd.DataFrame, tag: str = ""):
    bad = df[~df["Main Label"].map(is_thai_text) | ~df["Sub Label"].map(is_thai_text) | ~df["Sub Group"].map(is_thai_text)]
    if not bad.empty:
        print(f"[WARN][{tag}] พบค่าที่ไม่ใช่ไทยใน Main/Sub/Sub:")
        print(bad.to_string(index=False))
    else:
        print(f"[OK][{tag}] ค่าหมวดทั้งหมดเป็นภาษาไทย")

def print_hierarchy(df: pd.DataFrame):
    grouped = df.groupby(["Main Label","Sub Label","Sub Group"], as_index=False)["Words"].first()
    main_groups = grouped.groupby("Main Label")
    for main, g1 in main_groups:
        print(f"\n# {main}")
        for sub, g2 in g1.groupby("Sub Label"):
            print(f"  - {sub}")
            for _, row in g2.iterrows():
                print(f"      • {row['Sub Group']}: {row['Words']}")

# ======================
# PROCESS (PER CHUNK)
# ======================
def process_chunk_adaptive(chunk_words: List[str]) -> pd.DataFrame:
    try:
        df = classify_words(chunk_words)
        if not df.empty:
            return df
    except Exception as e:
        if len(chunk_words) > 5:
            mid = len(chunk_words)//2
            left = process_chunk_adaptive(chunk_words[:mid])
            right = process_chunk_adaptive(chunk_words[mid:])
            return pd.concat([left, right], ignore_index=True)
        else:
            print(f"[FAIL] chunk ขนาดเล็กยังล้มเหลว: {e}")
            return pd.DataFrame(columns=["Main Label","Sub Label","Sub Group","Words"])
    return pd.DataFrame(columns=["Main Label","Sub Label","Sub Group","Words"])

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print("--- เริ่มกระบวนการจัดหมวดหมู่คำศัพท์ด้วย Prompt Engineering ---")

    if not API_KEY:
        raise SystemExit("❌ ไม่พบ API_KEY — โปรดตรวจสอบค่า")

    words = read_words(INPUT_CSV)
    if not words:
        print("--- สิ้นสุดการทำงาน เนื่องจากไม่พบคำ ---")
        raise SystemExit(1)

    base_chunks = chunk_list(words, MAX_WORDS_PER_PROMPT)
    print(f"[INFO] แบ่งเป็น {len(base_chunks)} ชุด ชุดละ ≤ {MAX_WORDS_PER_PROMPT} คำ (รวม {len(words)} คำ)")

    frames = []
    for i, chunk in enumerate(base_chunks, start=1):
        print(f"\n=== [PROCESS] ชุดที่ {i}/{len(base_chunks)} (เริ่มต้น {len(chunk)} คำ) ===")
        print(f"[DEBUG] ตัวอย่างคำ 20 คำแรก: {chunk[:20]}")
        df = process_chunk_adaptive(chunk)
        if df is None or df.empty:
            print(f"[WARN] ชุดที่ {i} ไม่ได้ผลลัพธ์ ข้าม…")
            continue
        report_non_thai_rows(df, tag=f"chunk #{i}")
        frames.append(df)

    if not frames:
        print("\n--- ไม่สามารถรับผลลัพธ์จากการจัดหมวดหมู่ได้ ---")
        raise SystemExit(0)

    result = pd.concat(frames, ignore_index=True)

    # รวมคำซ้ำต่อกลุ่ม (unique + sorted)
    result["Words"] = result["Words"].fillna("")
    def merge_words(series: pd.Series) -> str:
        bag = set()
        for part in series:
            for w in str(part).split(","):
                w = w.strip()
                if w:
                    bag.add(w)
        return ", ".join(sorted(bag))

    result = (
        result.groupby(["Main Label","Sub Label","Sub Group"], as_index=False)["Words"]
        .apply(merge_words)
    )

    report_non_thai_rows(result, tag="FINAL")

    print("\n=== ลำดับชั้นผลลัพธ์ ===")
    print_hierarchy(result)

    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[สำเร็จ] บันทึกผลที่ไฟล์: {OUTPUT_CSV}")
