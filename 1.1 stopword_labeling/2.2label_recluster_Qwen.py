# -*- coding: utf-8 -*-
import re
import time
import pandas as pd
import requests
from typing import List, Optional, Tuple

# ======================
# CONFIG
# ======================
API_URL = "https://api.modelharbor.com/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-235b-a22b-instruct-2507"  # Using Qwen model
INPUT_CSV = "non_stopwords_QWEN.csv"
OUTPUT_CSV = "classified_nonstopwords_QWEN_output.csv"

API_KEY = "sk-wkujsX0ljqnir_57UhF4Fg"  # ใช้สำหรับทดสอบตามที่ผู้ใช้ระบุ (อย่า commit ขึ้น repo สาธารณะ)

TEMPERATURE = 0.1
MAX_TOKENS = 6144
REQUEST_TIMEOUT = 180
MAX_WORDS_PER_PROMPT = 30
MAX_RETRIES = 3
BACKOFF_SECONDS = 3
MIN_CHUNK_SIZE = 10

# ======================
# MASTER PROMPT (ฉบับเต็ม + ข้อกำชับภาษา + ตัวอย่าง 2–3 คำ/กลุ่มย่อย)
# ======================
MASTER_PROMPT_TEMPLATE = """
**หัวข้อ: การจัดระเบียบคำศัพท์ด้านศิลปวัฒนธรรมไทยตามโครงสร้างที่กำหนด**

**คำสั่งหลัก:**
คุณคือผู้เชี่ยวชาญด้านการจัดหมวดหมู่ข้อมูลศิลปวัฒนธรรมไทย (Thai Arts and Culture Classification Specialist) ภารกิจของคุณคือการนำคำศัพท์แต่ละคำจาก **"รายการคำศัพท์ที่ต้องจัดหมวดหมู่"** ที่ให้มาด้านล่าง ไปใส่ในตารางโครงสร้างที่กำหนดให้ถูกต้องและครบถ้วน

**หลักการสำคัญที่ต้องยึดถือ:**
1. **ยึดมั่นในโครงสร้างที่กำหนด (Strict Adherence to Structure):** **ห้ามสร้าง** Main Label, Sub Label, หรือ Sub Group ใหม่โดยเด็ดขาด ให้ใช้โครงสร้างจาก **"ตารางโครงสร้างหลักสำหรับชี้นำ"** ที่ให้มาเป็นพิมพ์เขียวเท่านั้น
2. **ความครบถ้วน (Completeness):** ต้องนำคำศัพท์ **ทุกคำ** จาก **"รายการคำศัพท์ที่ต้องจัดหมวดหมู่"** มาใส่ในตารางผลลัพธ์ให้ครบถ้วน ห้ามมีคำตกหล่น
3. **การจัดกลุ่มเชิงความหมาย (Semantic Placement):** สำหรับคำศัพท์แต่ละคำ ให้พิจารณาความหมายและจัดวางลงใน Sub Group ที่มีความหมายสอดคล้องกันมากที่สุด
4. **การรวมคำพ้องความหมาย (Synonym Consolidation):** ภายใน Sub Group หากมีคำที่เป็นคำพ้องความหมาย ให้แสดงไว้ในวงเล็บต่อท้ายกัน (เช่น มโนห์รา, มโนราห์)
5. **ข้อกำชับภาษา (สำคัญมาก):**
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
---
"""

# ======================
# MARKDOWN PARSER
# ======================
HEADER_RE = re.compile(r'^\s*\|\s*Main Label\s*\|\s*Sub Label\s*\|\s*Sub Group\s*\|\s*Words\s*\|\s*$', re.I)
SEP_RE    = re.compile(r'^\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*$')
ROW_RE    = re.compile(r'^\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|\s*$')

def extract_table_lines(text: str) -> list:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out, in_table, saw_header = [], False, False
    for ln in lines:
        if not in_table and HEADER_RE.match(ln):
            in_table, saw_header = True, True
            out.append(ln); continue
        if in_table and SEP_RE.match(ln):
            out.append(ln); continue
        if in_table:
            if ROW_RE.match(ln):
                out.append(ln)
            elif saw_header:
                break
    return out

def parse_markdown_table(md: str) -> pd.DataFrame:
    print("[DEBUG] เริ่มพาร์สตาราง Markdown...")
    lines = extract_table_lines(md)
    if not lines:
        print("[WARN] ไม่พบบรรทัดตารางในข้อความ")
        return pd.DataFrame(columns=["Main Label","Sub Label","Sub Group","Words"])
    rows = []
    for ln in lines:
        if HEADER_RE.match(ln) or SEP_RE.match(ln):
            continue
        m = ROW_RE.match(ln)
        if m:
            cols = [c.strip().strip("*_`") for c in m.groups()]
            rows.append(cols)
    df = pd.DataFrame(rows, columns=["Main Label","Sub Label","Sub Group","Words"])
    for c in df.columns:
        df[c] = df[c].map(lambda x: re.sub(r'\s+', ' ', x).strip())
    print(f"[DEBUG] ได้ข้อมูลตาราง {len(df)} แถว")
    return df

# ======================
# LANGUAGE CHECKER (ไทยเท่านั้นสำหรับ 3 คอลัมน์แรก)
# ======================
LATIN_RE = re.compile(r"[A-Za-z]")

def is_thai_only(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return LATIN_RE.search(text) is None

def report_non_thai_rows(df: pd.DataFrame, tag: str = ""):
    if df.empty:
        return
    bad = df[
        (~df["Main Label"].map(is_thai_only)) |
        (~df["Sub Label"].map(is_thai_only)) |
        (~df["Sub Group"].map(is_thai_only))
    ]
    if not bad.empty:
        print(f"[WARN] พบค่าไม่ใช่ไทยในคอลัมน์ Label ({len(bad)} แถว){' - '+tag if tag else ''}")
        print(bad[["Main Label","Sub Label","Sub Group","Words"]].head(5))

# ======================
# IO
# ======================
def read_words(path: str) -> List[str]:
    print(f"[STEP] อ่านไฟล์: {path}")
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] ไม่พบไฟล์: {path}")
        return []
    if "word" not in df.columns:
        print(f"[ERROR] ไม่พบคอลัมน์ 'word' ใน {path}")
        return []
    words = (df["word"].dropna().astype(str).map(str.strip))
    words = words[words.str.len() > 0].unique().tolist()
    print(f"[INFO] พบคำไม่ซ้ำทั้งหมด {len(words)} คำ")
    return words

def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

# ======================
# API CALL
# ======================
def call_api(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    คืนค่า: (content, finish_reason, raw_json)
    """
    print("[DEBUG] เรียก API ด้วย Prompt ขนาด", len(prompt), "ตัวอักษร")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    for attempt in range(1, MAX_RETRIES+1):
        print(f"[STEP] ส่งคำขอ API (ครั้งที่ {attempt}/{MAX_RETRIES})...")
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            print(f"[DEBUG] สถานะ HTTP: {resp.status_code}")
            if resp.status_code != 200:
                print("[DEBUG] ตัวอย่างเนื้อหา:", resp.text[:500])
            resp.raise_for_status()
            data = resp.json()

            finish_reason = None
            content = ""
            if "choices" in data and data["choices"]:
                ch = data["choices"][0]
                finish_reason = ch.get("finish_reason")
                msg = ch.get("message") or {}
                content = (msg.get("content") or ch.get("text") or "") if isinstance(ch, dict) else ""

            if content:
                print(f"[INFO] ได้ข้อความจากโมเดล ขนาด {len(content)} ตัวอักษร | finish_reason={finish_reason}")
                print("[DEBUG] ตัวอย่าง 400 ตัวอักษรแรก:\n", content[:400])
                return content, finish_reason, data
            else:
                print(f"[WARN] เนื้อหาว่าง | finish_reason={finish_reason}")
                print("[DEBUG] JSON (ย่อ):", str(data)[:500])
                if finish_reason == "length":
                    return None, "length", data
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API ล้มเหลว: {e}")
            try:
                print("[DEBUG] body:", resp.text[:500])
            except Exception:
                pass
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_SECONDS * attempt)
    return None, None, None

# ======================
# CHUNK PROCESSOR WITH ADAPTIVE SPLIT
# ======================
def process_chunk_adaptive(words_chunk: List[str]) -> Optional[pd.DataFrame]:
    size = len(words_chunk)
    while size >= MIN_CHUNK_SIZE:
        print(f"[PROCESS] ลองประมวลผล chunk ขนาด {size} คำ")
        prompt = MASTER_PROMPT_TEMPLATE.format(word_list=", ".join(words_chunk[:size]))
        content, reason, _ = call_api(prompt)
        if content:
            df = parse_markdown_table(content)
            if not df.empty:
                return df
            else:
                print("[WARN] ได้ content แต่พาร์สไม่ได้ → หั่น chunk เพิ่ม")
        elif reason == "length":
            print("[INFO] ถูกตัดเพราะยาว (finish_reason=length) → หั่น chunk ครึ่งหนึ่งแล้วลองใหม่")
        else:
            print("[WARN] ไม่ได้ content และไม่ใช่ length → หั่น chunk เพิ่มเพื่อความชัวร์")

        size = max(MIN_CHUNK_SIZE, size // 2)
        if size < len(words_chunk):
            words_chunk = words_chunk[:size]
    print("[STOP] หั่นจนถึงขนาดต่ำสุดแล้วยังไม่สำเร็จ")
    return None

# ======================
# HIERARCHY PRINTER (พิมพ์ลำดับชั้น + คำ)
# ======================
def print_hierarchy(df: pd.DataFrame):
    if df.empty:
        print("[INFO] ไม่มีข้อมูลสำหรับพิมพ์ลำดับชั้น")
        return
    out = {}
    for _, row in df.iterrows():
        main = row["Main Label"]
        sub  = row["Sub Label"]
        grp  = row["Sub Group"]
        words = [w.strip() for w in (row["Words"] or "").split(",") if w.strip()]
        out.setdefault(main, {}).setdefault(sub, {}).setdefault(grp, set()).update(words)

    for main, sub_map in out.items():
        print(main)
        for sub, grp_map in sub_map.items():
            print(f"  --{sub}")
            for grp, wset in grp_map.items():
                print(f"     ---{grp}")
                if wset:
                    joined = ", ".join(sorted(wset))
                    print(f"        -----{joined}")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print("--- เริ่มกระบวนการจัดหมวดหมู่คำศัพท์ด้วย Prompt Engineering (Qwen 3 235B) ---")
    words = read_words(INPUT_CSV)
    if not words:
        print("--- สิ้นสุดการทำงาน เนื่องจากไม่พบคำ ---")
        raise SystemExit(1)

    base_chunks = chunk_list(words, MAX_WORDS_PER_PROMPT)
    print(f"[INFO] แบ่งเป็น {len(base_chunks)} ชุด เริ่มต้นชุดละ ≤ {MAX_WORDS_PER_PROMPT} คำ")

    frames = []
    for i, chunk in enumerate(base_chunks, start=1):
        print(f"\n=== [PROCESS] ชุดที่ {i}/{len(base_chunks)} (เริ่มต้น {len(chunk)} คำ) ===")
        print(f"[DEBUG] ตัวอย่างคำ 20 คำแรก: {chunk[:20]}")
        df = process_chunk_adaptive(chunk)
        if df is None:
            print(f"[WARN] ชุดที่ {i} ล้มเหลวหลัง adapt แล้ว ข้าม…")
            continue
        report_non_thai_rows(df, tag=f"chunk #{i}")
        frames.append(df)

    if not frames:
        print("\n--- ไม่สามารถรับผลลัพธ์จากการจัดหมวดหมู่ได้ ---")
        raise SystemExit(0)

    result = pd.concat(frames, ignore_index=True)
    result["Words"] = result["Words"].fillna("")
    result = (
        result.groupby(["Main Label","Sub Label","Sub Group"], as_index=False)["Words"]
        .apply(lambda s: ", ".join(sorted(set(w.strip() for part in s for w in part.split(",") if w.strip()))))
    )

    report_non_thai_rows(result, tag="FINAL")

    print("\n=== ลำดับชั้นผลลัพธ์ ===")
    print_hierarchy(result)

    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[สำเร็จ] บันทึกผลที่ไฟล์: {OUTPUT_CSV}")
