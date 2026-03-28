# -*- coding: utf-8 -*-
"""
2.3label_recluster_Deepseek.py (Robust Version v2.4)
- Model: deepseek/deepseek-r1-0528 via ModelHarbor chat/completions
- Adds: resilient Markdown parser, HTTP retries with backoff+jitter,
        adaptive chunking, retry pass for failed chunks,
        checkpoints after each chunk, raw JSONL logging,
        flexible header/sep parsing, JSON error guard, separator-row cleanup,
        timezone-aware timestamps, coverage verification (per-chunk & final),
        and automatic re-label passes for missing words.
"""

import os
import re
import time
import random
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd
import requests
from typing import List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ======================
# CONFIG
# ======================
API_URL = "https://api.modelharbor.com/v1/chat/completions"
API_KEY = "sk-wkujsX0ljqnir_57UhF4Fg"  # ← ฝังไว้ตามต้องการ (อย่า commit สาธารณะ)
MODEL_NAME = "deepseek/deepseek-r1-0528"  # ← ใช้ DeepSeek R1-0528

INPUT_CSV = "non_stopwords_DEEPSEEK.csv"
OUTPUT_CSV = "classified_nonstopwords_DEEPSEEK_output.csv"
PARTIAL_CSV = "classified_nonstopwords_DEEPSEEK_output.partial.csv"
RAW_JSONL  = "deepseek_raw.jsonl"

TEMPERATURE = 0.1
MAX_TOKENS = 6144
REQUEST_TIMEOUT = 240            # ↑ ขยาย timeout
MAX_WORDS_PER_PROMPT = 24        # ↓ ลดคำต่อชุดเพื่อกัน timeout/ตัดคำ
MAX_RETRIES = 3
BACKOFF_SECONDS = 3
MIN_CHUNK_SIZE = 10

# Auto re-label settings
MAX_RELABEL_PASSES = 2
RELABEL_MAX_WORDS_PER_PROMPT = 16

def _mask_key(k: str) -> str:
    if not isinstance(k, str) or len(k) < 8:
        return "***"
    return k[:6] + "..." + k[-4:]

print(f"[INFO] ใช้ API Key: {_mask_key(API_KEY)}")  # แสดงเฉพาะบางส่วน

# ======================
# MASTER PROMPT (ฉบับเต็ม)
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

# Prompt เฉพาะรอบซ่อม (เข้มงวดขึ้น)
RELABEL_PROMPT_TEMPLATE = MASTER_PROMPT_TEMPLATE + """
**ข้อกำชับเฉพาะรอบซ่อม (STRICT):**
- ห้ามพิมพ์คำใด ๆ นอกเหนือจากที่อยู่ใน **รายการคำศัพท์ที่ต้องจัดหมวดหมู่** สำหรับรอบนี้
- ห้ามแปล/ขยายความ/เพิ่มคำใหม่โดยเด็ดขาด
- ถ้าคำหนึ่งคำสามารถอยู่ได้หลายกลุ่ม ให้เลือก **เพียงกลุ่มเดียวที่เหมาะสมที่สุด** เท่านั้น
"""

# ======================
# MARKDOWN PARSER (ทนทานต่อรูปแบบตารางแตกต่าง)
# ======================
# Header อนุญาต ** รอบข้อความได้
HEADER_RE = re.compile(
    r'^\s*\|\s*\**\s*Main Label\s*\**\s*\|\s*\**\s*Sub Label\s*\**\s*\|\s*\**\s*Sub Group\s*\**\s*\|\s*\**\s*Words\s*\**\s*\|\s*$',
    re.I
)
# แถวคั่น: ตาราง 4 คอลัมน์ (ยืดหยุ่นรองรับ :---: / ---: / :--- / ---)
SEP_RE = re.compile(r'^\s*\|(?:\s*:?-+:?\s*\|){4}\s*$', re.I)
# แถวข้อมูลทั่วไป (อนุญาตอักขระตกแต่ง ** __ ` รอบค่า)
ROW_RE = re.compile(r'^\s*\|(.+?)\|(.+?)\|(.+?)\|(.+?)\|\s*$', re.I)

def _strip_md(x: str) -> str:
    # เอา ** __ ` และช่องว่างซ้ำๆ ออก
    x = re.sub(r'[\*_`]+', '', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def extract_table_lines(text: str) -> list:
    # รับได้แม้มีข้อความนำหน้า เช่น "ตารางผลลัพธ์:"
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, in_table, saw_header = [], False, False
    for ln in lines:
        if not ln.strip():
            continue
        if not in_table and HEADER_RE.match(ln):
            in_table, saw_header = True, True
            out.append(ln)
            continue
        if in_table:
            if SEP_RE.match(ln):
                out.append(ln)
                continue
            if ROW_RE.match(ln):
                out.append(ln)
                continue
            if ln.strip().startswith('|') and ln.count('|') >= 4:
                out.append(ln)
                continue
            if saw_header:
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
        if not m:
            continue
        cols = [_strip_md(c) for c in m.groups()]
        rows.append(cols)

    df = pd.DataFrame(rows, columns=["Main Label","Sub Label","Sub Group","Words"])
    for c in df.columns:
        df[c] = df[c].map(lambda x: re.sub(r'\s+', ' ', x).strip())

    # กันแถวคั่นที่เล็ดลอดเข้ามาเป็นข้อมูล
    def _is_sep_row(series):
        return all(bool(re.fullmatch(r'-+', str(series[c]).strip())) for c in df.columns)
    if not df.empty:
        mask_sep = df.apply(_is_sep_row, axis=1)
        if mask_sep.any():
            df = df.loc[~mask_sep].reset_index(drop=True)

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
# Coverage Verification Helpers
# ======================
def _collect_words(df: pd.DataFrame) -> set:
    s = set()
    if df.empty or "Words" not in df.columns:
        return s
    for cell in df["Words"].dropna():
        for w in map(str.strip, str(cell).split(",")):
            if w:
                s.add(w)
    return s

def check_chunk_coverage(words_chunk: List[str], df: pd.DataFrame) -> List[str]:
    placed = _collect_words(df)
    chunk_set = set(map(str.strip, words_chunk))
    missing = sorted(chunk_set - placed)
    print(f"[VERIFY] คำใน chunk = {len(chunk_set)} | ใส่ตารางแล้ว = {len(chunk_set)-len(missing)} | ขาด = {len(missing)}")
    if missing:
        print("[MISSING]", ", ".join(missing[:20]), "..." if len(missing) > 20 else "")
    return missing

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
# HTTP Session with Retry & Jitter
# ======================
def _build_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.8,                 # base backoff
        status_forcelist=[429, 500, 502, 503, 504, 524],
        allowed_methods=frozenset(['POST']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

_SESSION = _build_session()

def _sleep_with_jitter(base: float, attempt: int):
    # Exponential backoff + jitter (max 15s)
    delay = base * (2 ** (attempt - 1)) + random.uniform(0, base)
    time.sleep(min(delay, 15))

def _log_raw(prompt: str, response_json: Optional[dict], finish_reason: Optional[str]):
    try:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rec = {
            "ts": ts,
            "model": MODEL_NAME,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "finish_reason": finish_reason,
            "response": response_json,
        }
        with open(RAW_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] เขียน RAW JSONL ไม่สำเร็จ: {e}")

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
            resp = _SESSION.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            print(f"[DEBUG] สถานะ HTTP: {resp.status_code}")
            if resp.status_code != 200:
                print("[DEBUG] ตัวอย่างเนื้อหา:", resp.text[:500])
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                print("[ERROR] ตอบกลับไม่ใช่ JSON แท้:", resp.text[:500])
                if attempt < MAX_RETRIES:
                    _sleep_with_jitter(BACKOFF_SECONDS, attempt)
                    continue
                _log_raw(prompt, {"non_json": resp.text[:1000]}, None)
                return None, None, None

            finish_reason, content = None, ""
            choices = data.get("choices") or []
            if choices:
                ch = choices[0]
                finish_reason = ch.get("finish_reason")
                msg = ch.get("message") or {}
                content = msg.get("content") or ch.get("text") or ""

            # บันทึก raw jsonl ทุกครั้ง
            _log_raw(prompt, data, finish_reason)

            if content:
                print(f"[INFO] ได้ข้อความจากโมเดล ขนาด {len(content)} ตัวอักษร | finish_reason={finish_reason}")
                print("[DEBUG] ตัวอย่าง 400 ตัวอักษรแรก:\n", content[:400])
                return content, finish_reason, data
            else:
                print(f"[WARN] เนื้อหาว่าง | finish_reason={finish_reason}")
                if finish_reason == "length":
                    return None, "length", data
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API ล้มเหลว: {e}")
            try:
                print("[DEBUG] body:", resp.text[:500])
            except Exception:
                pass
        if attempt < MAX_RETRIES:
            _sleep_with_jitter(BACKOFF_SECONDS, attempt)
    # บันทึกกรณีล้มเหลวทั้งหมด
    _log_raw(prompt, None, None)
    return None, None, None

# ======================
# CHUNK PROCESSORS
# ======================
def process_chunk_adaptive_with_template(words_chunk: List[str], prompt_template: str) -> Optional[pd.DataFrame]:
    size = len(words_chunk)
    local_chunk = list(words_chunk)
    while size >= MIN_CHUNK_SIZE:
        print(f"[PROCESS] ลองประมวลผล chunk ขนาด {size} คำ")
        prompt = prompt_template.format(word_list=", ".join(local_chunk[:size]))
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
        if size < len(local_chunk):
            local_chunk = local_chunk[:size]
    print("[STOP] หั่นจนถึงขนาดต่ำสุดแล้วยังไม่สำเร็จ")
    return None

def process_chunk_adaptive(words_chunk: List[str]) -> Optional[pd.DataFrame]:
    return process_chunk_adaptive_with_template(words_chunk, MASTER_PROMPT_TEMPLATE)

# ======================
# RELABEL (สำหรับคำที่ตกหล่น)
# ======================
def relabel_missing_words(missing_words: List[str], pass_no: int) -> Optional[pd.DataFrame]:
    if not missing_words:
        return pd.DataFrame(columns=["Main Label","Sub Label","Sub Group","Words"])
    subchunks = chunk_list(missing_words, RELABEL_MAX_WORDS_PER_PROMPT)
    frames = []
    for i, sc in enumerate(subchunks, start=1):
        print(f"\n[RELABEL] รอบซ่อม {pass_no} | ชุดย่อย {i}/{len(subchunks)} ({len(sc)} คำ)")
        df = process_chunk_adaptive_with_template(sc, RELABEL_PROMPT_TEMPLATE)
        if df is None:
            print("[RELABEL] ชุดย่อยนี้ยังไม่สำเร็จ")
            continue
        report_non_thai_rows(df, tag=f"relabel pass {pass_no} subchunk {i}")
        _ = check_chunk_coverage(sc, df)
        frames.append(df)
        # เช็คพอยท์รอบซ่อม
        try:
            pd.concat(frames, ignore_index=True).to_csv(f"relabel_pass_{pass_no}.partial.csv", index=False, encoding="utf-8-sig")
            print(f"[CHECKPOINT] saved -> relabel_pass_{pass_no}.partial.csv")
        except Exception as e:
            print(f"[WARN] บันทึกเช็คพอยท์รอบซ่อมล้มเหลว: {e}")
    if not frames:
        return None
    relabeled = pd.concat(frames, ignore_index=True)
    relabeled.to_csv(f"relabel_pass_{pass_no}.csv", index=False, encoding="utf-8-sig")
    print(f"[RELABEL] pass {pass_no} summary saved -> relabel_pass_{pass_no}.csv")
    return relabeled

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
    print("--- เริ่มกระบวนการจัดหมวดหมู่คำศัพท์ด้วย DeepSeek R1-0528 ---")
    words = read_words(INPUT_CSV)
    if not words:
        print("--- สิ้นสุดการทำงาน เนื่องจากไม่พบคำ ---")
        raise SystemExit(1)

    base_chunks = chunk_list(words, MAX_WORDS_PER_PROMPT)
    print(f"[INFO] แบ่งเป็น {len(base_chunks)} ชุด (≤ {MAX_WORDS_PER_PROMPT} คำ/ชุด)")

    frames: List[pd.DataFrame] = []
    failed_idx: List[int] = []

    # รอบแรก
    for i, chunk in enumerate(base_chunks, start=1):
        print(f"\n=== [PROCESS] ชุดที่ {i}/{len(base_chunks)} ({len(chunk)} คำ) ===")
        df = process_chunk_adaptive(chunk)
        if df is None:
            print(f"[WARN] ชุดที่ {i} ล้มเหลวหลัง adapt แล้ว—คิวไว้ลองใหม่รอบท้าย")
            failed_idx.append(i-1)
        else:
            report_non_thai_rows(df, tag=f"chunk #{i}")
            frames.append(df)
            # ตรวจความครบถ้วนของ chunk นี้
            _ = check_chunk_coverage(chunk, df)
            # ✓ Checkpoint หลังจบแต่ละชุด
            try:
                pd.concat(frames, ignore_index=True).to_csv(PARTIAL_CSV, index=False, encoding="utf-8-sig")
                print(f"[CHECKPOINT] saved -> {PARTIAL_CSV}")
            except Exception as e:
                print(f"[WARN] บันทึกเช็คพอยท์ล้มเหลว: {e}")

    # รอบชดเชย: ลดขนาด batch และลองใหม่
    if failed_idx:
        print(f"\n[RETRY PASS] ลองใหม่ {len(failed_idx)} ชุดแบบลดขนาด batch")
        old_max = MAX_WORDS_PER_PROMPT
        try:
            new_max = max(18, min(22, old_max - 2))  # รอบแก้ใหม่ ลดลงอีกนิด
            print(f"[RETRY PASS] ปรับ MAX_WORDS_PER_PROMPT: {old_max} → {new_max}")
            for j in failed_idx:
                subwords = base_chunks[j]
                subchunks = chunk_list(subwords, new_max)
                for sc in subchunks:
                    df = process_chunk_adaptive(sc)
                    if df is not None:
                        report_non_thai_rows(df, tag=f"retry-of-chunk #{j+1}")
                        frames.append(df)
                        # ตรวจความครบถ้วนของ subchunk
                        _ = check_chunk_coverage(sc, df)
                        # ✓ Checkpoint ทุก subchunk ที่สำเร็จ
                        try:
                            pd.concat(frames, ignore_index=True).to_csv(PARTIAL_CSV, index=False, encoding="utf-8-sig")
                            print(f"[CHECKPOINT] saved -> {PARTIAL_CSV}")
                        except Exception as e:
                            print(f"[WARN] บันทึกเช็คพอยท์ล้มเหลว: {e}")
                    else:
                        print(f"[FAIL] ยังไม่สำเร็จในรอบชดเชย (chunk #{j+1})")
        finally:
            MAX_WORDS_PER_PROMPT = old_max

    if not frames:
        print("\n--- ไม่สามารถรับผลลัพธ์จากการจัดหมวดหมู่ได้ ---")
        raise SystemExit(0)

    result = pd.concat(frames, ignore_index=True)

    # รวมคำซ้ำตาม label เดียวกัน และรวมคำในคอลัมน์ Words
    result["Words"] = result["Words"].fillna("")
    result = (
        result.groupby(["Main Label","Sub Label","Sub Group"], as_index=False)["Words"]
        .agg(lambda s: ", ".join(sorted(set(w.strip() for part in s for w in part.split(",") if w.strip()))))
    )

    # ตรวจภาษาไทยอีกครั้งในผลลัพธ์สุดท้าย
    report_non_thai_rows(result, tag="FINAL")

    # ✅ ตรวจความครบถ้วนระดับทั้งงาน
    all_missing = sorted(set(words) - _collect_words(result))
    print(f"[VERIFY-FINAL] ทั้งหมด {len(words)} คำ | ครอบคลุมแล้ว {len(words)-len(all_missing)} | ขาด {len(all_missing)}")
    if all_missing:
        try:
            pd.Series(all_missing, name="missing").to_csv("missing_words.csv", index=False, encoding="utf-8-sig")
            print("[OUTPUT] missing_words.csv -> มีคำที่ยังไม่ถูกจัดหมวด โปรดตรวจ")
        except Exception as e:
            print(f"[WARN] บันทึก missing_words.csv ล้มเหลว: {e}")

        # ===== Automatic re-label passes =====
        for pass_no in range(1, MAX_RELABEL_PASSES + 1):
            if not all_missing:
                break
            print(f"\n[AUTO-RELABEL] เริ่มรอบซ่อมครั้งที่ {pass_no} สำหรับ {len(all_missing)} คำที่ขาด")
            relabeled_df = relabel_missing_words(all_missing, pass_no)
            if relabeled_df is None or relabeled_df.empty:
                print(f"[AUTO-RELABEL] ไม่ได้ผลลัพธ์ในรอบ {pass_no}")
                continue
            # รวมเข้ากับผลเดิม
            result = pd.concat([result, relabeled_df], ignore_index=True)

            # รวมคำซ้ำและ normalize อีกครั้ง
            result["Words"] = result["Words"].fillna("")
            result = (
                result.groupby(["Main Label","Sub Label","Sub Group"], as_index=False)["Words"]
                .agg(lambda s: ", ".join(sorted(set(w.strip() for part in s for w in part.split(",") if w.strip()))))
            )

            # เช็ครับอีกครั้ง
            all_missing = sorted(set(words) - _collect_words(result))
            print(f"[VERIFY-FINAL] หลังรอบซ่อม {pass_no} | ขาด {len(all_missing)} คำ")
            if all_missing:
                try:
                    pd.Series(all_missing, name=f"missing_after_pass_{pass_no}").to_csv(
                        f"missing_words_pass_{pass_no}.csv", index=False, encoding="utf-8-sig"
                    )
                    print(f"[OUTPUT] missing_words_pass_{pass_no}.csv -> รายการคำที่ยังขาดหลังรอบซ่อม {pass_no}")
                except Exception as e:
                    print(f"[WARN] บันทึก missing_words_pass_{pass_no}.csv ล้มเหลว: {e}")

    # ✅ พิมพ์ผลแบบหมวดหมู่ + คำตามรูปแบบใหม่
    print("\n=== ลำดับชั้นผลลัพธ์ ===")
    print_hierarchy(result)

    # บันทึกไฟล์ CSV ไว้ใช้งานต่อ
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[สำเร็จ] บันทึกผลที่ไฟล์: {OUTPUT_CSV}")
