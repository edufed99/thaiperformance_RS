# -*- coding: utf-8 -*-
r"""
======================================================================
สคริปต์ประเมินประสิทธิภาพโมเดลจัดหมวดคำ (STRICT WangchanBERTa)
- แสดงผลลัพธ์ทุกขั้นตอน + "กราฟ" บนหน้าจอ และฝังกราฟลง Excel
- ใช้ WangchanBERTa คำนวณ semantic coherence เท่านั้น (ไม่มี TF-IDF)
- Tokenization ไม่จำกัดเฉพาะอักษรไทย (ไทย/อังกฤษ/ตัวเลข)
- ปรับ coverage_parse_rate ไม่ให้เกิน 1 (นับ raw token ด้วย TOKEN_RE)
- แก้ปัญหาฟอนต์ไทยใน Matplotlib ด้วย "วิธี A"
======================================================================

Prerequisites:
    pip install torch transformers pandas numpy scikit-learn openpyxl matplotlib pillow

WangchanBERTa (STRICT):
- แนะนำดาวน์โหลดโมเดลไว้เครื่อง แล้วตั้งแวดล้อม:
    Windows: set WCB_LOCAL_DIR=C:\models\wcb
    macOS/Linux: export WCB_LOCAL_DIR=/path/to/wcb
- ถ้าไม่ตั้ง WCB_LOCAL_DIR โค้ดจะพยายามโหลดจาก HF:
    "airesearch/wangchanberta-base-att-spm-uncased" (ต้องออนไลน์)
"""

import os
import re
import math
import time
import datetime
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# กราฟ
import matplotlib.pyplot as plt

# ==== ตั้งค่าฟอนต์ไทยสำหรับ Matplotlib (วิธี A) ====
from matplotlib import font_manager as fm, rcParams

# 1) ถ้ามี path ฟอนต์เฉพาะ ให้ระบุผ่าน ENV: THAI_FONT_PATH (ไฟล์ .ttf/.otf)
#    เช่น C:\Windows\Fonts\LeelawUI.ttf, C:\Windows\Fonts\THSarabunNew.ttf, NotoSansThai-Regular.ttf
font_path = os.getenv("THAI_FONT_PATH", "").strip()

def _activate_font_by_name(name: str) -> bool:
    try:
        rcParams["font.family"] = name
        rcParams["font.sans-serif"] = [name]
        rcParams["axes.unicode_minus"] = False  # ให้เครื่องหมายลบแสดงถูก
        return True
    except Exception:
        return False

def setup_thai_font():
    tried = []
    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            # รีโหลดฐานข้อมูลฟอนต์ เพื่อให้เห็นฟอนต์ใหม่ทันที
            try:
                fm._load_fontmanager(try_read_cache=False)  # อาจไม่มีในบางเวอร์ชัน → โอเค
            except Exception:
                pass
            name = fm.FontProperties(fname=font_path).get_name()
            _activate_font_by_name(name)
            print(f"[INFO] ใช้ฟอนต์ไทยจากไฟล์: {name} ({font_path})")
            return
        except Exception as e:
            tried.append(f"{font_path} ({e})")

    # ชื่อฟอนต์ยอดนิยมบน Windows/ระบบทั่วไป
    candidates = [
        "Leelawadee UI",  # Windows ส่วนมากมี
        "Tahoma",
        "Sarabun",        # ถ้าติดตั้งไว้
        "TH Sarabun New",
        "Noto Sans Thai", # ถ้าติดตั้งไว้
        "Cordia New",
        "Angsana New",
    ]
    for name in candidates:
        if _activate_font_by_name(name):
            print(f"[INFO] ใช้ฟอนต์ไทยในระบบ: {name}")
            return

    print("[WARN] ไม่พบฟอนต์ไทย: กรุณาติดตั้ง Noto Sans Thai / Sarabun "
          "หรือกำหนด THAI_FONT_PATH ให้ชี้ไปยังไฟล์ฟอนต์ .ttf/.otf")
    if tried:
        print("Tried:", tried)

setup_thai_font()
# ==== จบตั้งค่าฟอนต์ไทย ====

# ฝังรูปลง Excel
from openpyxl.drawing.image import Image as XLImage

# ลด log/คำเตือนกวนใจจาก transformers
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message=r"`encoder_attention_mask` is deprecated.*",
    category=FutureWarning,
)

# แสดง DataFrame บนจอแบบสวยงาม หากมี ace_tools
_HAVE_ACE = False
try:
    import ace_tools as _tools  # ใช้ได้ในบางสภาพแวดล้อมเท่านั้น
    _HAVE_ACE = True
except Exception:
    pass

def _now():
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

def display_df(title: str, df: pd.DataFrame, head: int = 10, round_ndigits: int = None):
    df_show = df.copy()
    if _HAVE_ACE:
        try:
            _tools.display_dataframe_to_user(
                title,
                df_show.head(head).round(round_ndigits) if round_ndigits is not None else df_show.head(head)
            )
            log(f"แสดงตาราง '{title}' (top {head}) แล้ว")
            return
        except Exception:
            pass
    print(f"\n=== {title} (top {head}) ===")
    print(df_show.head(head).round(round_ndigits) if round_ndigits is not None else df_show.head(head))
    print("===")

# --- ใช้ WangchanBERTa แบบบังคับ: ถ้า import ไม่ได้ → error ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise RuntimeError(
        "ต้องติดตั้งแพ็กเกจสำหรับ WangchanBERTa ก่อน: pip install torch transformers"
    ) from e


# -------------------------
# พาธอินพุต/เอาต์พุต
# -------------------------
PATHS = {
    "GPT": "classified_nonstopwords_GPT_output.csv",
    "Qwen": "classified_nonstopwords_QWEN_output.csv",
    "DeepSeek": "classified_nonstopwords_DEEPSEEK_output.csv",
    "Gemini": "classified_nonstopwords_gemini_output.csv",
}

EXCEL_PATH = "model_eval_all.xlsx"
OUT_SUMMARY = "model_eval_summary_wcb.csv"
OUT_RANKED  = "model_eval_ranked_wcb.csv"
OUT_PAIRK   = "model_pairwise_kappa.csv"
OUT_GROUPS  = "coherence_per_group_wcb.csv"

# โฟลเดอร์สำหรับเซฟรูปกราฟ
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# Tokenizer ดึงคำจากคอลัมน์ Words (ไทย/อังกฤษ/ตัวเลข)
# -------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]+")


def read_labeled_csv(path: str) -> pd.DataFrame:
    """อ่านไฟล์ CSV และรีเนมคอลัมน์ให้เป็นมาตรฐาน"""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_main = pick("Main Label", "Main_label", "MainLabel", "main label", "main")
    c_sub  = pick("Sub Label", "Sub_label", "SubLabel", "sub label", "sub")
    c_group= pick("Sub Group", "Sub_group", "SubGroup", "sub group", "group")
    c_words= pick("Words", "Word", "words", "word")

    if not all([c_main, c_sub, c_group, c_words]):
        raise ValueError(f"คอลัมน์ไม่ครบในไฟล์: {path} (พบ {list(df.columns)})")

    df = df[[c_main, c_sub, c_group, c_words]].rename(columns={
        c_main: "Main Label",
        c_sub: "Sub Label",
        c_group: "Sub Group",
        c_words: "Words"
    })
    return df


def tokenize_words(cell) -> List[str]:
    """
    แยกคำจากคอลัมน์ Words:
    - รองรับตัวคั่น: ',', ';', '|', '、'
    - ใช้ TOKEN_RE ดึงคำ (ไทย/อังกฤษ/ตัวเลข)
    - ลบคำซ้ำในแถว (preserve order)
    """
    if pd.isna(cell):
        return []
    s = str(cell)
    s = s.replace("、", ",").replace(";", ",").replace("|", ",")
    candidates = []
    for part in s.split(","):
        part = part.strip()
        toks = TOKEN_RE.findall(part)
        for t in toks:
            t = t.strip()
            if t:
                candidates.append(t)
    seen = set()
    out = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_wordmap(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    สร้างแผนที่ "คำ → ป้ายกำกับ (Main|Sub|Group)" สำหรับ 1 โมเดล
    พร้อมสถิติ coverage ที่นับแบบ **token ดิบจริง**:
      - raw_tokens_total: จำนวน token ดิบ (นับด้วย TOKEN_RE) ในแต่ละแถว รวมทั้งไฟล์
      - parsed_tokens_total: จำนวนคำหลัง tokenize_words (unique ภายในแถว)
    ทำให้ parsed_tokens_total <= raw_tokens_total ⇒ coverage ≤ 1 เสมอ
    """
    word2label = {}
    raw_tokens_total = 0
    parsed_tokens_total = 0

    for _, r in df.iterrows():
        # ---- นับ "token ดิบ" ด้วย TOKEN_RE (ไม่ตัดซ้ำ) ----
        s = str(r["Words"]).replace("、", ",").replace(";", ",").replace("|", ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        raw_tokens = []
        for part in parts:
            raw_tokens.extend(TOKEN_RE.findall(part))
        raw_tokens_total += len(raw_tokens)

        # ---- นับ "คำที่ parse แล้ว" (unique ภายในแถว) ----
        toks = tokenize_words(r["Words"])
        parsed_tokens_total += len(toks)

        # ---- สร้าง mapping คำ → label ----
        label = f"{str(r['Main Label']).strip()}|{str(r['Sub Label']).strip()}|{str(r['Sub Group']).strip()}"
        for w in toks:
            if w not in word2label:
                word2label[w] = label

    # resolve multi-label ในไฟล์เดียวกัน → เอา label ที่พบบ่อยสุด
    counts = Counter()
    for _, r in df.iterrows():
        label = f"{str(r['Main Label']).strip()}|{str(r['Sub Label']).strip()}|{str(r['Sub Group']).strip()}"
        for w in tokenize_words(r['Words']):
            counts[(w, label)] += 1

    final_map = {}
    unique_words = set(k[0] for k in counts.keys())
    for w in unique_words:
        cand = [(lab, c) for (ww, lab), c in counts.items() if ww == w]
        cand.sort(key=lambda x: (-x[1], x[0]))
        final_map[w] = cand[0][0]

    stats = {
        "raw_tokens_total": raw_tokens_total,
        "parsed_tokens_total": parsed_tokens_total
    }
    return final_map, stats


def entropy(probs: List[float]) -> float:
    """Shannon entropy (บิต)"""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def diversity_metrics(word2label: Dict[str, str]) -> Dict[str, float]:
    """คำนวณ n_words, n_subgroups_used, entropy_bits, max_group_share"""
    grp_counts = Counter()
    for w, lab in word2label.items():
        grp = lab.split("|", 2)[-1].strip()
        grp_counts[grp] += 1

    total = sum(grp_counts.values()) if grp_counts else 1
    probs = [c / total for c in grp_counts.values()]
    ent = entropy(probs)
    max_share = max(probs) if probs else 0.0

    return {
        "n_words": total,
        "n_subgroups_used": len(grp_counts),
        "entropy_bits": ent,
        "max_group_share": max_share
    }


# -------------------------
# Coherence ด้วย WangchanBERTa (STRICT)
# -------------------------

def load_wcb_strict():
    """
    โหลด WangchanBERTa แบบ STRICT:
    - ถ้ามี WCB_LOCAL_DIR → โหลดจากเครื่อง
    - ถ้าไม่มี → พยายามโหลดจาก HF: 'airesearch/wangchanberta-base-att-spm-uncased'
    - โหลดไม่ได้ → error (ไม่มี fallback)
    """
    model_id = os.getenv("WCB_MODEL_ID", "airesearch/wangchanberta-base-att-spm-uncased")
    local_dir = os.getenv("WCB_LOCAL_DIR", "").strip()

    if local_dir:
        if not os.path.isdir(local_dir):
            raise RuntimeError(f"WCB_LOCAL_DIR ถูกตั้งค่าไว้ แต่ไม่พบโฟลเดอร์: {local_dir}")
        name_or_path = local_dir
    else:
        name_or_path = model_id

    try:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModel.from_pretrained(name_or_path)
    except Exception as e:
        raise RuntimeError(
            "โหลด WangchanBERTa ไม่สำเร็จ\n"
            "- ถ้าออฟไลน์: โปรดดาวน์โหลดโมเดลไว้ล่วงหน้า แล้วตั้ง WCB_LOCAL_DIR ให้ถูกต้อง\n"
            "- ถ้าออนไลน์: ตรวจสอบสิทธิ์/การเชื่อมต่อกับ Hugging Face hub และชื่อโมเดล"
        ) from e

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def embed_words_wcb(words: List[str], tokenizer, model, device="cpu", batch_size=64, max_len=64) -> np.ndarray:
    """ฝังคำเป็นเวกเตอร์ด้วย mean pooling ของ last_hidden_state"""
    vecs = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                out = model(input_ids=input_ids, attention_mask=attn)

            last = out.last_hidden_state  # (B, T, H)
            mask = attn.unsqueeze(-1).expand(last.size()).float()
            summed = (last * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            mean = summed / counts
            vecs.append(mean.cpu().numpy())

    if not vecs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(vecs)


def coherence_wcb(word2label: Dict[str, str], tokenizer, model, device="cpu", min_group_size=3):
    """
    คืนค่า:
      - coh (float): coherence ระดับโมเดล (ถ่วงน้ำหนักด้วยขนาดกลุ่ม)
      - per_group (dict): {group: coherence เฉลี่ยภายในกลุ่ม}
      - sizes (dict): {group: จำนวนคำ (unique)}
    """
    groups = defaultdict(list)
    for w, lab in word2label.items():
        grp = lab.split("|", 2)[-1].strip()
        groups[grp].append(w)

    per_group = {}
    sizes = {}
    for g, words in groups.items():
        uniq = list(dict.fromkeys(words))
        sizes[g] = len(uniq)
        if len(uniq) < min_group_size:
            continue

        ve = embed_words_wcb(uniq, tokenizer, model, device=device)
        if len(uniq) < 2:
            per_group[g] = float("nan")
        else:
            sim = cosine_similarity(ve)
            n = sim.shape[0]
            vals = [sim[i, j] for i in range(n) for j in range(i + 1, n)]
            per_group[g] = float(np.mean(vals)) if vals else float("nan")

    weights, values = [], []
    for g, v in per_group.items():
        if v == v:  # ไม่ใช่ NaN
            weights.append(sizes[g])
            values.append(v)
    coh = float(np.average(values, weights=weights)) if values else float("nan")
    return coh, per_group, sizes


# -------------------------
# Cross-model Agreement
# -------------------------

def fleiss_kappa(counts_matrix: np.ndarray) -> float:
    M, k = counts_matrix.shape
    n = np.sum(counts_matrix[0])
    p = np.sum(counts_matrix, axis=0) / (M * n)
    P_i = (np.sum(counts_matrix**2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p**2)
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def agreement_stats(model_maps: Dict[str, Dict[str, str]]):
    names = list(model_maps.keys())
    word_sets = [set(m.keys()) for m in model_maps.values()]
    intersect = set.intersection(*word_sets) if word_sets else set()

    labels = sorted(set(m[w] for m in model_maps.values() for w in intersect))
    lab2idx = {lab: i for i, lab in enumerate(labels)}

    M = len(intersect)
    K = len(labels)
    if M == 0 or K == 0:
        return {
            "n_items_intersection": 0,
            "fleiss_kappa": float("nan"),
            "pairwise_kappa": pd.DataFrame(),
            "majority_match": {n: float("nan") for n in names}
        }

    mat = np.zeros((M, K), dtype=int)
    words_sorted = sorted(intersect)
    for i, w in enumerate(words_sorted):
        for mname, m in model_maps.items():
            lab = m[w]
            mat[i, lab2idx[lab]] += 1

    fk = float(fleiss_kappa(mat))
    per_model_labels = {mname: np.array([lab2idx[m[w]] for w in words_sorted]) for mname, m in model_maps.items()}

    pair_k = pd.DataFrame(index=names, columns=names, dtype=float)

    def cohens_kappa(a: np.ndarray, b: np.ndarray, ncat: int) -> float:
        agree = np.mean(a == b)
        pa = np.bincount(a, minlength=ncat) / len(a)
        pb = np.bincount(b, minlength=ncat) / len(b)
        pe = np.sum(pa * pb)
        if pe == 1:
            return 1.0
        return (agree - pe) / (1 - pe)

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            pair_k.loc[n1, n2] = 1.0 if i == j else float(cohens_kappa(per_model_labels[n1], per_model_labels[n2], K))

    majority_labels_idx = np.argmax(mat, axis=1)
    majority_match = {mname: float(np.mean(per_model_labels[mname] == majority_labels_idx)) for mname in names}

    return {
        "n_items_intersection": M,
        "fleiss_kappa": fk,
        "pairwise_kappa": pair_k,
        "majority_match": majority_match
    }


# -------------------------
# Normalize + Scoring
# -------------------------

def minmax(s: pd.Series) -> pd.Series:
    if s.isnull().all():
        return s
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


# -------------------------
# Utilities: วาดกราฟ + ฝังลง Excel
# -------------------------

def save_bar_series(series: pd.Series, title: str, ylabel: str, filename: str):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    fig.tight_layout()
    path = os.path.join("figs", filename)
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)
    return path

def save_heatmap(df: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots()
    im = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join("figs", filename)
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)
    return path

def save_top_groups_bar(per_group_long: pd.DataFrame, winner: str, topk: int = 10):
    sub = per_group_long[(per_group_long["model"] == winner) & (per_group_long["coherence"].notna())]
    if sub.empty:
        return None
    sub = sub.sort_values("coherence", ascending=False).head(topk)
    fig, ax = plt.subplots()
    ax.barh(sub["group"], sub["coherence"])
    ax.invert_yaxis()
    ax.set_title(f"Top {topk} groups by coherence — {winner}")
    ax.set_xlabel("Coherence (cosine)")
    fig.tight_layout()
    fn = f"top_groups_{winner}.png"
    path = os.path.join("figs", fn)
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)
    return path

def embed_images_to_excel(writer, image_paths: List[str]):
    """สร้างชีต 'charts' แล้ววางรูปภาพตามลำดับลงไป"""
    wb = writer.book
    ws = wb.create_sheet("charts")
    row_anchor = 1
    for p in image_paths:
        if p and os.path.exists(p):
            try:
                img = XLImage(p)
                cell = f"A{row_anchor}"
                ws.add_image(img, cell)
                # ขยับ anchor สำหรับภาพต่อไป (ค่าประมาณ)
                row_anchor += 30
            except Exception as e:
                print(f"[WARN] แนบรูป {p} ลง Excel ไม่สำเร็จ: {e}")


# -------------------------
# main(): จุดเริ่มทำงาน
# -------------------------

def main():
    t0 = time.time()
    log("เริ่มประเมินผล…")

    # ---------- 1) โหลดไฟล์ & Preview ----------
    dfs = {}
    for name, path in PATHS.items():
        log(f"[STEP 1] โหลดไฟล์ของโมเดล {name} → {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"ไม่พบไฟล์อินพุตของโมเดล {name}: {path}")
        df = read_labeled_csv(path)
        dfs[name] = df
        display_df(f"{name} — Preview (คอลัมน์มาตรฐาน)", df)

    # ---------- 2) สร้าง word→label + coverage ----------
    maps = {}
    stats_rows = []
    for name, df in dfs.items():
        log(f"[STEP 2] สร้าง word→label map และคำนวณ coverage สำหรับ {name}")
        wmap, rawstats = build_wordmap(df)
        maps[name] = wmap

        rawtokens = rawstats["raw_tokens_total"]
        cov_raw = (rawstats["parsed_tokens_total"] / rawtokens) if rawtokens else 0.0
        if cov_raw > 1.0 + 1e-9:
            print(f"[WARN] coverage > 1 พบใน {name}: {cov_raw:.6f}  (ควรตรวจข้อมูลแถวที่อาจผิดรูป)")
        cov = min(max(cov_raw, 0.0), 1.0)

        stats_rows.append({
            "model": name,
            "coverage_parse_rate": cov,
            "raw_tokens": rawtokens,
            "parsed_tokens": rawstats["parsed_tokens_total"],
            "unique_words": len(wmap)
        })

        # ตัวอย่าง mapping 10 รายการ
        sample_pairs = list(wmap.items())[:10]
        print(f"\nตัวอย่าง word→label ของ {name} (10 รายการ):")
        for w, lab in sample_pairs:
            print(f"  {w} → {lab}")
        print("—")

    stats_df = pd.DataFrame(stats_rows).set_index("model")
    display_df("Coverage & Word Stats — per model", stats_df)

    # ---------- 3) Diversity metrics ----------
    div_rows = []
    for name, wmap in maps.items():
        log(f"[STEP 3] คำนวณ diversity metrics สำหรับ {name}")
        divm = diversity_metrics(wmap); divm["model"] = name
        div_rows.append(divm)
    div_df = pd.DataFrame(div_rows).set_index("model")
    display_df("Diversity metrics — per model", div_df, round_ndigits=4)

    # ---------- 4) โหลด WangchanBERTa ----------
    log("[STEP 4] โหลด WangchanBERTa (STRICT)…")
    tokenizer, model, device = load_wcb_strict()
    log(f"โหลดสำเร็จ — ใช้งานบนอุปกรณ์: {device.upper()}")

    # ---------- 5) Coherence — per model + top/bottom groups ----------
    coh_rows = []
    per_group_long_rows = []

    for name, wmap in maps.items():
        log(f"[STEP 5] คำนวณ semantic coherence (WCB) สำหรับ {name}")
        coh, per_group, sizes = coherence_wcb(wmap, tokenizer, model, device=device, min_group_size=3)
        coh_rows.append({"model": name, "coherence_wcb": coh,
                         "groups_with_coherence": len([v for v in per_group.values() if v == v])})
        for g, v in per_group.items():
            per_group_long_rows.append({"model": name, "group": g, "coherence": v, "size": sizes.get(g, 0)})

        gdf = pd.DataFrame(
            [{"group": g, "coherence": v, "size": sizes.get(g, 0)}
             for g, v in per_group.items() if v == v]
        ).sort_values("coherence", ascending=False)
        if not gdf.empty:
            display_df(f"{name} — Coherence per group (Top 10)", gdf, head=10, round_ndigits=4)
            display_df(f"{name} — Coherence per group (Bottom 10)", gdf.sort_values("coherence"), head=10, round_ndigits=4)
        else:
            log(f"{name}: ไม่มี group ที่ผ่าน min_group_size สำหรับการคำนวณ coherence")

    coh_df = pd.DataFrame(coh_rows).set_index("model")
    per_group_long = pd.DataFrame(per_group_long_rows)
    display_df("Coherence summary — per model", coh_df, round_ndigits=4)

    # ---------- 6) Cross-model Agreement ----------
    log("[STEP 6] คำนวณ cross-model agreement (Fleiss’ κ, Cohen’s κ, Majority-match)…")
    agree = agreement_stats(maps)
    pair_k = agree["pairwise_kappa"]
    majority_match = pd.Series(agree["majority_match"], name="majority_match_rate")

    log(f"Fleiss’ κ (รวมระบบ): {agree['fleiss_kappa']:.4f}  |  จำนวนคำใน intersection: {agree['n_items_intersection']}")
    if not pair_k.empty:
        display_df("Pairwise Cohen’s κ (4×4)", pair_k, round_ndigits=3)
    else:
        log("ไม่มีตาราง Cohen’s κ (ไม่มีคำร่วมกันระหว่างโมเดลครบทั้ง 4)")

    mm_df = majority_match.to_frame()
    display_df("Majority-match rate — per model", mm_df, round_ndigits=4)

    # ---------- 7) รวมเมตริก → ModelScore + Ranking ----------
    log("[STEP 7] รวมเมตริกทั้งหมด → Normalize + ถ่วงน้ำหนัก → ModelScore → จัดอันดับ")
    summary = pd.concat([
        stats_df[["coverage_parse_rate"]],
        div_df[["n_words", "n_subgroups_used", "entropy_bits", "max_group_share"]],
        coh_df[["coherence_wcb"]],
        majority_match
    ], axis=1)

    norm = pd.DataFrame(index=summary.index)
    norm["cov"]   = minmax(summary["coverage_parse_rate"])
    norm["ent"]   = minmax(summary["entropy_bits"])
    norm["diversity_inverse_conc"] = 1 - minmax(summary["max_group_share"])
    norm["coh"]   = minmax(summary["coherence_wcb"])
    norm["majmatch"] = minmax(summary["majority_match_rate"])

    weights = {
        "cov": 0.1667,
        "ent": 0.2222,
        "diversity_inverse_conc": 0.1667,
        "coh": 0.2778,
        "majmatch": 0.1667
    }
    score = sum(norm[k]*v for k, v in weights.items())
    summary["ModelScore"] = score

    ranked = summary.sort_values("ModelScore", ascending=False)

    display_df("Summary metrics (รวมทุกเมตริก) — per model", summary, round_ndigits=4)
    display_df("Ranking — จัดอันดับด้วย ModelScore", ranked, round_ndigits=4)

    # ---------- 7.1) วาดกราฟ + เซฟ PNG ----------
    log("[STEP 7.1] วาดกราฟและบันทึกเป็นรูป…")
    img_paths = []
    # 1) ModelScore bar (เรียงตามอันดับ)
    img_paths.append(save_bar_series(ranked["ModelScore"], "ModelScore — Ranking", "Score", "plot_modelscore.png"))
    # 2) Coverage, Coherence, Entropy (แยกภาพ)
    img_paths.append(save_bar_series(summary["coverage_parse_rate"].loc[ranked.index], "Coverage parse rate", "Rate", "plot_coverage.png"))
    img_paths.append(save_bar_series(summary["coherence_wcb"].loc[ranked.index], "Semantic coherence (WCB)", "Cosine similarity", "plot_coherence.png"))
    img_paths.append(save_bar_series(summary["entropy_bits"].loc[ranked.index], "Entropy of subgroup distribution", "Bits", "plot_entropy.png"))
    # 3) Heatmap pairwise κ
    if not pair_k.empty:
        img_paths.append(save_heatmap(pair_k, "Pairwise Cohen's κ", "heatmap_pairwise_kappa.png"))
    # 4) Top groups ของโมเดลชนะ
    winner = ranked.index[0]
    img_paths.append(save_top_groups_bar(per_group_long, winner, topk=10))

    # ---------- 8) Save CSV + Excel + ฝังกราฟ ----------
    log("[STEP 8] บันทึกผลลัพธ์เป็น CSV และ Excel หลายชีต + ฝังกราฟ…")
    summary.to_csv(OUT_SUMMARY, encoding="utf-8-sig")
    ranked.to_csv(OUT_RANKED,  encoding="utf-8-sig")
    pair_k.to_csv(OUT_PAIRK,   encoding="utf-8-sig")
    per_group_long.to_csv(OUT_GROUPS, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        meta = pd.DataFrame({
            "key":   ["generated_at", "coherence_method", "fleiss_kappa", "n_items_intersection", "winner_model"],
            "value": [datetime.datetime.now().isoformat(), "WangchanBERTa", agree["fleiss_kappa"], agree["n_items_intersection"], winner]
        })
        meta.to_excel(writer, sheet_name="meta", index=False)
        summary.to_excel(writer, sheet_name="model_eval_summary_wcb")
        ranked.to_excel(writer,  sheet_name="model_eval_ranked_wcb")
        pair_k.to_excel(writer,  sheet_name="model_pairwise_kappa")
        per_group_long.to_excel(writer, sheet_name="coherence_per_group_wcb", index=False)

        # ฝังรูปทั้งหมดไว้ในชีต "charts"
        embed_images_to_excel(writer, [p for p in img_paths if p])

    log("บันทึกเสร็จสิ้น ✓")
    print("\nไฟล์ที่บันทึก:")
    for p in [EXCEL_PATH, OUT_SUMMARY, OUT_RANKED, OUT_PAIRK, OUT_GROUPS]:
        print(" -", os.path.abspath(p))
    print("โฟลเดอร์รูป:", os.path.abspath(FIG_DIR))

    print("\n=== สรุปอันดับโมเดล (STRICT WangchanBERTa) ===")
    for i, (m, r) in enumerate(ranked['ModelScore'].items(), 1):
        print(f"{i}. {m}  (ModelScore={r:.4f})")

    log(f"ใช้เวลา: {time.time()-t0:.2f} วินาที")


# จุดเริ่มรันสคริปต์
if __name__ == "__main__":
    main()
