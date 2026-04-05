from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import optuna
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import NotFoundError
from tqdm.auto import tqdm
import torch

import re

# -----------------------------
# 경로/상수 정의
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

SP500_CSV = BASE_DIR / "csv" / "sp500_list.csv"
TRAIN_JSON = BASE_DIR / "json" / "train_news.json"
TWEET_JSON1 = BASE_DIR / "json" / "train_tweet_joined.json"
TWEET_JSON2 = BASE_DIR / "json" / "train_tweet_stock.json"
KEYWORD_DIR = BASE_DIR / "json" / "summary_json"
INDUSTRY_DIR = BASE_DIR / "json" / "industry_group"

STATIC_CHROMA_PATH = BASE_DIR.parent / "chromaDB" / "static"
DYNAMIC_CHROMA_PATH = BASE_DIR.parent / "chromaDB" / "dynamic"

FINBERT_COL = "finbert"
SBERT_COL = "sbert"

TOP_K = 5

MODEL_DIR = BASE_DIR / "model"
MODEL_PKL = MODEL_DIR / "model.pkl"

# ------------------------------------------------
# 공통 상수: STOPWORDS, BAD_TICKERS, ALIASES
# ------------------------------------------------

# 일반적인 영문 stopword들 (티커/alias 매칭에서 제외)
STOPWORDS: Set[str] = {
    "a", "an", "the",
    "all", "has", "are", "well", "now", "one", "two",
    "and", "or", "not",
    "on", "in", "it", "be", "to", "for", "of",
}

# 티커 형태지만 평범한 단어라서 '티커로는' 매칭하지 않을 것들
BAD_TICKERS: Set[str] = {
    "ALL", "HAS", "ARE", "NOW", "IT", "A", "L", "O", "ON", "WELL", "SO",
    "ONE", "IN", "BE",
}

# 브랜드/제품명 → 티커 alias
# (필요할 때 여기 계속 추가해서 튜닝하면 됨)
ALIASES: Dict[str, List[str]] = {
    # Big Tech 예시
    "AAPL": ["apple", "iphone", "ipad", "macbook", "ios", "imac"],
    "GOOGL": ["google", "android", "youtube", "gmail", "pixel", "waymo"],
    "GOOG":  ["google", "android", "youtube", "gmail", "pixel", "waymo"],
    "META":  ["facebook", "instagram", "whatsapp", "oculus", "meta"],
    "AMZN":  ["amazon", "aws", "prime video", "prime", "alexa"],
    "TSLA":  ["tesla", "model 3", "model y", "cybertruck", "roadster"],
    "NVDA":  ["nvidia", "geforce", "rtx", "cuda"],

    # 다른 회사도 필요하면 여기에 추가
    # 예: "MSFT": ["microsoft", "xbox", "office", "windows"],
}

# 회사-고유 키워드에서 제거하고 싶은 generic 단어들
GENERIC_COMPANY_KEYWORDS: Set[str] = {
    "tech",
    "technology",
    "technologies",
    "system",
    "systems",
    "group",
    "global",
    "international",
    "service",
    "services",
    "solution",
    "solutions",
    "company",
    "companies",
    "corp",
    "corporation",
    "holding",
    "holdings",
    "inc",
    "ltd",
}

# 언급 레벨 정의
MENTION_LEVEL_NONE = 0
MENTION_LEVEL_KEYWORD = 1   # company_keywords / name_keywords
MENTION_LEVEL_ALIAS = 2     # ALIASES
MENTION_LEVEL_TICKER = 3    # 티커 직접 언급

DEFAULT_INDUSTRY_GROUP_WEIGHT = 1.0
DEFAULT_MENTION_BONUS_WEIGHT = 1.0

# 학습 사이즈 관련
BATCH_SIZE = 32  # 필요하면 16, 8로 줄여도 됨
MAX_NEWS = None        # 뉴스는 일단 전부 사용 (None이면 제한 없음)
MAX_JOINED = 10000     # tweet_joined 최대 1만 개
MAX_STOCK = 10000      # tweet_stock 최대 1만 개



# -----------------------------
# 공통 util
# -----------------------------


def load_sp500() -> pd.DataFrame:
    return pd.read_csv(SP500_CSV)


def get_name_column(df: pd.DataFrame) -> str:
    if "security" in df.columns:
        return "security"
    if "company_name" in df.columns:
        return "company_name"
    return "ticker"


def _load_single_json(path: Path, source: str) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Training file not found ({source}): {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    # 어디서 온 데이터인지 구분하고 싶으면 라벨을 하나 추가
    for r in rows:
        r.setdefault("source", source)
    return rows


def load_training_data() -> List[Dict[str, Any]]:
    """
    train_news.json, train_tweet_joined.json, train_tweet_stock.json
    세 JSON을 읽어서 하나의 리스트로 병합한다.

    각 레코드는 최소한 아래 필드를 가진다.
      - description: str
      - sp500_labels: List[str]  (S&P500 티커로 필터링된 라벨)
      - split: "train" / "valid" / "test" (없으면 "train"으로 기본값)

    train_news.json의 경우:
      - sp500_labels가 있으면 그것 + S&P500 교집합
      - 없으면 tickers에서 S&P500 교집합으로 sp500_labels 생성

    tweet_joined / tweet_stock의 경우:
      - sp500_labels에서 S&P500 교집합만 사용
      - 라벨이 하나도 없으면 학습에서 제외
    """

    # S&P500 티커 집합
    sp500_set: Set[str] = set()
    if SP500_CSV.exists():
        df_sp = pd.read_csv(SP500_CSV)
        sp500_set = set(df_sp["ticker"].astype(str).str.upper())

    def _filter_sp500_labels(raw: Any) -> List[str]:
        """
        아무 리스트나 받아서 S&P500에 속하는 ticker만 대문자로 정제해서 반환
        """
        if not raw:
            return []
        labels = [str(t).upper() for t in raw if t]
        if not sp500_set:
            # 혹시라도 S&P500 CSV를 못 읽었다면 그냥 대문자만 만든다
            return labels
        return [t for t in labels if t in sp500_set]

    all_records: List[Dict[str, Any]] = []

    # ----------------------------
    # 1) train_news.json
    # ----------------------------
    if TRAIN_JSON.exists():
        with TRAIN_JSON.open("r", encoding="utf-8") as f:
            news_data = json.load(f)

        news_count = 0
        for rec in tqdm(news_data, desc="Loading train_news.json"):
            # description이 없으면 학습에 쓸 수 없음
            desc = rec.get("description")
            if not desc:
                continue

            # sp500_labels 우선 사용, 없으면 tickers에서 생성
            labels = rec.get("sp500_labels")
            if labels is not None:
                labels = _filter_sp500_labels(labels)
            else:
                labels = _filter_sp500_labels(rec.get("tickers", []))

            # S&P500 라벨이 하나도 없으면 학습에 의미가 없으니 제외
            if not labels:
                continue

            new_rec = dict(rec)
            new_rec["sp500_labels"] = labels
            if "split" not in new_rec:
                new_rec["split"] = "train"

            all_records.append(new_rec)
            news_count += 1

            if MAX_NEWS is not None and news_count >= MAX_NEWS:
                break

    # ----------------------------
    # 2) train_tweet_joined.json
    # ----------------------------
    if TWEET_JSON1.exists():
        with TWEET_JSON1.open("r", encoding="utf-8") as f:
            tweet_joined = json.load(f)

        joined_records: List[Dict[str, Any]] = []
        for rec in tqdm(tweet_joined, desc="Loading train_tweet_joined.json"):
            desc = rec.get("description")
            if not desc:
                continue

            labels = _filter_sp500_labels(rec.get("sp500_labels", []))
            if not labels:
                continue  # 라벨 없는 트윗은 Hit@K 학습에 사용 불가

            new_rec = dict(rec)
            new_rec["sp500_labels"] = labels
            if "split" not in new_rec:
                new_rec["split"] = "train"

            joined_records.append(new_rec)
            if MAX_JOINED is not None and len(joined_records) >= MAX_JOINED:
                break

        all_records.extend(joined_records)

    # ----------------------------
    # 3) train_tweet_stock.json
    # ----------------------------
    if TWEET_JSON2.exists():
        with TWEET_JSON2.open("r", encoding="utf-8") as f:
            tweet_stock = json.load(f)

        stock_records: List[Dict[str, Any]] = []
        for rec in tqdm(tweet_stock, desc="Loading train_tweet_stock.json"):
            desc = rec.get("description")
            if not desc:
                continue

            labels = _filter_sp500_labels(rec.get("sp500_labels", []))
            if not labels:
                continue

            new_rec = dict(rec)
            new_rec["sp500_labels"] = labels
            if "split" not in new_rec:
                new_rec["split"] = "train"

            stock_records.append(new_rec)
            if MAX_STOCK is not None and len(stock_records) >= MAX_STOCK:
                break

        all_records.extend(stock_records)

    if not all_records:
        raise RuntimeError(
            "No training data found. "
            f"Checked: {TRAIN_JSON}, {TWEET_JSON1}, {TWEET_JSON2}"
        )

    print(f"[INFO] Loaded {len(all_records)} raw training records.")
    return all_records



def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def load_company_keywords() -> Dict[str, set[str]]:
    """
    KEYWORD_DIR 안의 *_keyword.json 파일들을 읽어서
    ticker -> set(keywords) 형태의 딕셔너리를 만든다.
    """
    keywords_by_ticker: Dict[str, set[str]] = {}
    if not KEYWORD_DIR.exists():
        print(f"[WARN] KEYWORD_DIR not found: {KEYWORD_DIR}")
        return keywords_by_ticker

    json_paths = sorted(KEYWORD_DIR.glob("*_keyword.json"))
    print(f"[INFO] Loading company keywords from {len(json_paths)} files in {KEYWORD_DIR}")

    for path in tqdm(json_paths, desc="Loading company keyword JSONs"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ticker = data["ticker"]
        static_kws = data.get("static_keywords", []) or data.get("static", [])
        dynamic_kws = data.get("dynamic_keywords", []) or data.get("dynamic", [])
        keywords_by_ticker[ticker] = set(static_kws) | set(dynamic_kws)

    return keywords_by_ticker



def filter_company_keywords(
    raw_keywords: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """
    company_keywords에서 너무 일반적인 generic 단어들을 제거한다.

    예: "tech", "technology", "group", "global", "company" 등은
        특정 회사만을 가리키지 않으므로 mention 탐지에 쓰지 않게 필터링.

    반환값은 원본과 동일한 구조의 dict 이지만, 각 ticker의 keyword set에서
    GENERIC_COMPANY_KEYWORDS 에 포함된 단어들이 빠져 있다.
    """
    if not raw_keywords:
        return raw_keywords

    generic_lower = {w.lower() for w in GENERIC_COMPANY_KEYWORDS}
    filtered: Dict[str, Set[str]] = {}

    for ticker, kws in raw_keywords.items():
        if not kws:
            filtered[ticker] = set()
            continue

        keep: Set[str] = set()
        for kw in kws:
            if not kw:
                continue
            kw_stripped = kw.strip()
            if not kw_stripped:
                continue

            kw_norm = kw_stripped.lower()
            if kw_norm in generic_lower:
                # "tech", "global" 같은 generic 단어는 버림
                continue

            keep.add(kw_stripped)

        filtered[ticker] = keep

    return filtered

def load_industry_group_keywords() -> Dict[str, set[str]]:
    """
    INDUSTRY_DIR 안의 *.json 파일에서 industry_group별 keywords를 읽어온다.
    """
    group_kws: Dict[str, set[str]] = {}
    if not INDUSTRY_DIR.exists():
        print(f"[WARN] INDUSTRY_DIR not found: {INDUSTRY_DIR}")
        return group_kws

    json_paths = sorted(INDUSTRY_DIR.glob("*.json"))
    print(f"[INFO] Loading industry-group keywords from {len(json_paths)} files in {INDUSTRY_DIR}")

    for path in tqdm(json_paths, desc="Loading industry-group keyword JSONs"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        group = data["industry_group"]
        kws = set(data.get("keywords", []))
        group_kws[group] = kws

    return group_kws



def build_ticker_to_group(df_sp500: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(df_sp500["ticker"], df_sp500["industry_group"]))


def load_company_embeddings(
    static_client,
    dynamic_client,
) -> Dict[str, Dict[str, np.ndarray]]:
    def safe_get_collection(client, name: str):
        try:
            return client.get_collection(name)
        except NotFoundError:
            print(f"[ERROR] Collection '{name}' not found.")
            print("        Available collections:")
            for c in client.list_collections():
                print("         -", c.name)
            raise

    col_static_fin = safe_get_collection(static_client, FINBERT_COL)
    col_static_sbt = safe_get_collection(static_client, SBERT_COL)
    col_dyn_fin = safe_get_collection(dynamic_client, FINBERT_COL)
    col_dyn_sbt = safe_get_collection(dynamic_client, SBERT_COL)

    res_static_fin = col_static_fin.get(include=["embeddings"])
    res_static_sbt = col_static_sbt.get(include=["embeddings"])
    res_dyn_fin = col_dyn_fin.get(include=["embeddings"])
    res_dyn_sbt = col_dyn_sbt.get(include=["embeddings"])

    def build_emb_dict(res, desc: str) -> Dict[str, np.ndarray]:
        ids = res.get("ids") or []
        embs = res.get("embeddings")
        if embs is None:
            embs = []
        out: Dict[str, np.ndarray] = {}

        for _id, emb in tqdm(
            list(zip(ids, embs)),
            total=len(ids),
            desc=desc,
        ):
            if emb is None:
                continue
            if isinstance(emb, list):
                emb_arr = np.array(emb, dtype=np.float32)
            else:
                emb_arr = np.asarray(emb, dtype=np.float32)
            out[_id] = emb_arr

        return out

    print("[INFO] Building static FinBERT embedding dict...")
    static_fin_dict = build_emb_dict(res_static_fin, "Static FinBERT")

    print("[INFO] Building static SBERT embedding dict...")
    static_sbt_dict = build_emb_dict(res_static_sbt, "Static SBERT")

    print("[INFO] Building dynamic FinBERT embedding dict...")
    dyn_fin_dict = build_emb_dict(res_dyn_fin, "Dynamic FinBERT")

    print("[INFO] Building dynamic SBERT embedding dict...")
    dyn_sbt_dict = build_emb_dict(res_dyn_sbt, "Dynamic SBERT")

    # 공통 ticker만 사용
    common_ids = (
        set(static_fin_dict.keys())
        & set(static_sbt_dict.keys())
        & set(dyn_fin_dict.keys())
        & set(dyn_sbt_dict.keys())
    )

    print(f"[INFO] Common tickers in all 4 embedding sets: {len(common_ids)}")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ticker in common_ids:
        out[ticker] = {
            "feature1_finbert": static_fin_dict[ticker],
            "feature1_sbert": static_sbt_dict[ticker],
            "feature2_finbert": dyn_fin_dict[ticker],
            "feature2_sbert": dyn_sbt_dict[ticker],
        }

    return out



def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------
# 회사 이름 기반 keyword
# -----------------------------


def build_name_keywords(df_sp500) -> Dict[str, Set[str]]:
    """
    각 ticker에 대해 회사명에서 '고유명사스럽게' 쓸만한 토큰만 뽑아서 keyword set을 만든다.
    예:
      "Meta Platforms, Inc." -> {"meta", "platforms"}
      "Tesla, Inc."          -> {"tesla"}
      "Alphabet Inc. Class A"-> {"alphabet", "class"} (class가 너무 흔하면 나중에 필터링해도 됨)

    규칙:
      - 콤마(,) 앞까지만 회사명으로 사용
      - 알파벳/숫자 토큰 단위로 쪼갬
      - 길이 >= 3
      - STOPWORDS 에 없는 것만 사용
    """
    mapping: Dict[str, Set[str]] = {}

    if "security" in df_sp500.columns:
        name_col = "security"
    elif "company_name" in df_sp500.columns:
        name_col = "company_name"
    else:
        name_col = "ticker"

    for _, row in df_sp500.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        full_name = str(row[name_col])

        # 콤마 이전까지만 사용: "Tesla, Inc." -> "Tesla"
        base_name = full_name.split(",")[0].lower()

        # 단어 단위 tokenize
        tokens = re.findall(r"\b\w+\b", base_name)

        kws = {
            tok
            for tok in tokens
            if len(tok) >= 3 and tok not in STOPWORDS
        }

        mapping[ticker] = kws

    return mapping

# ------------------------------------------------
# 3) 텍스트에서 언급된 ticker 탐지
# ------------------------------------------------

def detect_mentions_in_text(
    text: str,
    ticker_to_name_keywords: Dict[str, Set[str]] | None,
    company_keywords: Dict[str, Set[str]] | None = None,
) -> Set[str]:
    """
    기사/트윗 텍스트에서 기업 언급 탐지.

    1) 정확한 ticker 매칭
       - BAD_TICKERS 에 포함된 티커는 무시
       - word boundary 기준으로만 매칭

    2) 회사-고유 키워드 매칭 (company_keywords, JSON 기반)
       - 예: AAPL → {iphone, macbook, app store, ios, imac, ...}
             TSLA → {tesla, model 3, model y, autopilot, ev, battery, ...}

    3) (보조) 회사 이름 keyword 매칭 (build_name_keywords)
       - 너무 일반적인 단어는 STOPWORDS / 길이 기준으로 필터링

    4) alias(브랜드/제품명) 매칭
       - ALIASES dict 사용 (기존 유지)
    """

    mentioned: Set[str] = set()
    text_raw = text or ""
    text_l = text_raw.lower()

    # ---------- 1) 티커 직접 매칭 ----------
    for ticker in (ticker_to_name_keywords or {}).keys():
        if ticker in BAD_TICKERS:
            continue
        pattern = rf"\b{re.escape(ticker)}\b"
        if re.search(pattern, text_raw):
            mentioned.add(ticker)

    # ---------- 2) 회사-고유 키워드(JSON 기반) 매칭 ----------
    if company_keywords:
        for ticker, kw_set in company_keywords.items():
            for kw in kw_set:
                if not kw:
                    continue
                kw_l = kw.lower().strip()
                if not kw_l:
                    continue
                if kw_l in STOPWORDS:
                    continue
                # 너무 짧은 단어는 노이즈라서 제외 (예: "ai", "vr" 등은 필요시 나중에 whitelist로 처리)
                if len(kw_l) < 3:
                    continue
                pattern = rf"\b{re.escape(kw_l)}\b"
                if re.search(pattern, text_l):
                    mentioned.add(ticker)
                    break  # 한 keyword라도 걸리면 그 ticker는 언급된 것으로 처리

    # ---------- 3) 회사 이름 keyword 매칭 (보조 용도) ----------
    if ticker_to_name_keywords:
        for ticker, name_kws in ticker_to_name_keywords.items():
            for kw in name_kws:
                if not kw:
                    continue
                kw_l = kw.lower().strip()
                if not kw_l:
                    continue

                # ★ generic 키워드 & stopwords & 너무 짧은 단어는 무시
                if kw_l in GENERIC_COMPANY_KEYWORDS:
                    continue
                if kw_l in STOPWORDS:
                    continue
                if len(kw_l) < 3:
                    continue

                pattern = rf"\b{re.escape(kw_l)}\b"
                if re.search(pattern, text_l):
                    mentioned.add(ticker)
                    break

    # ---------- 4) alias(브랜드/제품명) 매칭 ----------
    for ticker, alias_list in ALIASES.items():
        for alias in alias_list:
            if not alias:
                continue
            a = alias.lower()
            if a in STOPWORDS:
                continue
            pattern = rf"\b{re.escape(a)}\b"
            if re.search(pattern, text_l):
                mentioned.add(ticker)
                break

    return mentioned



def build_query_repr(
    data: List[Dict],
    finbert_model: SentenceTransformer,
    sbert_model: SentenceTransformer,
    kw_model: KeyBERT,
    ticker_to_name: Dict[str, str],
    ticker_to_name_keywords: Dict[str, Set[str]],
    company_keywords: Dict[str, Set[str]],
) -> List[Dict]:
    """
    각 row(description)에 대해
      - q_fin, q_sbert
      - query_keywords
      - mention_levels
    를 미리 계산해서 캐싱.

    ※ FinBERT / SBERT는 batch encoding으로 속도 최적화
    """

    # 1) description 있는 row만 텍스트/인덱스 분리
    texts: List[str] = []
    indices: List[int] = []
    for i, rec in enumerate(data):
        desc = rec.get("description")
        if not desc:
            continue
        text_clean = " ".join(str(desc).split())
        texts.append(text_clean)
        indices.append(i)

    if not texts:
        print("[WARN] No descriptions found in training data.")
        return []

    print(f"[INFO] Encoding {len(texts)} texts with FinBERT (batch_size={BATCH_SIZE})...")
    q_fin_all = finbert_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(f"[INFO] Encoding {len(texts)} texts with SBERT (batch_size={BATCH_SIZE})...")
    q_sbt_all = sbert_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    out: List[Dict] = []

    print("[INFO] Running KeyBERT + mention level per text...")
    for emb_idx, rec_idx in tqdm(
        list(enumerate(indices)),
        total=len(indices),
        desc="Post-processing (keywords & mention levels)",
    ):
        rec = data[rec_idx]
        text_clean = texts[emb_idx]

        q_fin = q_fin_all[emb_idx]
        q_sbt = q_sbt_all[emb_idx]

        # KeyBERT → top 10 키워드
        kw_candidates = kw_model.extract_keywords(
            text_clean,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=10,
        )
        q_kws: Set[str] = {w for w, _score in kw_candidates}

        # 회사 이름/티커 기반 mention level 계산
        mention_levels = compute_mention_levels(
            text_clean,
            ticker_to_name_keywords,
            company_keywords,
        )

        item = dict(rec)
        item["q_fin"] = q_fin
        item["q_sbert"] = q_sbt
        item["query_keywords"] = list(q_kws)
        item["mention_levels"] = mention_levels

        out.append(item)

    print(f"[INFO] Built query representations for {len(out)} records.")
    return out




def compute_mention_levels(
    text: str,
    ticker_to_name_keywords: Dict[str, Set[str]] | None,
    company_keywords: Dict[str, Set[str]] | None = None,
) -> Dict[str, int]:
    """
    텍스트에서 각 ticker가 어느 '레벨'로 언급됐는지 계산한다.

    - Level 3: 티커 직접 언급 (TSLA, AAPL 등)
    - Level 2: alias(브랜드명)로 언급
    - Level 1: 회사-고유 키워드 또는 회사 이름 키워드로 언급
    - Level 0: 언급 없음

    반환: {ticker: level}
    """
    levels: Dict[str, int] = {}
    text_raw = text or ""
    text_l = text_raw.lower()

    def update_level(ticker: str, level: int) -> None:
        if not ticker:
            return
        prev = levels.get(ticker, MENTION_LEVEL_NONE)
        if level > prev:
            levels[ticker] = level

    # ---------- 1) 티커 직접 매칭 ----------
    if ticker_to_name_keywords:
        for ticker in ticker_to_name_keywords.keys():
            if ticker in BAD_TICKERS:
                continue
            pattern = rf"\b{re.escape(ticker)}\b"
            if re.search(pattern, text_raw):
                update_level(ticker, MENTION_LEVEL_TICKER)

    # ---------- 2) 회사-고유 키워드(company_keywords) ----------
    if company_keywords:
        generic_lower = {w.lower() for w in GENERIC_COMPANY_KEYWORDS}
        for ticker, kw_set in company_keywords.items():
            for kw in kw_set:
                if not kw:
                    continue
                kw_l = kw.lower().strip()
                if not kw_l:
                    continue
                if kw_l in STOPWORDS:
                    continue
                if kw_l in generic_lower:
                    continue
                if len(kw_l) < 3:
                    continue
                pattern = rf"\b{re.escape(kw_l)}\b"
                if re.search(pattern, text_l):
                    update_level(ticker, MENTION_LEVEL_KEYWORD)
                    break

    # ---------- 3) 회사 이름 keyword (build_name_keywords) ----------
    if ticker_to_name_keywords:
        generic_lower = {w.lower() for w in GENERIC_COMPANY_KEYWORDS}
        for ticker, name_kws in ticker_to_name_keywords.items():
            for kw in name_kws:
                if not kw:
                    continue
                kw_l = kw.lower().strip()
                if not kw_l:
                    continue
                if kw_l in STOPWORDS:
                    continue
                if kw_l in generic_lower:
                    continue
                if len(kw_l) < 3:
                    continue
                pattern = rf"\b{re.escape(kw_l)}\b"
                if re.search(pattern, text_l):
                    update_level(ticker, MENTION_LEVEL_KEYWORD)
                    break

    # ---------- 4) alias(브랜드/제품명) ----------
    for ticker, alias_list in ALIASES.items():
        for alias in alias_list:
            if not alias:
                continue
            a = alias.lower().strip()
            if not a:
                continue
            if a in STOPWORDS:
                continue
            pattern = rf"\b{re.escape(a)}\b"
            if re.search(pattern, text_l):
                update_level(ticker, MENTION_LEVEL_ALIAS)
                break

    return levels


def compute_pair_score(
    query_fin: np.ndarray,
    query_sbert: np.ndarray,
    query_keywords: List[str],
    mention_levels: Dict[str, int],   # ✅ 기존 mentioned_tickers 대신 level dict 사용
    ticker: str,
    company_emb: Dict[str, Dict[str, np.ndarray]],
    company_keywords: Dict[str, Set[str]],
    ticker_to_group: Dict[str, str],
    group_keywords: Dict[str, Set[str]],
    alpha1: float,
    alpha2: float,
    beta1: float,
    beta2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> float:
    """
    하나의 (query, ticker) 쌍에 대한 최종 점수 계산.

    - Base score = a1*static_sbert + a2*static_finbert
                 + b1*dynamic_sbert + b2*dynamic_finbert
      (여기서 a1=alpha2, a2=alpha1, b1=beta2, b2=beta1)
    - keyword: query vs 회사 키워드 교집합 개수 * lambda1
    - industry: 인더스트리 키워드 매칭 여부(0/1) * lambda2
    - mention: mention level(0~3) * lambda3
    """
    emb = company_emb.get(ticker)
    if emb is None:
        return 0.0

    # 1) 회사 임베딩 (static/dynamic × fin/sbert)
    cf1_fin = emb["feature1_finbert"]
    cf1_sbt = emb["feature1_sbert"]
    cf2_fin = emb["feature2_finbert"]
    cf2_sbt = emb["feature2_sbert"]

    # 2) 코사인 유사도 (numpy 기반)
    sim_fin1 = cosine(query_fin, cf1_fin)
    sim_sbt1 = cosine(query_sbert, cf1_sbt)
    sim_fin2 = cosine(query_fin, cf2_fin)
    sim_sbt2 = cosine(query_sbert, cf2_sbt)

    base_sim = (
        alpha1 * sim_fin1
        + alpha2 * sim_sbt1
        + beta1 * sim_fin2
        + beta2 * sim_sbt2
    )

    # 3) keyword overlap (query_keywords vs company_keywords[ticker])
    c_kws = company_keywords.get(ticker, set())
    if not isinstance(c_kws, set):
        c_kws = set(c_kws)
    kw_overlap = len(set(query_keywords) & c_kws)
    kw_part = lambda1 * float(kw_overlap)

    # 4) same industry 여부 (query_keywords ∩ industry_group keywords)
    same_industry = 0.0
    group = ticker_to_group.get(ticker)
    if group is not None:
        grp_kws = group_keywords.get(group, set())
        if set(query_keywords) & set(grp_kws):
            same_industry = 1.0
    industry_part = lambda2 * same_industry

    # 5) mention level (0~3) 보너스
    level = 0
    if mention_levels is not None:
        level = int(mention_levels.get(ticker, 0))
        if level < 0:
            level = 0
    mention_part = lambda3 * float(level)

    final_score = base_sim + kw_part + industry_part + mention_part
    return final_score



def evaluate_hit_at_k(
    query_data: List[Dict[str, Any]],
    all_tickers: List[str],
    company_emb: Dict[str, Dict[str, np.ndarray]],
    company_keywords: Dict[str, Set[str]],
    ticker_to_group: Dict[str, str],
    group_keywords: Dict[str, Set[str]],
    alpha1: float,
    alpha2: float,
    beta1: float,
    beta2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    top_k: int = 5,
) -> float:
    """
    validation query들에 대해 Hit@K를 계산.

    각 query q는 아래 필드를 가진다고 가정:
      - q["q_fin"]
      - q["q_sbert"]
      - q["query_keywords"]
      - q["mention_levels"]
      - q["sp500_labels"]
    """
    hits = 0
    total = 0

    for q in tqdm(query_data, desc=f"Eval Hit@{top_k} (valid)"):
        labels = q.get("sp500_labels") or []
        if not labels:
            continue
        labels_set = set(labels)

        query_fin = q["q_fin"]
        query_sbert = q["q_sbert"]
        query_keywords = list(q.get("query_keywords", []))
        mention_levels = q.get("mention_levels", {})

        scores: List[Tuple[str, float]] = []

        for ticker in all_tickers:
            if ticker not in company_emb:
                continue

            s = compute_pair_score(
                query_fin,
                query_sbert,
                query_keywords,
                mention_levels,
                ticker,
                company_emb,
                company_keywords,
                ticker_to_group,
                group_keywords,
                alpha1,
                alpha2,
                beta1,
                beta2,
                lambda1,
                lambda2,
                lambda3,
            )
            scores.append((ticker, s))

        if not scores:
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scores[:top_k]]

        total += 1
        if labels_set.intersection(top):
            hits += 1

    if total == 0:
        return 0.0
    return hits / total


# -----------------------------
# main
# -----------------------------

def main() -> None:
    df_sp500 = load_sp500()

    # 1) 회사 이름 기반 keyword 사전 (mention 탐지에 사용)
    ticker_to_name_keywords = build_name_keywords(df_sp500)

    # 2) ticker -> full company name 매핑 (기존 그대로)
    name_col = get_name_column(df_sp500)
    ticker_to_name = dict(zip(df_sp500["ticker"], df_sp500[name_col]))

    # 3) 학습 데이터 로드 + 각종 매핑/벡터 준비
    training_data_raw = load_training_data()
    ticker_to_group = build_ticker_to_group(df_sp500)
    all_tickers = sorted(df_sp500["ticker"].unique())

    company_keywords = load_company_keywords()
    company_keywords = filter_company_keywords(company_keywords)
    group_keywords = load_industry_group_keywords()

    static_client = chromadb.PersistentClient(path=str(STATIC_CHROMA_PATH))
    dynamic_client = chromadb.PersistentClient(path=str(DYNAMIC_CHROMA_PATH))
    company_emb = load_company_embeddings(static_client, dynamic_client)

    finbert_model = SentenceTransformer("yiyanghkust/finbert-tone")
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sbert_model)

    # 4) 전체 query repr 생성 (여기서 각 row의 description(desc)을 사용해서 mention 계산)
    query_data_all = build_query_repr(
    training_data_raw,
    finbert_model,
    sbert_model,
    kw_model,
    ticker_to_name,
    ticker_to_name_keywords,
    company_keywords,
    )

    # split에 따라 분리 (정제 라벨: sp500_labels)
    train_queries = [
        q
        for q, r in zip(query_data_all, training_data_raw)
        if r.get("split") == "train" and r.get("sp500_labels")
    ]
    valid_queries = [
        q
        for q, r in zip(query_data_all, training_data_raw)
        if r.get("split") == "valid" and r.get("sp500_labels")
    ]

    print(f"[INFO] #train queries: {len(train_queries)}")
    print(f"[INFO] #valid queries: {len(valid_queries)}")

    # train_queries는 현재 objective 계산에는 직접 안 쓰고,
    # 필요하면 나중에 regularization이나 별도 metric에 활용 가능
    # 지금은 valid_queries만으로 Hit@K 최적화

    def objective(trial: optuna.Trial) -> float:
        alpha1 = trial.suggest_float("alpha1", 0.0, 2.0)
        alpha2 = trial.suggest_float("alpha2", 0.0, 2.0)
        beta1 = trial.suggest_float("beta1", 0.0, 2.0)
        beta2 = trial.suggest_float("beta2", 0.0, 2.0)
        lambda1 = trial.suggest_float("lambda1", 0.0, 2.0)
        lambda2 = trial.suggest_float("lambda2", 0.0, 2.0)
        lambda3 = trial.suggest_float("lambda3", 0.5, 5.0)

        hit_k = evaluate_hit_at_k(
            valid_queries,
            all_tickers,
            company_emb,
            company_keywords,
            ticker_to_group,
            group_keywords,
            alpha1,
            alpha2,
            beta1,
            beta2,
            lambda1,
            lambda2,
            lambda3,
            top_k=TOP_K,
        )
        return hit_k


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("[RESULT] Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"[RESULT] Best Hit@{TOP_K} (valid): {study.best_value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_state = {
        "alpha1": study.best_params["alpha1"],
        "alpha2": study.best_params["alpha2"],
        "beta1": study.best_params["beta1"],
        "beta2": study.best_params["beta2"],
        "lambda1": study.best_params["lambda1"],
        "lambda2": study.best_params["lambda2"],
        "lambda3": study.best_params["lambda3"],
        "top_k": TOP_K,
        "best_hit_at_k_valid": study.best_value,
    }
    with MODEL_PKL.open("wb") as f:
        pickle.dump(model_state, f)
    print(f"[RESULT] Saved best model config to {MODEL_PKL}")




if __name__ == "__main__":
    main()
