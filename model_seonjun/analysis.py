from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import NotFoundError
from tqdm.auto import tqdm
import torch

import re

BASE_DIR = Path(__file__).resolve().parent

SP500_CSV = BASE_DIR / "csv" / "sp500_list.csv"
TRAIN_JSON = BASE_DIR / "json" / "training2_clean.json"
KEYWORD_DIR = BASE_DIR / "json" / "keyword"
INDUSTRY_DIR = BASE_DIR / "json" / "industry_group"

STATIC_CHROMA_PATH = BASE_DIR / "chromaDB" / "static"
DYNAMIC_CHROMA_PATH = BASE_DIR / "chromaDB" / "dynamic"

FINBERT_COL = "finbert"
SBERT_COL = "sbert"

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


TEXT_MENTION_BONUS: float = 1.2
# 언급 레벨 정의
BASE_MENTION_BONUS: float = 0.3
MENTION_LEVEL_NONE = 0
MENTION_LEVEL_KEYWORD = 1   # company_keywords / name_keywords
MENTION_LEVEL_ALIAS = 2     # ALIASES
MENTION_LEVEL_TICKER = 3    # 티커 직접 언급

# -----------------------------
# util
# -----------------------------


def load_sp500() -> pd.DataFrame:
    return pd.read_csv(SP500_CSV)


def get_name_column(df: pd.DataFrame) -> str:
    if "security" in df.columns:
        return "security"
    if "company_name" in df.columns:
        return "company_name"
    return "ticker"


def load_training_data() -> List[Dict[str, Any]]:
    with TRAIN_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


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

        ticker = data.get("ticker")
        if not ticker:
            continue

        static_kws = set(data.get("static_keywords", []))
        dyn_kws = set(data.get("dynamic_keywords", []))
        all_kws = static_kws | dyn_kws

        keywords_by_ticker[ticker] = all_kws

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
            if isinstance(emb, list):
                emb_arr = np.array(emb, dtype=np.float32)
            else:
                emb_arr = np.asarray(emb, dtype=np.float32)
            out[_id] = emb_arr

        return out

    print("[INFO] Building static FinBERT embedding dict...")
    emb_static_fin = build_emb_dict(res_static_fin, "Static FinBERT")

    print("[INFO] Building static SBERT embedding dict...")
    emb_static_sbt = build_emb_dict(res_static_sbt, "Static SBERT")

    print("[INFO] Building dynamic FinBERT embedding dict...")
    emb_dyn_fin = build_emb_dict(res_dyn_fin, "Dynamic FinBERT")

    print("[INFO] Building dynamic SBERT embedding dict...")
    emb_dyn_sbt = build_emb_dict(res_dyn_sbt, "Dynamic SBERT")

    # 공통 ticker만 사용
    common_ids = (
        set(emb_static_fin.keys())
        & set(emb_static_sbt.keys())
        & set(emb_dyn_fin.keys())
        & set(emb_dyn_sbt.keys())
    )

    print(f"[INFO] Common tickers in all 4 embedding sets: {len(common_ids)}")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ticker in common_ids:
        out[ticker] = {
            "feature1_finbert": emb_static_fin[ticker],
            "feature1_sbert": emb_static_sbt[ticker],
            "feature2_finbert": emb_dyn_fin[ticker],
            "feature2_sbert": emb_dyn_sbt[ticker],
        }

    return out



def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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
    2) 회사-고유 키워드(JSON 기반) 매칭
    3) (보조) 회사 이름 keyword 매칭
    4) alias(브랜드/제품명) 매칭
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
                if len(kw_l) < 3:
                    continue
                pattern = rf"\b{re.escape(kw_l)}\b"
                if re.search(pattern, text_l):
                    mentioned.add(ticker)
                    break

    # ---------- 3) 회사 이름 keyword 매칭 (보조) ----------
    if ticker_to_name_keywords:
        for ticker, name_kws in ticker_to_name_keywords.items():
            for kw in name_kws:
                if not kw:
                    continue
                kw_l = kw.lower().strip()
                if not kw_l:
                    continue

                # ★ 여기에도 generic 필터 추가
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



def encode_query(
    text: str,
    finbert_model: SentenceTransformer,
    sbert_model: SentenceTransformer,
    kw_model: KeyBERT,
) -> Dict[str, Any]:
    text_clean = " ".join(text.split())
    q_fin = finbert_model.encode([text_clean], convert_to_numpy=True)[0]
    q_sbert = sbert_model.encode([text_clean], convert_to_numpy=True)[0]
    kw_candidates = kw_model.extract_keywords(
        text_clean, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=10
    )
    q_kws = {w for w, _ in kw_candidates}
    return {
        "text": text_clean,
        "q_fin": q_fin,
        "q_sbert": q_sbert,
        "query_keywords": q_kws,
    }


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
    mention_levels: Dict[str, int],
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
      (a1=alpha2, a2=alpha1, b1=beta2, b2=beta1)
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


# -----------------------------
# 질적 평가 + Hit@K (test split)
# -----------------------------

def qualitative_eval_on_test(
    n_samples: int,
    params: Dict[str, float],
    top_k: int,
    finbert_model: SentenceTransformer,
    sbert_model: SentenceTransformer,
    kw_model: KeyBERT,
    df_sp500: pd.DataFrame,
    test_data: List[Dict[str, Any]],
    company_emb: Dict[str, Dict[str, np.ndarray]],
    company_keywords: Dict[str, set[str]],
    ticker_to_group: Dict[str, str],
    group_keywords: Dict[str, set[str]],
    ticker_to_name_keywords: Dict[str, Set[str]],
    ticker_to_name: Dict[str, str],
) -> None:
    all_tickers = sorted(df_sp500["ticker"].unique())

    if not test_data:
        print("[WARN] No test data.")
        return

    # n_samples 개만 뽑아서 qualitative evaluation
    samples = test_data[:n_samples]

    for i, row in enumerate(samples, start=1):
        desc = row["description"]
        true_tickers = row.get("sp500_labels", [])

        q = encode_query(desc, finbert_model, sbert_model, kw_model)
        mention_levels = compute_mention_levels(
            desc,
            ticker_to_name_keywords,
            company_keywords,
        )

        scores: List[Tuple[str, float]] = []
        for ticker in all_tickers:
            if ticker not in company_emb:
                continue
            s = compute_pair_score(
                q["q_fin"],
                q["q_sbert"],
                list(q["query_keywords"]),
                mention_levels,
                ticker,
                company_emb,
                company_keywords,
                ticker_to_group,
                group_keywords,
                params["alpha1"],
                params["alpha2"],
                params["beta1"],
                params["beta2"],
                params["lambda1"],
                params["lambda2"],
                params["lambda3"],
            )
            scores.append((ticker, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        print(f"\n=== SAMPLE {i} / {len(samples)} ===")
        print("[DESCRIPTION]")
        print(desc.strip())
        print()
        print("[LABEL TICKERS (sp500_labels)]")
        print(true_tickers)
        print()
        print(f"[TOP-{top_k} PREDICTIONS]")
        for t, s in top:
            name = ticker_to_name.get(t, t)
            mark = "✓" if t in true_tickers else " "
            print(f" {mark} {t:5s}  {name:40s}  {s:.4f}")
        print()



def compute_hit_at_k_on_test(
    test_data: List[Dict[str, Any]],
    params: Dict[str, float],
    top_k: int,
    finbert_model: SentenceTransformer,
    sbert_model: SentenceTransformer,
    kw_model: KeyBERT,
    df_sp500: pd.DataFrame,
    company_emb: Dict[str, Dict[str, np.ndarray]],
    company_keywords: Dict[str, set[str]],
    ticker_to_group: Dict[str, str],
    group_keywords: Dict[str, set[str]],
    ticker_to_name_keywords: Dict[str, Set[str]],
) -> float:
    all_tickers = sorted(df_sp500["ticker"].unique())
    hits = 0
    total = 0

    for row in tqdm(
        test_data,
        desc=f"Compute Hit@{top_k} on TEST (S&P500 labels only)",
    ):
        desc = row["description"]
        true_tickers = set(row.get("sp500_labels", []))
        if not true_tickers:
            continue

        q = encode_query(desc, finbert_model, sbert_model, kw_model)
        mention_levels = compute_mention_levels(
            desc,
            ticker_to_name_keywords,
            company_keywords,
        )

        scores: List[Tuple[str, float, bool]] = []
        for ticker in all_tickers:
            if ticker not in company_emb:
                continue
            s = compute_pair_score(
                q["q_fin"],
                q["q_sbert"],
                list(q["query_keywords"]),
                mention_levels,
                ticker,
                company_emb,
                company_keywords,
                ticker_to_group,
                group_keywords,
                params["alpha1"],
                params["alpha2"],
                params["beta1"],
                params["beta2"],
                params["lambda1"],
                params["lambda2"],
                params["lambda3"],
            )
            level = mention_levels.get(ticker, MENTION_LEVEL_NONE)
            mention_flag = level > MENTION_LEVEL_NONE
            scores.append((ticker, s, mention_flag))

        if not scores:
            continue

        # mention이 있는 기업 먼저, 그 다음 점수 순
        scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
        top = [t for t, _, _ in scores[:top_k]]

        if true_tickers & set(top):
            hits += 1
        total += 1

    if total == 0:
        return 0.0
    return hits / total



def infer_for_text_file(
    text_path: Path,
    params: Dict[str, float],
    top_k: int,
    finbert_model: SentenceTransformer,
    sbert_model: SentenceTransformer,
    kw_model: KeyBERT,
    df_sp500: pd.DataFrame,
    company_emb: Dict[str, Dict[str, np.ndarray]],
    company_keywords: Dict[str, set[str]],
    ticker_to_group: Dict[str, str],
    group_keywords: Dict[str, set[str]],
    ticker_to_name_keywords: Dict[str, Set[str]],
    ticker_to_name: Dict[str, str],
) -> None:
    text = text_path.read_text(encoding="utf-8")
    q = encode_query(text, finbert_model, sbert_model, kw_model)

    mention_levels = compute_mention_levels(
        text,
        ticker_to_name_keywords,
        company_keywords,
    )

    all_tickers = sorted(df_sp500["ticker"].unique())
    scores: List[Tuple[str, float, bool]] = []

    for ticker in tqdm(all_tickers, desc="Scoring tickers for text"):
        if ticker not in company_emb:
            continue

        s = compute_pair_score(
            q["q_fin"],
            q["q_sbert"],
            list(q["query_keywords"]),
            mention_levels,
            ticker,
            company_emb,
            company_keywords,
            ticker_to_group,
            group_keywords,
            params["alpha1"],
            params["alpha2"],
            params["beta1"],
            params["beta2"],
            params["lambda1"],
            params["lambda2"],
            params["lambda3"],
        )

        level = mention_levels.get(ticker, MENTION_LEVEL_NONE)
        mention_flag = level > MENTION_LEVEL_NONE

        # 레벨별 보너스 (티커 > alias > 키워드)
        if level > MENTION_LEVEL_NONE:
            bonus = BASE_MENTION_BONUS * float(level)
            s += bonus

        scores.append((ticker, s, mention_flag))



    # 1) baseline: score만 기준
    baseline_top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    # 2) booster: (mention_flag, score) 기준
    boosted_top = sorted(scores, key=lambda x: (x[2], x[1]), reverse=True)[:top_k]

    print("=" * 80)
    print("[INPUT TEXT PREVIEW]")
    print(q["text"][:500] + ("..." if len(q["text"]) > 500 else ""))
    print()
    print("[MENTIONED TICKERS DETECTED]")
    print(sorted(mentioned_tickers))
    print()

    print(f"[BASELINE TOP-{top_k} (no hard booster)]")
    for t, s, m in baseline_top:
        mark_m = "*" if m else " "
        print(f" {mark_m} {t:5s}  {s:.4f}")
    print()

    print(f"[BOOSTED TOP-{top_k} (mention first)]")
    for t, s, m in boosted_top:
        mark_m = "*" if m else " "
        print(f" {mark_m} {t:5s}  {s:.4f}")

    print()



# -----------------------------
# main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "text"],
        default="train",
        help="train: test split 위 질적/정량 평가, text: 임의 기사 텍스트 inference",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="mode=train일 때 사용할 test 샘플 개수",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="mode=text일 때 사용할 기사 텍스트 파일 경로",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="출력할 top-k 기업 수 (미지정 시 model.pkl의 top_k 사용)",
    )
    args = parser.parse_args()

    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"{MODEL_PKL} not found. 먼저 model.py를 돌려서 pkl을 생성하세요.")

    with MODEL_PKL.open("rb") as f:
        cfg = pickle.load(f)

    params = {
        "alpha1": cfg["alpha1"],
        "alpha2": cfg["alpha2"],
        "beta1": cfg["beta1"],
        "beta2": cfg["beta2"],
        "lambda1": cfg["lambda1"],
        "lambda2": cfg["lambda2"],
        "lambda3": cfg["lambda3"],
    }

    top_k = args.top_k if args.top_k is not None else cfg.get("top_k", 5)

    print("[INFO] Loaded model config from", MODEL_PKL)
    print("[INFO] Params:", params)
    print("[INFO] top_k:", top_k)

    df_sp500 = load_sp500()
    name_col = get_name_column(df_sp500)
    ticker_to_name = dict(zip(df_sp500["ticker"], df_sp500[name_col]))
    ticker_to_name_keywords = build_name_keywords(df_sp500)

    training_data = load_training_data()
    test_data = [r for r in training_data if r.get("split") == "test" and r.get("sp500_labels")]

    print(f"[INFO] Total cleaned samples: {len(training_data)}")
    print(f"[INFO] TEST samples (with sp500_labels): {len(test_data)}")

    company_keywords = load_company_keywords()
    company_keywords = filter_company_keywords(company_keywords)
    group_keywords = load_industry_group_keywords()
    ticker_to_group = build_ticker_to_group(df_sp500)

    static_client = chromadb.PersistentClient(path=str(STATIC_CHROMA_PATH))
    dynamic_client = chromadb.PersistentClient(path=str(DYNAMIC_CHROMA_PATH))
    company_emb = load_company_embeddings(static_client, dynamic_client)

    finbert_model = SentenceTransformer("yiyanghkust/finbert-tone")
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sbert_model)

    if args.mode == "train":
        if not test_data:
            print("[WARN] No TEST samples with S&P500 labels. Nothing to evaluate.")
            return

        qualitative_eval_on_test(
            n_samples=args.n_samples,
            params=params,
            top_k=top_k,
            finbert_model=finbert_model,
            sbert_model=sbert_model,
            kw_model=kw_model,
            df_sp500=df_sp500,
            test_data=test_data,
            company_emb=company_emb,
            company_keywords=company_keywords,
            ticker_to_group=ticker_to_group,
            group_keywords=group_keywords,
            ticker_to_name_keywords=ticker_to_name_keywords,
            ticker_to_name=ticker_to_name,
        )

        hit = compute_hit_at_k_on_test(
            test_data=test_data,
            params=params,
            top_k=top_k,
            finbert_model=finbert_model,
            sbert_model=sbert_model,
            kw_model=kw_model,
            df_sp500=df_sp500,
            company_emb=company_emb,
            company_keywords=company_keywords,
            ticker_to_group=ticker_to_group,
            group_keywords=group_keywords,
            ticker_to_name_keywords=ticker_to_name_keywords,
        )

        print(f"[RESULT] Hit@{top_k} on TEST (S&P500-labeled samples): {hit:.4f}")

    elif args.mode == "text":
        if not args.text_file:
            raise ValueError("--text-file 을 지정해야 합니다.")
        infer_for_text_file(
            text_path=Path(args.text_file),
            params=params,
            top_k=top_k,
            finbert_model=finbert_model,
            sbert_model=sbert_model,
            kw_model=kw_model,
            df_sp500=df_sp500,
            company_emb=company_emb,
            company_keywords=company_keywords,
            ticker_to_group=ticker_to_group,
            group_keywords=group_keywords,
            ticker_to_name_keywords=ticker_to_name_keywords,  # ★ 추가
            ticker_to_name=ticker_to_name,
        )



if __name__ == "__main__":
    main()