"""Pipeline for SEC and wiki summary creation, embedding, and keyword extraction.

The script is separated into two stages:
1) 요약본 저장: Generate static/dynamic summaries from SEC and wiki sources.
2) 임베딩 및 keyword 추출: Embed the summaries with FinBERT and SBERT, and
   extract keywords with KeyBERT.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import chromadb
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
SEC_DIR = BASE_DIR / "json" / "sec_json"
WIKI_DIR = BASE_DIR / "json" / "wiki_json"
SUMMARY_DIR = BASE_DIR / "json" / "summary_json"
STATIC_VEC_DIR = BASE_DIR / "chromaDB" / "static"
DYNAMIC_VEC_DIR = BASE_DIR / "chromaDB" / "dynamic"
METADATA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+\.json$")


@dataclass
class SummaryArtifacts:
    ticker: str
    company_name: str
    static_source_text: str
    dynamic_source_text: str
    static_summary_path: Path
    dynamic_summary_path: Path
    keyword_path: Path


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_sentences(text: str) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return re.split(r"(?<=[.!?])\s+", cleaned)


def simple_summarize(text: str, max_sentences: int = 5, max_chars: int = 1200) -> str:
    sentences = split_sentences(text)
    summary = " ".join(sentences[:max_sentences])
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def extract_text(data: Dict, key: str) -> str:
    value = data.get(key, "")
    if isinstance(value, list):
        return " ".join([v for v in value if isinstance(v, str)])
    if isinstance(value, str):
        return value
    return ""


def build_summary_payload(ticker: str, company_name: str, summary: str) -> Dict:
    return {"ticker": ticker, "company_name": company_name, "summary": summary}


@dataclass
class TickerContext:
    ticker: str
    company_name: str
    wiki_text: str


def build_ticker_context(wiki_metadata_path: Path) -> TickerContext | None:
    if not wiki_metadata_path.exists():
        return None

    wiki_data = load_json(wiki_metadata_path)

    ticker_field = wiki_data.get("tickers")
    resolved_ticker = ""
    if isinstance(ticker_field, list) and ticker_field:
        resolved_ticker = str(ticker_field[0]).upper()
    elif isinstance(ticker_field, str):
        resolved_ticker = ticker_field.upper()
    else:
        resolved_ticker = str(wiki_data.get("ticker", "")).upper()

    if not resolved_ticker:
        print(f"[WARN] metadata file missing ticker(s): {wiki_metadata_path}")
        return None

    company_name = (
        wiki_data.get("wiki_title_used")
        or wiki_data.get("company_name")
        or wiki_data.get("corrected_name")
        or resolved_ticker
    )
    wiki_text = extract_text(wiki_data, "clean_text")

    return TickerContext(
        ticker=resolved_ticker,
        company_name=company_name,
        wiki_text=wiki_text,
    )


def generate_summaries(context: TickerContext) -> SummaryArtifacts:
    ticker_upper = context.ticker.upper()

    ten_k_path = SEC_DIR / f"{ticker_upper}_latest_10K_sections.json"
    ten_q_path = SEC_DIR / f"{ticker_upper}_latest_10Q_sections.json"

    item1_text = ""
    item7_text = ""
    item2_text = ""

    if ten_k_path.exists():
        ten_k_data = load_json(ten_k_path)
        item1_text = extract_text(ten_k_data, "item1")
        item7_text = extract_text(ten_k_data, "item7")
    else:
        print(f"[WARN] Missing 10-K sections for {ticker_upper}: {ten_k_path}")

    if ten_q_path.exists():
        ten_q_data = load_json(ten_q_path)
        item2_text = extract_text(ten_q_data, "item2")
    else:
        print(f"[WARN] Missing 10-Q sections for {ticker_upper}: {ten_q_path}")

    static_parts = []
    if item1_text.strip():
        static_parts.append(("item1", item1_text))
    if context.wiki_text.strip():
        static_parts.append(("wiki", context.wiki_text))
    else:
        print(f"[WARN] Missing wiki clean_text for {ticker_upper}")

    dynamic_parts = []
    if item2_text.strip():
        dynamic_parts.append(("item2", item2_text))
    if item7_text.strip():
        dynamic_parts.append(("item7", item7_text))

    static_source = "\n".join([text for _, text in static_parts])
    dynamic_source = "\n".join([text for _, text in dynamic_parts])

    static_summary = simple_summarize(static_source)
    dynamic_summary = simple_summarize(dynamic_source)

    static_path = SUMMARY_DIR / f"{ticker_upper}_static.json"
    dynamic_path = SUMMARY_DIR / f"{ticker_upper}_dynamic.json"

    save_json(
        static_path, build_summary_payload(ticker_upper, context.company_name, static_summary)
    )
    save_json(
        dynamic_path,
        build_summary_payload(ticker_upper, context.company_name, dynamic_summary),
    )

    print(f"[INFO] Saved static summary: {static_path}")
    print(f"[INFO] Saved dynamic summary: {dynamic_path}")

    keyword_path = SUMMARY_DIR / f"{ticker_upper}_keyword.json"

    return SummaryArtifacts(
        ticker=ticker_upper,
        company_name=context.company_name,
        static_source_text=static_source,
        dynamic_source_text=dynamic_source,
        static_summary_path=static_path,
        dynamic_summary_path=dynamic_path,
        keyword_path=keyword_path,
    )


# -------------------- 임베딩 및 keyword 추출 -------------------- #

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / (
        input_mask_expanded.sum(1) + 1e-9
    )


def embed_with_finbert(text: str, tokenizer, model, device: str) -> List[float]:
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
    return pooled.detach().cpu().numpy()[0].tolist()


def embed_with_sbert(model: SentenceTransformer, text: str) -> List[float]:
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


def load_summary_text(path: Path) -> str:
    if not path.exists():
        return ""
    data = load_json(path)
    return data.get("summary", "")


def extract_keywords(kw_model: KeyBERT, text: str, top_n: int = 10) -> List[str]:
    if not text.strip():
        return []
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words="english")
    return [kw for kw, _ in keywords]


def get_chroma_collections():
    static_client = chromadb.PersistentClient(path=str(STATIC_VEC_DIR))
    dynamic_client = chromadb.PersistentClient(path=str(DYNAMIC_VEC_DIR))

    static_sbert = static_client.get_or_create_collection("sbert")
    static_finbert = static_client.get_or_create_collection("finbert")
    dynamic_sbert = dynamic_client.get_or_create_collection("sbert")
    dynamic_finbert = dynamic_client.get_or_create_collection("finbert")

    return {
        "static_sbert": static_sbert,
        "static_finbert": static_finbert,
        "dynamic_sbert": dynamic_sbert,
        "dynamic_finbert": dynamic_finbert,
    }


def process_embeddings_and_keywords(artifacts: SummaryArtifacts) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finbert_name = "yiyanghkust/finbert-tone"
    sbert_name = "sentence-transformers/all-MiniLM-L6-v2"

    finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_name)
    finbert_model = AutoModel.from_pretrained(finbert_name).to(device)
    finbert_model.eval()

    sbert_model = SentenceTransformer(sbert_name, device=device)
    kw_model = KeyBERT(model=sbert_model)

    static_summary = load_summary_text(artifacts.static_summary_path)
    dynamic_summary = load_summary_text(artifacts.dynamic_summary_path)

    # Embed using the combined source texts so that static captures item1 + wiki
    # and dynamic captures item2 + item7 content.
    static_embedding_text = artifacts.static_source_text or static_summary
    dynamic_embedding_text = artifacts.dynamic_source_text or dynamic_summary

    embeddings = {
        "static_finbert": embed_with_finbert(
            static_embedding_text, finbert_tokenizer, finbert_model, device
        ),
        "dynamic_finbert": embed_with_finbert(
            dynamic_embedding_text, finbert_tokenizer, finbert_model, device
        ),
        "static_sbert": embed_with_sbert(sbert_model, static_embedding_text),
        "dynamic_sbert": embed_with_sbert(sbert_model, dynamic_embedding_text),
    }

    chroma_collections = get_chroma_collections()

    chroma_collections["static_finbert"].upsert(
        ids=[artifacts.ticker],
        embeddings=[embeddings["static_finbert"]],
        metadatas=[{"ticker": artifacts.ticker, "company_name": artifacts.company_name}],
    )
    chroma_collections["dynamic_finbert"].upsert(
        ids=[artifacts.ticker],
        embeddings=[embeddings["dynamic_finbert"]],
        metadatas=[{"ticker": artifacts.ticker, "company_name": artifacts.company_name}],
    )
    chroma_collections["static_sbert"].upsert(
        ids=[artifacts.ticker],
        embeddings=[embeddings["static_sbert"]],
        metadatas=[{"ticker": artifacts.ticker, "company_name": artifacts.company_name}],
    )
    chroma_collections["dynamic_sbert"].upsert(
        ids=[artifacts.ticker],
        embeddings=[embeddings["dynamic_sbert"]],
        metadatas=[{"ticker": artifacts.ticker, "company_name": artifacts.company_name}],
    )

    print(
        f"[INFO] Stored embeddings in ChromaDB (static: {STATIC_VEC_DIR}, dynamic: {DYNAMIC_VEC_DIR})"
    )

    keyword_payload = {
        "ticker": artifacts.ticker,
        "static_keywords": extract_keywords(kw_model, static_summary),
        "dynamic_keywords": extract_keywords(kw_model, dynamic_summary),
    }

    save_json(artifacts.keyword_path, keyword_payload)
    print(f"[INFO] Saved keywords: {artifacts.keyword_path}")


def main():
    metadata_files = [
        p
        for p in sorted(WIKI_DIR.glob("*.json"))
        if METADATA_NAME_PATTERN.match(p.name)
    ]

    if not metadata_files:
        print(f"[WARN] No metadata files found in {WIKI_DIR}")
        return

    for metadata_file in tqdm(metadata_files, desc="Processing tickers"):
        context = build_ticker_context(metadata_file)
        if context is None:
            continue
        artifacts = generate_summaries(context)
        process_embeddings_and_keywords(artifacts)


if __name__ == "__main__":
    main()