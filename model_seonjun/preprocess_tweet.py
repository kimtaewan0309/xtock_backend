import pandas as pd
import re
import json
from pathlib import Path

# =========================================================
# 1) 설정
# =========================================================

# 광고/스팸 키워드
AD_KEYWORDS = [
    "join now", "free", "subscribe", "premium", "alert",
    "breaking picks", "100% gain", "dm us", "signals",
    "our picks", "bonus", "get rich", "indicator", "💰"
]

URL_REGEX = r"http[s]?://\S+"
TICKER_REGEX = r"\$[A-Za-z]{1,5}\b"

MIN_LEN = 15  # body 길이 기준


# =========================================================
# 2) 헬퍼 함수: 전처리 필터
# =========================================================

def is_spam_or_ad(text: str) -> bool:
    """광고성 문구 포함 여부"""
    t = text.lower()
    return any(k in t for k in AD_KEYWORDS)


def ticker_only_or_list(text: str) -> bool:
    """티커만 나열한 문장 여부"""
    tokens = text.strip().split()
    if len(tokens) <= 6:  # 지나친 제거 방지용
        tickers = re.findall(TICKER_REGEX, text)
        non_tickers = [tok for tok in tokens if not tok.startswith("$")]

        # 티커 비율이 과도하게 높은 경우 제거
        if len(tokens) > 0 and len(tickers) / len(tokens) >= 0.7:
            return True

        # 텍스트 전체가 ticker-only면 제거
        if len(non_tickers) == 0:
            return True

    return False


def too_much_url(text: str) -> bool:
    """문장 내 URL 비율이 과다한 경우"""
    urls = re.findall(URL_REGEX, text)
    if not urls:
        return False

    url_len = sum(len(u) for u in urls)
    if url_len / max(len(text), 1) > 0.5:  # 50% 이상 URL
        return True

    return False


def clean_text(text: str) -> str:
    """기본 텍스트 정리"""
    return text.replace("\n", " ").strip()


# =========================================================
# 3) Tweet.csv + Company_Tweet.csv 전처리
# =========================================================

def preprocess_joined_tweets(tweet_csv, company_csv, sp500_tickers):
    print("[INFO] Loading Tweet.csv and Company_Tweet.csv...")

    df_tweet = pd.read_csv(tweet_csv)
    df_company = pd.read_csv(company_csv)

    # inner join (공통 tweet_id만 유지)
    df_merge = df_company.merge(df_tweet, on="tweet_id", how="inner")
    df_merge = df_merge.rename(columns={"body": "text", "ticker_symbol": "ticker"})
    df_merge["text"] = df_merge["text"].astype(str)

    print("[INFO] Joined rows:", len(df_merge))

    # groupby → tweet_id 기준 multi-label 구성
    grouped = df_merge.groupby("tweet_id").agg({
        "text": "first",
        "ticker": list
    }).reset_index()

    print("[INFO] Grouped unique tweets:", len(grouped))

    cleaned_rows = []

    for _, row in grouped.iterrows():
        text = clean_text(row["text"])
        tickers = [t for t in row["ticker"] if t in sp500_tickers]

        # S&P500 티커 없는 row 제거
        if not tickers:
            continue

        # 삭제 조건 적용
        if len(text) < MIN_LEN:
            continue
        if is_spam_or_ad(text):
            continue
        if ticker_only_or_list(text):
            continue
        if too_much_url(text):
            continue

        cleaned_rows.append({
            "doc_id": f"tweet_{row['tweet_id']}",
            "source": "twitter_join",
            "description": text,
            "sp500_labels": list(set(tickers))  # 중복 제거
        })

    print("[INFO] Cleaned join tweets:", len(cleaned_rows))
    return cleaned_rows


# =========================================================
# 4) stock_tweets.csv 전처리
# =========================================================

def preprocess_stock_tweets(stock_csv, sp500_tickers):
    print("[INFO] Loading stock_tweets.csv...")

    df = pd.read_csv(stock_csv)
    df = df.rename(columns={"Tweet": "text", "Stock Name": "ticker"})
    df["text"] = df["text"].astype(str)

    cleaned_rows = []

    for idx, row in df.iterrows():
        text = clean_text(row["text"])
        ticker = row["ticker"]

        if ticker not in sp500_tickers:
            continue

        # 삭제 조건
        if len(text) < MIN_LEN:
            continue
        if is_spam_or_ad(text):
            continue
        if ticker_only_or_list(text):
            continue
        if too_much_url(text):
            continue

        cleaned_rows.append({
            "doc_id": f"stock_{idx}",
            "source": "stock_tweet",
            "description": text,
            "sp500_labels": [ticker]
        })

    print("[INFO] Cleaned stock tweets:", len(cleaned_rows))
    return cleaned_rows


# =========================================================
# 5) 실행 및 저장
# =========================================================

if __name__ == "__main__":
    print("====== Tweet Preprocessing START ======")

    BASE_DIR = Path(__file__).resolve().parent
    # S&P500 티커 목록 로드
    sp500_path = BASE_DIR / "csv" / "sp500_list.csv"
    if sp500_path.exists():
        df_sp = pd.read_csv(sp500_path)
        sp500_tickers = set(df_sp["ticker"].tolist())
    else:
        print("[WARN] sp500.csv가 없어 모든 ticker 허용으로 진행합니다.")
        sp500_tickers = None

    # 경로 설정
    tweet_csv =  BASE_DIR / "csv" / "Tweet.csv"
    company_csv = BASE_DIR / "csv" / "Company_Tweet.csv"
    stock_csv = BASE_DIR / "csv" / "stock_tweets.csv"

    joined_clean = preprocess_joined_tweets(tweet_csv, company_csv, sp500_tickers)
    stock_clean = preprocess_stock_tweets(stock_csv, sp500_tickers)

    # 저장
    out1 = "preprocessed_joined_tweets.json"
    out2 = "preprocessed_stock_tweets.json"

    with open(out1, "w") as f:
        json.dump(joined_clean, f, indent=2)

    with open(out2, "w") as f:
        json.dump(stock_clean, f, indent=2)

    print("[INFO] Saved:", out1, out2)
    print("====== Preprocessing DONE ======")
