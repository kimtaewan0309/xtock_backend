# main.py
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv


#yfinance
import datetime as dt

import pandas as pd
import yfinance as yf

from pydantic import BaseModel

#Mongo DB
from pymongo import MongoClient


from chart.router import router as chart_router

# .env 파일에서 환경변수 불러오기
load_dotenv()

# .env에 넣어둔 Bearer Token 읽어오기
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")

#if not BEARER_TOKEN:
#    raise RuntimeError("X_BEARER_TOKEN 환경변수가 설정되어 있지 않습니다.")

# X API v2 최근 트윗 검색 엔드포인트
BASE_URL = "https://api.x.com/2/tweets/search/recent"
# 만약 이 주소가 안 되면 아래 주석 풀고 위는 주석 처리해서 테스트:
# BASE_URL = "https://api.twitter.com/2/tweets/search/recent"


MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "xtock")
MONGODB_COLLECTION_TWEET_IMPACT = os.getenv("MONGODB_COLLECTION_TWEET_IMPACT", "tweet_impact")

mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = mongo_client[MONGODB_DB_NAME] if mongo_client is not None else None
tweet_impact_col = db[MONGODB_COLLECTION_TWEET_IMPACT] if db is not None else None

app = FastAPI(
    title="Xtock Xignal Backend",
    description="X API 연동 테스트용 백엔드",
)

# 시뮬레이션 파일
from simulation.router import router as simulation_router
app.include_router(simulation_router)

# 차트 파일
from chart.router import router as chart_router
app.include_router(chart_router)


async def call_x_recent_search(
    query: str,
    max_results: int = 10,
    next_token: Optional[str] = None,
):
    """
    실제로 X API를 호출하는 함수
    """
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
    }

    params = {
        "query": query,
        "max_results": max_results,
        # 필요한 필드는 여기서 추가 가능
        "tweet.fields": "created_at,author_id,public_metrics,lang",
    }
    if next_token:
        params["next_token"] = next_token  # 페이지네이션 용

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(BASE_URL, headers=headers, params=params)

    # 200이 아니면 에러로 보내기
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail={"msg": "X API error", "body": resp.json()},
        )

    return resp.json()


class TweetImpactRequest(BaseModel):
    symbol: str                  # 티커 예: "TSLA"
    tweet_created_at: str        # 예: "2025-11-14T15:30:00.000Z"
    tweet_id: Optional[str] = None
    tweet_text: Optional[str] = None


def infer_base_date_from_tweet_created_at(created_at: str) -> dt.date:
    """
    트윗의 created_at(ISO8601 문자열)을 받아서
    기준 날짜(date)만 뽑아낸다.

    예:
    "2025-11-25T10:06:51.000Z" -> date(2025, 11, 25)
    """
    s = created_at.strip()

    # X API는 보통 끝에 'Z' (UTC)를 붙이니까, Python이 이해할 수 있게 바꿔줌
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"  # Z -> +00:00

    # 만약 타임존이 아예 없으면, UTC로 간주
    if len(s) == 19:  # "YYYY-MM-DDTHH:MM:SS"
        s += "+00:00"

    dt_obj = dt.datetime.fromisoformat(s)
    return dt_obj.date()



#익일 수익률 계산 함수
def calculate_next_day_return(symbol: str, date_str: str):
    """
    특정 날짜(date_str)의 '다음날 수익률'을 계산.
    수식: (다음날 종가 - 해당일 종가) / 해당일 종가 * 100

    date_str: 'YYYY-MM-DD'
    - 기준일이 장 휴일(주말 등)이면, 기준일 이전의 가장 가까운 거래일을 기준으로 삼는다.
    """
    base_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

    # 기준일 앞뒤로 넉넉하게 1주일 정도 가져오기
    start = base_date - dt.timedelta(days=7)
    end = base_date + dt.timedelta(days=7)

    #기준일 데이터 가져오기(1주일 분량)
    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="column",
        auto_adjust=False,
        progress=False,
    )

    if df.empty or len(df) < 2:
        return None  # 계산 불가

    df = df.reset_index()

    # 첫 번째 컬럼을 날짜로 사용 (보통 'Date')
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    # 날짜 기준 정렬
    df = df.sort_values(date_col).reset_index(drop=True)

    # 기준일(base_date) 이전/같은 거래일들 중 마지막 하나 찾기
    candidates = df[df[date_col] <= base_date]
    if candidates.empty:
        # 기준일보다 이전 거래일도 없으면 계산 불가
        return None

    base_idx = candidates.index[-1]  # 기준이 되는 거래일 인덱스
    next_idx = base_idx + 1

    if next_idx >= len(df):
        # 다음 거래일이 없으면 계산 불가
        return None

    base_row = df.loc[base_idx]
    next_row = df.loc[next_idx]

    base_close = float(base_row["Close"])
    next_close = float(next_row["Close"])

    next_return = (next_close - base_close) / base_close * 100.0

    base_date_out = base_row[date_col]
    next_date_out = next_row[date_col]


    # date 객체를 문자열로 (YYYY-MM-DD)
    if hasattr(base_date_out, "isoformat"):
        base_date_str = base_date_out.isoformat()
    else:
        base_date_str = str(base_date_out)[:10]

    if hasattr(next_date_out, "isoformat"):
        next_date_str = next_date_out.isoformat()
    else:
        next_date_str = str(next_date_out)[:10]



    return {
        "symbol": symbol,
        "base_date": base_date_str,
        "base_close": base_close,
        "next_date": next_date_str,
        "next_close": next_close,
        "next_day_return": next_return,
    }



#주가 조회 함수
def fetch_price_history(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
):
    """
    yfinance로 특정 종목의 가격 데이터를 가져오는 함수.
    start, end는 'YYYY-MM-DD' 형식.
    """
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        group_by="column",   # 멀티인덱스 방지
        auto_adjust=False,   # 경고 제거용 명시
        progress=False,
    )

    if df.empty:
        return []

    # 인덱스를 컬럼으로 리셋
    df = df.reset_index()

    # 첫 번째 컬럼을 '날짜'라고 생각하고 사용
    date_col = df.columns[0]   # 보통 'Date' 혹은 비슷한 이름
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    records = []
    for _, row in df.iterrows():
        date_value = row[date_col]

        # date_value 타입에 상관없이 문자열로 안전하게 변환
        if hasattr(date_value, "strftime"):
            date_str = date_value.strftime("%Y-%m-%d")
        else:
            date_str = str(date_value)[:10]

        records.append(
            {
                "date": date_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )

    return records


def save_tweet_impact(doc: dict):
    """
    tweet_impact 컬렉션에 upsert.
    tweet_id 기준으로 한 번만 저장되도록 함.
    MongoDB가 설정되지 않았으면 그냥 패스.
    """
    if tweet_impact_col is None:
        return

    key = {}
    if doc.get("tweet_id"):
        key["tweet_id"] = doc["tweet_id"]
    else:
        # tweet_id 없으면 symbol + base_date 조합으로라도 키 만들기
        key = {"symbol": doc.get("symbol"), "base_date": doc.get("base_date")}

    tweet_impact_col.update_one(
        key,
        {"$set": doc},
        upsert=True,
    )


@app.get("/api/tweets")
async def get_tweets(
    q: str = Query(..., description="검색 쿼리 (예: $TSLA, from:elonmusk 등)"),
    max_results: int = Query(10, ge=10, le=100),
    next_token: Optional[str] = Query(None, description="페이지네이션용 next_token"),
):
    """
    프론트엔드에서 사용할 엔드포인트

    예시:
    - /api/tweets?q=$TSLA
    - /api/tweets?q=from:elonmusk
    - /api/tweets?q="NVIDIA stock"
    """
    data = await call_x_recent_search(q, max_results=max_results, next_token=next_token)

    # 나중에 여기서 필요한 필드만 뽑아서 리턴해도 됨
    return {
        "query": q,
        "max_results": max_results,
        "raw": data,
    }

@app.get("/api/price")
def get_price_history(
    symbol: str,
    start: str,
    end: str,
):
    """
    기간별 주가 조회 API.
    예시:
    /api/price?symbol=TSLA&start=2025-11-01&end=2025-11-20
    """
    data = fetch_price_history(symbol, start, end)
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "prices": data,
    }

@app.get("/api/next-return")
def get_next_day_return(
    symbol: str,
    date: str,
):
    """
    특정 날짜 기준 익일 수익률 계산 API.
    예시:
    /api/next-return?symbol=TSLA&date=2025-11-25
    """
    result = calculate_next_day_return(symbol, date)
    if result is None:
        raise HTTPException(status_code=404, detail="Not enough data to compute return")
    return result


@app.post("/api/tweet-impact")
def tweet_impact(payload: TweetImpactRequest):
    """
    트윗 1개와 종목(symbol)을 받아서
    해당 트윗 기준 '다음 거래일 수익률'까지 같이 반환하는 API.

    요청 예시(JSON):

    {
      "symbol": "TSLA",
      "tweet_created_at": "2025-11-14T15:30:00.000Z",
      "tweet_id": "1993260082088296958",
      "tweet_text": "Tesla to the moon"
    }
    """
    # 1) 트윗 생성 시각 → 기준 날짜 추출
    try:
        base_date = infer_base_date_from_tweet_created_at(payload.tweet_created_at)
        base_date_str = base_date.strftime("%Y-%m-%d")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tweet_created_at format: {str(e)}",
        )

    # 2) 기준 날짜 기준 다음 거래일 수익률 계산
    result = calculate_next_day_return(payload.symbol, base_date_str)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Not enough data to compute return for this tweet",
        )

    # 3) 트윗 정보 + 수익률 정보를 합쳐서 반환
    doc = {
        "symbol": payload.symbol,
        "tweet_id": payload.tweet_id,
        "tweet_text": payload.tweet_text,
        "tweet_created_at": payload.tweet_created_at,
        "base_date": result["base_date"],
        "base_close": result["base_close"],
        "next_date": result["next_date"],
        "next_close": result["next_close"],
        "next_day_return": result["next_day_return"],
        # 나중에 SBERT/FinBERT 붙일 자리
        "sentiment": None,
        "matches": [],
    }

    # 4) MongoDB에 저장
    save_tweet_impact(doc)

    return doc

# 기업 매칭 Placeholder
@app.post("/api/match-company")
def match_company(payload: dict):
    """
    Placeholder: 나중에 SBERT 임베딩 모델 연결할 자리
    현재는 dummy로 반환
    """
    text = payload.get("text", "")

    # TODO: 여기에 SBERT 로직 붙이기
    # 예: embedding → cosine similarity → top companies

    return {
        "input_text": text,
        "matches": [
            {"symbol": "TSLA", "score": 0.8}, 
            {"symbol": "NVDA", "score": 0.4}
        ],
        "note": "SBERT model not implemented yet"
    }


#감성 분석 Placeholder
@app.post("/api/sentiment")
def analyze_sentiment(payload: dict):
    """
    Placeholder: 나중에 FinBERT 감성 분석 모델 연결할 자리
    현재는 dummy로 반환
    """
    text = payload.get("text", "")

    # TODO: 여기에 FinBERT 로직 붙이기
    # 예: model(text) → logits → softmax → label

    return {
        "input_text": text,
        "sentiment": "Neutral",
        "confidence": 0.5,
        "note": "FinBERT model not implemented yet"
    }
