"""
Script 1: Build industry_group-level keyword lists.

입력:
  - csv/sp500_list.csv
      columns: ["ticker", "industry_group", ...]
  - json/keyword/{ticker}_keyword.json
      {
        "ticker": "AAPL",
        "static_keywords": [...],
        "dynamic_keywords": [...]
      }

출력:
  - json/industry_group/{industry_group}.json
      {
        "industry_group": "<group name>",
        "tickers": [ "AAPL", "MSFT", ... ],
        "keywords": [ "smartphone", "hardware", ... ]  # 중복 제거된 set
      }
"""

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SP500_CSV = BASE_DIR / "csv" / "sp500_list.csv"
KEYWORD_DIR = BASE_DIR / "json" / "keyword"
OUT_DIR = BASE_DIR / "json" / "industry_group"


def load_company_keywords(ticker: str) -> set[str]:
    """
    json/keyword/{ticker}_keyword.json 을 읽어서
    static_keywords + dynamic_keywords 를 합친 set을 반환.
    파일이 없으면 빈 set.
    """
    path = KEYWORD_DIR / f"{ticker}_keyword.json"
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 필드명은 설계대로: static_keywords, dynamic_keywords
    static_kws = data.get("static_keywords", []) or data.get("static", [])
    dynamic_kws = data.get("dynamic_keywords", []) or data.get("dynamic", [])
    return set(static_kws) | set(dynamic_kws)


def sanitize_filename(name: str) -> str:
    """
    industry_group 이름이 파일명으로 곤란한 문자(/, 공백 등)를 포함할 수 있으니
    최소한 / 와 공백만 치환.
    """
    return name.replace("/", "_").replace(" ", "_")


def main() -> None:
    df = pd.read_csv(SP500_CSV)

    # industry_group별로 그룹화
    grouped = df.groupby("industry_group")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for industry_group, group_df in grouped:
        tickers = sorted(group_df["ticker"].unique())

        keyword_set: set[str] = set()
        for ticker in tickers:
            kws = load_company_keywords(ticker)
            keyword_set |= kws

        # industry_group에 속한 기업이 아무 키워드도 없을 수도 있으니 그대로 저장
        payload = {
            "industry_group": industry_group,
            "tickers": tickers,
            "keywords": sorted(keyword_set),
        }

        filename = sanitize_filename(industry_group) + ".json"
        out_path = OUT_DIR / filename

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved keywords for industry_group={industry_group} → {out_path}")


if __name__ == "__main__":
    main()
