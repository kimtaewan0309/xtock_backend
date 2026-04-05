import pandas as pd
import wikipedia
import yfinance as yf
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import os
import time
import nltk
from nltk.corpus import stopwords
import json

# ---------------------------------------------------------
# [설정] 기존 pipeline_final.py의 설정 및 모델 로드
# ---------------------------------------------------------
# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_OUTPUT_FILE = os.path.join(BASE_DIR, "sp500_sbert_input.jsonl")

wikipedia.set_lang("en")

# KeyBERT 및 SBERT 모델 로드 (KeyBERT 모델도 BGE-M3로 설정)
print("Loading KeyBERT Model (BGE-M3)...")
# 만약 GPU를 사용한다면 device='cuda'를 추가하세요.
sentence_model = SentenceTransformer('BAAI/bge-m3') 
kw_model = KeyBERT(model=sentence_model)

# 불용어 설정
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')
custom_stops = ['inc', 'corp', 'company', 'ltd', 'incorporated', 'group', 'class', 'plc', 'holding', 'holdings']
stop_words.extend(custom_stops)

# [핵심] 수동 매핑 (Alphabet Inc. 오류 수정)
MANUAL_MAPPING = {
    "GOOG": "Alphabet Inc.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon (company)", 
    "V": "Visa Inc.",
    "O": "Realty Income",
    "T": "AT&T",
    "MA": "Mastercard",
    "BA": "Boeing",
    "CAT": "Caterpillar Inc.",
    "ALL": "Allstate",
    "DD": "DuPont",
    "M": "Macy's",
    "F": "Ford Motor Company",
    "PODD": "Insulet", 
    "BF-B": "Brown-Forman", 
    "BRK-B": "Berkshire Hathaway", 
    "SOLV": "Solventum", 
    "KVUE": "Kenvue"
}

# 수동 키워드 설정 (추가 태그)
MANUAL_KEYWORDS = {
    "GOOG": "Gemini, Bard, DeepMind, LLM, Generative AI, Search, YouTube",
    "GOOGL": "Gemini, Bard, DeepMind, LLM, Generative AI, Search, YouTube",
    # (다른 기업 키워드도 여기에 있어야 함)
}

# ---------------------------------------------------------
# [함수] 데이터 처리 함수 (pipeline_final.py와 동일)
# ---------------------------------------------------------

def clean_company_name(name):
    """괄호 및 불필요한 이름 제거"""
    name = re.sub(r'\s*\(.*?\)', '', name)
    name = re.sub(r'\.com', '', name)
    return name.strip()

def smart_search(name, ticker):
    """위키피디아에서 정보 가져오기"""
    try:
        query = MANUAL_MAPPING.get(ticker, clean_company_name(name))
        res = wikipedia.search(query)
        if not res: res = wikipedia.search(ticker)
        if not res: return "", "Fail"
        
        # 첫 번째 검색 결과를 가져옵니다.
        page = wikipedia.page(res[0], auto_suggest=False)
        content = page.content
        if "== References ==" in content: content = content.split("== References ==")[0]
        return content.strip(), "Success"
    except Exception as e:
        # print(f"Wikipedia Error for {ticker}: {e}")
        return "", "Fail"

def clean_text(text):
    """KeyBERT 입력용 텍스트 정제"""
    if not text: return ""
    text = re.sub(r'\[\d+\]', ' ', text) # [1], [2] 같은 주석 제거
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def enrich_data(ticker, raw_name, wiki_text):
    """단일 항목에 대한 전체 데이터 생성 로직"""
    
    # yfinance Summary 가져오기
    try: yf_summary = yf.Ticker(ticker).info.get('longBusinessSummary', "")
    except: yf_summary = ""

    # KeyBERT 추출
    combined = f"{yf_summary} {wiki_text}"
    cleaned = clean_text(combined)
    auto_kwd = ""
    try:
        if len(cleaned) > 50:
            kwd_list = kw_model.extract_keywords(cleaned, keyphrase_ngram_range=(1, 2), stop_words=stop_words, top_n=20)
            auto_kwd = ", ".join([k[0] for k in kwd_list])
    except: pass
    
    # 최종 데이터 조립
    manual_kwd = MANUAL_KEYWORDS.get(ticker, "")
    enriched = f"Ticker: {ticker}. Name: {raw_name}. Key Tags: {manual_kwd}. Keywords: {auto_kwd}. Summary: {yf_summary}."
    
    if len(yf_summary) < 100: enriched += f" Context: {cleaned[:500]}"
    
    return {
        "Ticker": ticker,
        "Name": clean_company_name(raw_name), # 이름 정규화
        "Enriched_Text": enriched,
        "Generated_Keywords": auto_kwd
    }

# ---------------------------------------------------------
# [Main] 패치 실행 로직
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. 기존 데이터 로드
    print(f"Loading existing data from {INPUT_OUTPUT_FILE}...")
    try:
        with open(INPUT_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("❌ File not found. Please check the path.")
        exit()

    updated_count = 0
    
    # 2. 항목 찾기 및 업데이트
    print("Searching for GOOGL/GOOG entries...")
    
    for i, item in enumerate(data):
        ticker = item.get('Ticker')
        raw_name = item.get('Name')
        
        # GOOGL 또는 GOOG 항목일 경우
        if ticker in ['GOOGL', 'GOOG']:
            print(f"Found {ticker}. Starting patch enrichment...")
            
            # 1. 위키피디아에서 수정된 매핑을 사용하여 재검색
            wiki_text, status = smart_search(raw_name, ticker)
            
            if status == "Fail" or not wiki_text:
                print(f"Failed to retrieve new Wikipedia text for {ticker}. Skipping update.")
                continue

            # 2. KeyBERT, Summary 등을 사용하여 데이터 재생성
            new_data = enrich_data(ticker, raw_name, wiki_text)
            
            # 3. 리스트에서 기존 데이터 교체
            data[i] = new_data
            updated_count += 1
            print(f"Successfully patched {ticker}.")

            # API 호출 지연 방지 (선택 사항, yfinance를 여러 번 호출할 경우)
            time.sleep(1) 

    # 3. 업데이트된 데이터를 파일에 다시 저장
    if updated_count > 0:
        print(f"\nSaving {updated_count} updated entries back to {INPUT_OUTPUT_FILE}...")
        try:
            with open(INPUT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print("Patching complete! File updated successfully.")
        except Exception as e:
            print(f"Error during file save: {e}")
    else:
        print("No entries were updated.")