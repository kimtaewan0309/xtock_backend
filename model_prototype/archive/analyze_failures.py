import json
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import ALIASES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH_STATIC = os.path.join(BASE_DIR, "chromaDB/static")
DB_PATH_DYNAMIC = os.path.join(BASE_DIR, "chromaDB/dynamic")
BEST_PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")

# 1. 최적 파라미터 로드
if os.path.exists(BEST_PARAMS_PATH):
    with open(BEST_PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print(f"Loaded Best Params: {params}")
    alpha = params['alpha']
    beta = params['beta']
    lambda_mention = params['lambda_mention']
else:
    print("best_params.json not found. Using defaults.")
    alpha, beta, lambda_mention = 0.7, 0.3, 1.0

# 2. 모델 및 DB 준비
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('BAAI/bge-m3', device=device)
client_static = chromadb.PersistentClient(path=DB_PATH_STATIC)
client_dynamic = chromadb.PersistentClient(path=DB_PATH_DYNAMIC)
col_static = client_static.get_collection("sbert")
col_dynamic = client_dynamic.get_collection("sbert")

def calculate_score(query_vec, vec_static, vec_dyn, mention_bonus):
    return (alpha * np.dot(query_vec, vec_static)) + (beta * np.dot(query_vec, vec_dyn)) + mention_bonus

def detect_mention_score(text, ticker):
    text_lower = text.lower()
    if f"${ticker.lower()}" in text_lower: return 1.0
    if ticker in ALIASES:
        for alias in ALIASES[ticker]:
            if alias in text_lower: return 1.0
    return 0.0

# 3. 테스트 (일부 데이터만)
TRAIN_FILE = os.path.join(BASE_DIR, "json/train_news.json")
with open(TRAIN_FILE, 'r') as f:
    data = json.load(f)[:50] # 50개만 테스트

print("\nAnalyzing Failure Cases (Rank > 1)...")
print("-" * 60)

for item in data:
    query = item.get('description') or item.get('text')
    if not query: continue
    
    true_labels = item.get('sp500_labels') or item.get('tickers')
    if not true_labels: continue
    if isinstance(true_labels, str): true_labels = [true_labels]
    
    query_vec = model.encode(query, convert_to_numpy=True)
    
    res = col_static.query(query_embeddings=[query_vec.tolist()], n_results=20, include=['embeddings'])
    if not res['ids']: continue
    
    candidates = []
    ids = res['ids'][0]
    embeddings = res['embeddings'][0]
    
    for i, ticker in enumerate(ids):
        vec_static = np.array(embeddings[i])
        try:
            res_dyn = col_dynamic.get(ids=[ticker], include=['embeddings'])
            if res_dyn['embeddings']:
                vec_dyn = np.array(res_dyn['embeddings'][0])
            else:
                vec_dyn = np.zeros_like(vec_static)
        except:
            vec_dyn = np.zeros_like(vec_static)
            
        mention = detect_mention_score(query, ticker) * lambda_mention
        score = calculate_score(query_vec, vec_static, vec_dyn, mention)
        candidates.append((ticker, score))
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 1등이 정답이 아닌 경우만 출력
    top_ticker = candidates[0][0]
    if top_ticker not in true_labels:
        print(f"Query: {query[:80]}...")
        print(f"    Expected: {true_labels}")
        print(f"    Model Pred: {top_ticker} (Rank 1)")
        
        found_rank = -1
        for rank, (t, s) in enumerate(candidates):
            if t in true_labels:
                found_rank = rank + 1
                break
        print(f"    Real Answer Rank: #{found_rank}")
        print("-" * 60)