import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as gemini
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "../.env"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    gemini.configure(api_key=GOOGLE_API_KEY)
    llm_model = gemini.GenerativeModel(
        'gemini-flash-lite-latest',
        generation_config = {"temperature": 0.0})
    
BENCHMARK_FILE = os.path.join(BASE_DIR, "benchmark.json")
with open(BENCHMARK_FILE, "r", encoding = "utf-8") as f:
    questions = json.load(f)
    
print("Loading MiniLM System...")
model_mini = SentenceTransformer('all-MiniLM-L6-v2')
client_mini = chromadb.PersistentClient(path = os.path.join(BASE_DIR, "chroma_db_mini"))
coll_mini = client_mini.get_collection("sp500_companies")

print("Loading BGE-M3 Model")
model_bge = SentenceTransformer('BAAI/bge-m3')
client_bge = chromadb.PersistentClient(path = os.path.join(BASE_DIR, "chroma_db_bge"))
coll_bge = client_bge.get_collection('sp500_companies')

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def ask_gemini_analysis(tweet):
    if not GOOGLE_API_KEY: return ""
    
    prompt = f"""
    You are a financial slang expert. Identify the company ticker related to this tweet.
    
    [Rules]
    - Understand stock market slang (e.g., "bag holder", "mooning", "diamond hands").
    - Identify CEO nicknames (e.g., "Zuck" -> META, "Jensen" -> NVDA, "Elon" -> TSLA).
    - If the tweet mentions a product (e.g., "Gemini", "Optimus"), find the parent company.
    
    [Examples]
    Tweet: "Zuck is printing money with Reels." -> Result: "Company: Meta Platforms, Ticker: META"
    Tweet: "Jensen's leather jacket is bullish for AI." -> Result: "Company: NVIDIA, Ticker: NVDA"
    Tweet: "Just got the Blue Screen of Death at the airport." -> Result: "Company: CrowdStrike, Ticker: CRWD"
    
    [Task]
    Tweet: "{tweet}"
    Return ONLY format: "Company: [Name], Ticker: [Ticker]"
    """
    
    try: return llm_model.generate_content(prompt).text.strip()
    except: return ""
    
def run_retrieval(model, collection, query):
    vec = model.encode(query).tolist()
    res = collection.query(query_embeddings = [vec], n_results = 3)
    if res['ids'] and res['ids'][0]:
        return res['ids'][0]
    return []

def run_rerank_strategy(model, collection, reranker, query):
    vec = model.encode(query).tolist()
    results = collection.query(query_embeddings = [vec], n_results = 10)
    
    if not results['ids'] or not results['ids'][0]:
        return []
    
    ids = results['ids'][0]
    docs = results['documents'][0]
    pairs = [[query, doc] for doc in docs]
    
    scores = reranker.predict(pairs)
    scored_candidates = []
    for i in range(len(ids)):
        scored_candidates.append({"ticker": ids[i], "score": scores[i]})
        
    scored_candidates.sort(key = lambda x: x['score'], reverse = True)
    
    return [item['ticker'] for item in scored_candidates[:3]]

results = {
    "MiniLM Only": 0,
    "BGE-M3 Only": 0,
    "MiniLM + Gemini": 0,
    "BGE-M3 + Gemini": 0,
    "Gemini Only": 0,
    "MiniLM + Gemini + Rerank": 0,
    "BGE-M3 + Gemini + Rerank": 0
}

print("\n Starting 5-Way Benchmark...")
print("="*60)

for q in tqdm(questions):
    tweet = q['tweet']
    target = q['target']
    
    ai_analysis = ask_gemini_analysis(tweet)
    hybrid_query = f"{ai_analysis} {tweet}"
    
    preds_1 = run_retrieval(model_mini, coll_mini, tweet)
    if target in preds_1: 
        results["MiniLM Only"] += 1
    
    preds_2 = run_retrieval(model_bge, coll_bge, tweet)
    if target in preds_2: 
        results["BGE-M3 Only"] += 1

    preds_3 = run_retrieval(model_mini, coll_mini, hybrid_query)
    if target in preds_3: 
        results["MiniLM + Gemini"] += 1
    else:
        wrong_pred = preds_3[0] if preds_3 else "None"
        print(f"\n[MiniLM+Gemini] Target: {target} | Predicted: {wrong_pred}")
        print(f"   Tweet: {tweet[:60]}...")
        print(f"   Gemini Hint: {ai_analysis[:100]}...")
        
    preds_4 = run_retrieval(model_bge, coll_bge, hybrid_query)
    if target in preds_4: 
        results["BGE-M3 + Gemini"] += 1
    else:
        wrong_pred = preds_4[0] if preds_4 else "None"
        print(f"\n[BGE+Gemini] Target: {target} | Predicted: {wrong_pred}")
        print(f"   Tweet: {tweet[:60]}...")
        
    preds_5 = run_rerank_strategy(model_mini, coll_mini, reranker, hybrid_query)
    if target in preds_5: 
        results["MiniLM + Gemini + Rerank"] += 1
    else:
        wrong_pred = preds_5[0] if preds_5 else "None"
        print(f"\n[MiniLM+Rerank] Target: {target} | Predicted: {wrong_pred}")
        
    preds_6 = run_rerank_strategy(model_bge, coll_bge, reranker, hybrid_query)
    if target in preds_6: 
        results["BGE-M3 + Gemini + Rerank"] += 1
    else:
        wrong_pred = preds_6[0] if preds_6 else "None"
        print(f"\n[BGE+Rerank] Target: {target} | Predicted: {wrong_pred}")
        print(f"   Tweet: {tweet[:60]}...")
        print(f"   Gemini Hint: {ai_analysis[:100]}...")
        
    if target in ai_analysis: 
        results["Gemini Only"] += 1
    
    time.sleep(0.5)
    
print("\n Benchmark Results")
print("-"*60)
print(f"{'Strategy':<20} | {'Score':<10} | {'Accuracy'}")
print("-"*60)

total = len(questions)
for name, score in results.items():
    acc = (score / total) * 100
    print(f"{name:<20} | {score}/{total:<10} | {acc:.1f}%")
print("-"*60)