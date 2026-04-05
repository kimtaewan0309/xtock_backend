import google.generativeai as genai
import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


load_dotenv(os.path.join(BASE_DIR, "../.env"))
 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    GOOGLE_API_KEY = GOOGLE_API_KEY.strip()
if not GOOGLE_API_KEY:
    raise ValueError("Don't Exist GOOGLE API KEY. Check .env file")
genai.configure(api_key = GOOGLE_API_KEY)

llm_model = genai.GenerativeModel('gemini-flash-lite-latest')
sbert_model = SentenceTransformer('BAAI/bge-m3')

db_path = os.path.join(BASE_DIR, './chroma_db')

client = chromadb.PersistentClient(path = db_path)
try:
    col_wiki = client.get_collection(name = 'sp500_companies')
    col_fin = client.get_collection(name = 'sp500_financials')
    print("Successfully loaded Wiki & Financial DB.")
except Exception as e:
    print(f"DB Load Error: {e}")
    exit()

def ask_gemini_for_context(tweet_text):
    prompt = f"""
    Analyze the following tweet and identify the specific public company related to it.
    If the product mentioned belongs to a specific company (e.g., 'Gemini' -> Google, 'Optimus' -> Tesla), identify that company.
    
    Tweet: "{tweet_text}"
    
    Return ONLY the Company Name and Ticker Symbol in this format:
    "Company: [Name], Ticker: [Ticker]"
    If you are not sure, return "Unknown".
    """
    
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f" Gemini API Error: {e}")
        return ""

    
def hybrid_search(tweet_text):
    ai_context = ask_gemini_for_context(tweet_text)
    print(f" Gemini Context: {ai_context}")
    
    if "Unknown" not in ai_context and ai_context != "":
        expanded_query = f"{ai_context} {tweet_text}"
    else:
        expanded_query = tweet_text
        
    query_vector = sbert_model.encode(expanded_query).tolist()

    financial_keywords = ['revenue', 'profit', 'loss', 'earnings', 'quarter', 'margin', 'debt', 'billion', 'million', 'fiscal']

    is_financial = any(word in tweet_text.lower() for word in financial_keywords)

    if is_financial:
        W_WIKI = 0.5
        W_FIN = 0.5

    else:
        W_WIKI = 0.8
        W_FIN = 0.2
    
    res_wiki = col_wiki.query(query_embeddings = [query_vector], n_results = 10)
    res_fin = col_fin.query(query_embeddings = [query_vector], n_results = 10)

    final_scores = {}
    metadata_map = {}
    
    if res_wiki['ids'] and res_wiki['ids'][0]:
        for i in range(len(res_wiki['ids'][0])):
            ticker = res_wiki['ids'][0][i]
            raw_score = 2.0 - res_wiki['distances'][0][i]
            weighted_score = raw_score * W_WIKI

            final_scores[ticker] = final_scores.get(ticker, 0) + weighted_score

            metadata_map[ticker] = res_wiki['metadatas'][0][i]

    if res_fin['ids'] and res_fin['ids'][0]:
        for i in range(len(res_fin['ids'][0])):
            ticker = res_fin['ids'][0][i]
            raw_score = 2.0 - res_fin['distances'][0][i]
            weighted_score = raw_score * W_FIN

            final_scores[ticker] = final_scores.get(ticker, 0) + weighted_score

            if ticker not in metadata_map:
                metadata_map[ticker] = res_fin['metadatas'][0][i]

    sorted_candidates = sorted(final_scores.items(), key = lambda x: x[1], reverse = True)[:3]

    final_matches = []
    for rank, (ticker, score) in enumerate(sorted_candidates):
        meta = metadata_map.get(ticker, {})
        company_name = meta.get('Name', meta.get('name', 'Unknown'))
        keywords = meta.get('Keywords', meta.get('keywords', meta.get('Generated_Ketwords', '')))
        final_matches.append({
            "rank": rank + 1,
            "ticker": ticker,
            "name": company_name,
            "score": score,
            "keywords": keywords
        })

    return {
        "input_text":tweet_text,
        "matches": final_matches
    }

        
if __name__ == "__main__":
    test_tweets = [
        "Tech giant posts record revenue of 100 billion dollars this quarter.", # Google/Apple 예상
        "The company reported a massive net loss due to heavy investment in AI infrastructure.", # 적자 기업 예상
        "Zepbound sales are exploding, overtaking competitors."    ]

    for tweet in test_tweets:
        print("=" * 60)
        
        # 1. 검색 함수 호출 (트윗 1개씩)
        matches = hybrid_search(tweet)

        
        if not matches:
            print("No matches found.")
        else:            
            print(f"Matched Companies: ")
            for m in matches['matches']:
                print(f" -{m['ticker']} ({m['name']}) Score: {m['score']:.2f}")
        print("\n")