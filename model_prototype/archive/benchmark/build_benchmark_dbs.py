import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import shutil
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "../data_pipeline/sp500_sbert_input.jsonl")

PATH_MINI = os.path.join(BASE_DIR, "chroma_db_mini")
PATH_BGE = os.path.join(BASE_DIR, "chroma_db_bge")

def build_specific_db(model_name, db_path):
    print(f"Building DB for [{model_name}] -> {db_path}")
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    
    df = pd.read_json(INPUT_FILE, lines = True)
    
    print("Loading Model...")
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path = db_path)
    collection = client.create_collection(name = "sp500_companies")
    
    batch_size = 1 if "bge" in model_name else 32
    
    print("cw Embedding and Indexing")
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i: i + batch_size]
        
        ids = batch['Ticker'].tolist()
        docs = batch['Enriched_Text'].tolist()
        
        metadatas = []
        for _, row in batch.iterrows():
            metadatas.append({
                "ticker": row['Ticker'],
                "name": row['Name'],
                "keywords": row.get('Generated_Keywords', '')
            })
            
        embeddings = model.encode(docs).tolist()
        collection.add(ids = ids, embeddings = embeddings, metadatas = metadatas, documents = docs)
        
    print(f"DB Created at {db_path}")
    
if __name__ == "__main__":
    build_specific_db('all-MiniLM-L6-v2', PATH_MINI)
    build_specific_db('BAAI/bge-m3', PATH_BGE)