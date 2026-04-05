import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def build_fin_db():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(BASE_DIR, "sp500_financials.jsonl")
    
    db_path = os.path.join(BASE_DIR, "chroma_db")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return
    print(f"Loading Financial Data from {input_file}")
    df = pd.read_json(input_file, lines = True)
    
    print("Loading Model (BGE-M3)")
    
    model = SentenceTransformer('BAAI/bge-m3')
    
    client = chromadb.PersistentClient(path = db_path)
    
    collection_name = "sp500_financials"
    
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(name = collection_name)
    
    print(f"Indexing {len(df)} financials records")
    
    batch_size = 5
    for i in tqdm(range(0, len(df), batch_size), desc = "indexing Finance"):
        batch = df.iloc[i : i + batch_size]
        
        ids = batch['Ticker'].tolist()
        docs = batch['Financial_Text'].tolist()
        metadatas = batch[['Ticker', 'Name']].to_dict(orient = 'records')
        
        embeddings = model.encode(docs, show_progress_bar = False).tolist()
        
        collection.add(
            ids = ids,
            embeddings = embeddings,
            metadatas = metadatas,
            documents = docs
        )
        
    print(f" Financial DB Added to '{db_path}' (Collection: {collection_name})")
    
if __name__ == "__main__":
    build_fin_db()    