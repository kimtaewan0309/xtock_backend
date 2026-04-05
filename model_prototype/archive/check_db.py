import chromadb
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH_STATIC = os.path.join(BASE_DIR, "chromaDB/static")
DB_PATH_DYNAMIC = os.path.join(BASE_DIR, "chromaDB/dynamic")

print(f"📂 Checking DB at: {BASE_DIR}/chromaDB")

try:
    client_static = chromadb.PersistentClient(path=DB_PATH_STATIC)
    col_static = client_static.get_collection("sbert")
    print(f"✅ Static DB Count: {col_static.count()}")
    
    client_dynamic = chromadb.PersistentClient(path=DB_PATH_DYNAMIC)
    col_dynamic = client_dynamic.get_collection("sbert")
    print(f"✅ Dynamic DB Count: {col_dynamic.count()}")
    
    # 샘플 데이터 하나 조회
    if col_dynamic.count() > 0:
        sample = col_dynamic.peek(1)
        print(f"🔎 Dynamic Sample ID: {sample['ids']}")
        print(f"🔎 Dynamic Sample Meta: {sample['metadatas']}")
    else:
        print("⚠️ Dynamic DB is EMPTY!")

except Exception as e:
    print(f"❌ DB Error: {e}")