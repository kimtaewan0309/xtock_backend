import chromadb
import os
import sys

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH_STATIC = os.path.join(BASE_DIR, "chromaDB/static")
DB_PATH_DYNAMIC = os.path.join(BASE_DIR, "chromaDB/dynamic")

print(f"📂 Comparing IDs between Static and Dynamic DB...")

try:
    # 1. Static DB ID 가져오기
    client_static = chromadb.PersistentClient(path=DB_PATH_STATIC)
    col_static = client_static.get_collection("sbert")
    static_ids = set(col_static.get()['ids'])
    print(f"✅ Static DB Keys: {len(static_ids)} (e.g., {list(static_ids)[:3]})")

    # 2. Dynamic DB ID 가져오기
    client_dynamic = chromadb.PersistentClient(path=DB_PATH_DYNAMIC)
    col_dynamic = client_dynamic.get_collection("sbert")
    dynamic_ids = set(col_dynamic.get()['ids'])
    print(f"✅ Dynamic DB Keys: {len(dynamic_ids)} (e.g., {list(dynamic_ids)[:3]})")

    # 3. 교집합(매칭) 확인
    common = static_ids.intersection(dynamic_ids)
    print(f"🔗 Matched IDs: {len(common)}")
    
    # 4. 불일치 분석
    missing_in_dynamic = static_ids - dynamic_ids
    if missing_in_dynamic:
        print(f"❌ Missing in Dynamic (Static엔 있는데 Dynamic엔 없음): {len(missing_in_dynamic)}")
        print(f"   Sample Missing: {list(missing_in_dynamic)[:5]}")
    else:
        print("🎉 Perfect Match! All Static IDs exist in Dynamic DB.")

    # 5. AMZN 강제 조회 테스트
    target = "AMZN"
    print(f"\n🔎 Testing Fetch '{target}'...")
    res = col_dynamic.get(ids=[target], include=['embeddings'])
    if res['ids']:
        print(f"   Found: {res['ids']}")
        print(f"   Embedding Size: {len(res['embeddings'][0]) if res['embeddings'] else 'None'}")
    else:
        print(f"   ❌ '{target}' NOT FOUND in Dynamic DB!")

except Exception as e:
    print(f"❌ Error: {e}")