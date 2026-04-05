import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
import torch

# 1. 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading Model on {device.upper()}...")
sentence_model = SentenceTransformer('BAAI/bge-m3', device=device)
kw_model = KeyBERT(model=sentence_model)

# 2. NLTK 리소스 다운로드 (Docker에 이미 있지만 안전하게)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet = True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # 품사 태깅용

stops = set(stopwords.words('english'))

# [기존 노이즈 키워드 유지]
NOISE_KEYWORDS = {
    "tech", "technology", "software", "hardware", "system", "systems", 
    "group", "global", "international", "service", "services", "solution", "solutions", 
    "company", "companies", "corp", "corporation", "holding", "holdings", 
    "inc", "ltd", "plc", "investment", "capital", "financial", "finance", "business",
    "sector", "industry", "provider", "offers", "provides", "operates", "headquartered",
    "subsidiaries", "subsidiary", "listed", "public", "private", "conglomerate",
    "component", "components", "index", "indices", "stock", "market", "nasdaq", "nyse",
    "role", "filled", "became", "become", "based", "located", "place", "headquarters",
    "sells", "sale", "provision", "involved", "related", "creation", "created",
    "segment", "segments", "operate", "operating", "operation", "operations",
    
    "history", "founded", "founder", "established", "incorporated", 
    "lawsuit", "sued", "legal", "settlement", "alleged", "accused", "violation", "investigation", 
    "controversy", "controversies", "ethical", "criticized", "criticism", "faced", "facing", 
    "bankruptcy", "bankrupt", "debt", "stolen", "seized", "account", 
    "termination", "terminated", "layoff", "layoffs", "mass", "fired", "resignation", 
    "antitrust", "monopoly", "fine", "fined", "penalty", "judge", "court", 
    "uber", "russia", "china", "eu", "european", "commission", 
    
    "largest", "biggest", "world", "major", "widely", "described", "numerous", "various",
    "united", "states", "american", "california", "york", "mountain", "view", 
    "million", "billion", "trillion", "revenue", "profit", "income", "year", "years",
    "ceo", "announced", "appointed", "member", "board", "executive", "officer",
    "employees", "shareholders", "including", "multinational", "third", "symbols", 
    "enterprises", "calendar", "impacted", "canada", "latin", "autonomy", "cleaner", 
    "accountable", "posts", "africa", "asia", "upon", "losing", "establishment", 
    "prompted", "remain", "sundar", "pichai", "larry", "page", "sergey", "brin",
    
    "also", "include", "includes", "including", "related", "various", "customers", 
    "products", "platforms", "services", "segment", "segments", "offers", "provides",
    "bets", "other", "purchases", "sale", "consumer", "provision", "involved",
    "based", "tools", "internet", "digital", "content", "devices", "apps", "customers",
    "infrastructure", "analytics", "collaboration", "communication", "enterprises",
    "amazon", "apple", "united", "states", "europe", "africa", "asia", "canada"
}
stops.update(NOISE_KEYWORDS)

# 3. 테스트할 텍스트
target_text = """
Alphabet Inc. offers various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud, and Other Bets segments. The Google Services segment provides products and services, including ads, Android, Chrome, devices, Gmail, Google Drive, Google Maps, Google Photos, Google Play, Search, and YouTube. It is also involved in the sale of apps and in-app purchases and digital content in the Google Play and YouTube; and devices, as well as in the provision of YouTube consumer subscription services. The Google Cloud segment offers AI infrastructure, Vertex AI platform, cybersecurity, data and analytics, and other services; Google Workspace that include cloud-based communication and collaboration tools for enterprises, such as Calendar, Gmail, Docs, Drive, and Meet; and other services for enterprise customers. The Other Bets segment sells healthcare-related and internet services. The company was incorporated in 1998 and is headquartered in Mountain View, California.. Alphabet Inc. is an American multinational technology conglomerate holding company headquartered in Mountain View, California. Alphabet is the world's third-largest technology company by revenue, after Amazon and Apple, the largest technology company by profit, and one of the world's most valuable companies. It was created through a restructuring of Google on October 2, 2015, and became the parent holding company of Google and several former Google subsidiaries. Alphabet is listed on the large-cap section of the Nasdaq under the ticker symbols GOOGL and GOOG; both classes of stock are components of major stock market indices such as the S&P 500 and Nasdaq-100. Alphabet has been described as a Big Tech company.
The establishment of Alphabet Inc. was prompted by a desire to make the core Google business "cleaner and more accountable" while allowing greater autonomy to group companies that operate in businesses other than Internet services. Founders Larry Page and Sergey Brin announced their resignation from their executive posts in December 2019, with the CEO role to be filled by Sundar Pichai, who is also the CEO of Google. Page and Brin remain employees, board members, and controlling shareholders of Alphabet Inc.
Alphabet Inc. has faced numerous legal and ethical controversies, including a 2017 lawsuit against Uber over stolen self-driving technology, a 2020 privacy settlement over Google+ data exposure, and multiple antitrust actions from the United States, France, and Japan. It has also been accused of labor law violations related to worker organizing and was forced to file for bankruptcy in Russia after its bank account was seized in 2022. In 2023, the company was widely criticized for mass layoffs that impacted 12,000 employees, many of whom discovered their termination only upon losing account access.
"""

def clean_text_aggressive(text):
    text = text.lower()
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text) # 숫자 제거
    text = re.sub(r'[^a-zA-Z\s.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("\n🧹 Cleaning Text...")
cleaned_source = clean_text_aggressive(target_text)

# 불용어 추가
current_stops = list(stops) + ["alphabet", "google", "goog", "googl"]

# [핵심 기능] 명사 필터링 함수
def extract_clean_phrases(text, top_n=20):
    # 3배수 추출 후 필터링
    candidates = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=current_stops,
        use_mmr=True,
        diversity=0.7,
        top_n=top_n * 3
    )
    
    final_keywords = []
    
    for kw, score in candidates:
        tokens = word_tokenize(kw)
        tags = pos_tag(tokens)
        
        # 유효성 검사 플래그
        is_valid = True
        has_noun = False
        
        for word, tag in tags:
            # 1. 명사(NN)가 하나라도 있어야 함
            if tag.startswith('NN'): 
                has_noun = True
            
            # 2. [엄격] 동사(VB), 부사(RB), 전치사(IN), 접속사(CC), 관형사(DT) 절대 금지
            # also(RB), include(VB), of(IN), the(DT) 등이 포함되면 즉시 탈락
            if tag.startswith('VB') or tag.startswith('RB') or tag.startswith('IN') or tag.startswith('CC') or tag.startswith('DT'):
                is_valid = False
                break
        
        # 3. 길이가 2글자 이하인 단어(예: 'ai'는 살리되, 'us', 'it' 등) 필터링
        if len(kw) < 3 and kw != "ai":
            is_valid = False

        if is_valid and has_noun:
            final_keywords.append((kw, score))
            
        if len(final_keywords) >= top_n:
            break
            
    return final_keywords

print("🔍 Extracting Keywords (Noun-Focused)...")
keywords_list = extract_clean_phrases(cleaned_source, top_n=20)

print("\n" + "="*50)
print("✅ FINAL GENERATED KEYWORDS (GOOGL)")
print("="*50)
print(", ".join([k[0] for k in keywords_list]))
print("="*50)