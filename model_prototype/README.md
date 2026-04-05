- 금융 관련 트윗을 확인 후 관련 기업을 매칭하는 모델 개발
- 1차로 AI 모델을 통해 가장 유사한 기업의 티커와 기업명을 get
- 2차로 wikipedia의 S&P500에 속하는 기업들의 내용을 SBERT와 KEYBERT를 통한 벡터화, 핵심 단어 추출 후 트윗과 유사도 계산
- 최종적으로 유사도 점수가 가장 높은 기업 top 3개 return
* 추후 개발할 부분
1. 특정 기업의 트윗이 아닌 미장 전반적인 내용의 트윗 처리 -> SPYM/QQQ와 같은 ETF 활용
2. 정확도 개선 부분 -> AI를 사용하지 않았을 경우 wikipedia의 내용이 최신 정보를 포함하지 않기 때문에 정확도 감소
3. 추가적인 재무 재표 활용 -> 어느 정도 비율로 가중치를 둬서 정확도를 올릴건지


**파일 실행 방법**
- Docker Container를 활용해서 필요 라이브러리를 추가적인 설치 없이 진행
- 최상단 폴더에 .env 파일을 만들고 파일 내에
GOOGLE_API_KEY=(본인의 GOOGLE_API_KEY value)를 넣고 저장
- Docker Container를 설치한 후 VScode를 통해 폴더를 열고 터미널에 "docker-compose up -d --build"를 입력
- 빌드가 완료되고 나서 data_pipeline 폴더 내에 있는 crawling.py -> pipeline.py 순으로 코드를 실행 (실행 코드는 "docker exec -it xtock-pipeline python (code_file_name).py)
이때 sp500_full_wiki.jsonl 파일이 생성된 후에 pipeline.py 코드를 실행하면 sp500_sbert_input.jsonl 파일이 생성
- 파일이 모두 생성됐으면 파일이 제대로 생성됐는지 확인 후 build_vector.py 코드를 실행 (실행 코드는 "docker exec -it xtock-pipeline python build_vector.py)
- backend 폴더에 chroma_db 폴더가 생성된 것을 확인 후에 test_hybrid_search.py 코드를 실행하면 test tweet을 기반으로 유사도가 높은 top3 기업을 출력(실행 코드는 "docker exec -it xtock-backend python test_hybrid_search.py)


- 벤치 마크 모델을 추가
  - SBERT 모델 중 MiniLM 모델과 BGE-M3 모델의 성능을 비교
  - Ai Studio를 활용해서 AI의 도움을 받을 경우와 받지 않을 경우의 성능을 비교
  - AI만 사용했을 때의 모델도 추가하여 섣능을 비교
  - * AI의 temperature를 0.0으로 설정하여 항상 같은 결과값이 나오도록 고정, 프롬프트는 동일하게 사용
  - 최초 성능은 SBERT 모델과 제미나이를 합친 모델의 성능이 가장 우수한 것으로 확인
  - 벤치마크를 위한 트윗의 난이도를 점점 올려가면서 시도한 결과 가장 성능이 우수한 모델은 BGE 모델과 제미나이를 합친 모델의 성능이 가장 우수한 것으로 확인
  - <img width="445" height="208" alt="image" src="https://github.com/user-attachments/assets/454d1a9a-3a93-4dd9-91a5-7873882c9a9d" />


# 12/10일자 업데이트

- 선준님이 개발한 모델 + 제가 개발한 모델을 통합하여 최종 모델을 생성
## **파일 실행 방법**
- 위와 마찬가지로 Docker Container를 사용해서 코드를 실행
### 1. 컨테이너 빌드 및 실행
- model_prototype 폴더로 이동한 다음
  ```docker-compose up -d --build```
  를 통해 도커를 시작
### 2. 데이터 수집
- 빌드가 완료된 후
  1. S&P 500 기업 리스트 최신화
    ```
    docker exec -it xtock-pipeline python data_pipeline/update_sp500_list.py 
    ```
  2. 위피키디아 데이터 수집 및 키워드 추출
    ```
    docker exec -it xtock-pipeline python data_pipeline/pipeline.py
    ```
  3. 최신 재무지표 수집
    ```
    docker exec -it xtock-pipeline python data_pipeline/fetch_financial_metrics.py
    ```
- 데이터 파일이 깃허브에 없기 때문에 새롭게 생성해야되는데 시간이 오래 소요됨
- 따라서 데이터 파일을 따로 빼서 관리 (google drive)
### 3. 벡터 DB 구축
- 수집된 데이터를 바탕으로 검색에 사용될 ChromaDB (Static + Dynamic)을 구축
  ```
  docker exec -it xtock-pipeline python data_pipeline/build_dual_db.py
  ```
- 마찬가지로 ChromaDB 파일의 크기가 크기 때문에 google drive를 통해 따로 관리
### 4. 모델 학습 (Hyperparameter Tuning)
- 검색 정확도를 높이기 위해 Optuna를 사용하여 최적의 가중치를 찾음
- 학습 결과는 backend/best_params.json에 저장 (google drive를 통해 관리)
  ```
  docker exec -it xtock-pipeline python backend/train_model.py
  ```
### 5. 검색 API 서버 실행
- FastAPI 서버를 실행하여 외부에서 검색 요청을 받을 수 있도록 함
  ```
  docker exec -it xtock-pipeline uvicorn backend.search_service:app --host 0.0.0.0 --port 8000 --reload
  ```
## API 테스트 하는 방법
- url: http://localhost:8001/docs
- 해당 url을 통해서 query를 전송할 수 있도 response 확인 가능

# 12/11일자 업데이트

- 통합한 모델을 백엔드와 통합
- 태완님이 작성한 백엔드 코드를 활용해서 최종적으로 코드 작성
  - X API를 통해서 트윗을 가져오는 부분도 성공
  - 다만 free tier에서는 최근 일주일간의 트윗만 가져올 수 있음
  - 현재로썬 모델의 성능을 확인하기 위해 impact_tweets.json 파일을 제작해서 관련 기업 내용이 검색되면 사용자에게 보여주도록 코드를 임시 수정
  - 해당 부분은 더 개발이 필요할듯
- Swagger UI를 통해서 response 확인 가능
- 수빈님이 제작하신 프론트 모델과 백엔드를 연결
- 더미 데이터 부분을 실제 백엔드 파트와 연결 완료
- 자잘한 워딩이나 로그인 기능, 사용자 정보 등등 추가 구현할 부분 필요
- **현재 종목 입력 칸에 아무 입력값만 넣어도 결과가 랜덤하게 생성되는 오류 발생**
- **해당 부분은 수정해서 관련 없는 입력값이 발생했을 경우 "관련 기업 없음"과 같이 사용자에게 보여지도록 수정해야될듯**


# 12/12일자 업데이트

- 회의때 말했던 내용 수정 완료
  - 기존 종목 분석은 최근 기업 근황 / 과거 영향 분석 탭으로 나눔
  - 최근 기업 근황은 X API를 사용해서 기업의 최근 트윗을 실시간으로 가져옴
    - S&P 500에 포함되는 모든 기업들의 트윗을 가져올 수 있도록 검색 창을 클릭시 S&P500에 해당되는 기업들의 이름과 티커가 리스트 형식으로 보여짐
    - 클릭시 바로 검색창에 입력되고 해당 기업의 공식 X 계정으로부터 트윗을 가져옴
    - 초보자도 보기 쉽게 최근 30일간의 OHLCV를 표로 보여줌
  - 과거 영향 분석은 kaggle에서 획득한 주가 관련 트윗을 활용함
    - 사용자가 검색창에 관련 인물 혹은 특정 사건을 입력 시 그와 관련된 트윗을 매핑해서 사용자에게 보여줌
    - **아직 성능이 100% 좋게 나오지는 않고 동일한 티커를 가지고 있으면 다 가져오는것 처럼 보여서 수정 필요**
    - 트윗이 발행됐을 당시의 주가와 +5일 후의 수익률을 함께 보여주도록 수정
   
## 서버 띄우는 방법
- Docker Container 필수
- 백엔드와 프론트엔드에서 사용되는 모든 json, csv, chromaDB 등등 용량 문제로 깃허브에 올라가지 않은 파일들은 공유해드린 구글 드라이브에서 동일한 폴더에 넣고 사용하시면 됩니다.
- **만약 PC에 GPU가 없는 경우에 'docker-compose.yml' 파일 내에 service: backend 부분에 deploy: 에 해당하는 모든 부분들을 주석처리해야합니다.**
- Docker Container 설치 후 VSCode에서 XTock-Xignal_prototype 폴더를 연 후
  ```
  cd model_prototype
  ```
  으로 디렉토리를 변경
- ```
  docker-compose up -d --build
  ```
  를 터미널에 입력 후 도커 컨테이너를 띄우면 서버 실행 (아마 시간이 오래 걸릴수도 있습니다. 노트북으로는 약 10분 ~20분 소요 가능)
- http://localhost:8000/docs 로 이동해서 백엔드 서버가 열렸는지, API 응답은 잘되는지 확인합니다.
  - 이때 에러가 나거나 응답이 안오면
    ```
    docker logs -f xtock-backend
    ```
    를 입력해서 백엔드 서버의 로그를 확인
- 백엔드 서버가 살아있으면 http://localhost:3000 으로 이동해서 웹페이지 확인
