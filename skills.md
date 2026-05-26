# AL-CCTV Platform - Skills 정리 및 실무 레퍼런스 가이드

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | AI CCTV 보안 분석 플랫폼 |
| **핵심 아키텍처** | OpenCV 1차 필터링 -> 이상 프레임만 LLM 전달 -> 위험도 분석 |
| **Python** | 3.14 (venv 가상환경) |
| **가상환경 활성화** | PowerShell: `.\venv\Scripts\Activate.ps1` <br> CMD: `.\venv\Scripts\activate.bat` |
| **LLM 모델** | GPT-4o-mini |
| **주요 패키지** | openai 2.36.0, python-dotenv 1.2.2, langchain 1.3.0, langchain-openai 1.2.1, langchain-community 0.4.1, langchain-classic 1.0.7, chromadb, numpy |

---

## Part 01 - LLM 기초 & ChatGPT API

### Skill 1: LLM 개념 계층 및 한계의 아키텍처적 극복
- **파일**: `part01_llm_chatgpt_api/ch01_llm개념이해.md`
- **핵심**: AI -> ML -> DL -> LLM -> ChatGPT의 포함 관계 및 LLM의 4대 한계 극복법
- **자료형 및 개념**:
  - 토큰(Token): API 비용 및 컨텍스트 윈도우의 기본 단위. (영어: 단어 파편, 한국어: 형태소 단위)
- **한계 극복 실무 패턴**:
  1. 지식 컷오프 -> 외부 지식 결합인 RAG(Part 03)로 실시간 대응
  2. 환각(Hallucination) -> 프롬프트 제약 및 RAG 레퍼런스 주입으로 극복
  3. 비용 오버헤드 -> OpenCV 1차 필터링(이상 행동/객체 탐지 프레임만 LLM 전달) 아키텍처 적용
  4. 컨텍스트 길이 -> LangChain LCEL 배치 처리 및 버퍼/요약 메모리 아키텍처(Part 02) 적용

### Skill 2: 환경 설정 (.env + dotenv + masked_key 처리)
- **파일**: `part01_llm_chatgpt_api/ch02_dotenv_apicall.py`
- **핵심**: `.env` 파일을 활용한 API 키 보안 로드 및 클라이언트 마스킹 출력 기법
- **핵심 구현 코드**:
  ```python
  from dotenv import load_dotenv
  import os
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")
  masked_key = api_key[:12] + "..." + api_key[-4:]
  client = OpenAI(api_key=api_key)
  ```
- **실무 주의사항 및 팁**:
  - API 키를 절대 소스 코드에 하드코딩하지 말고 `.gitignore`에 `.env`를 등록해 보안을 유지해야 합니다.
  - 마스킹 기법을 통해 디버깅 로그에 실 키가 노출되는 사고를 방지합니다.

### Skill 3: messages 구조 (3가지 역할과 무상태성 대응)
- **파일**: `part01_llm_chatgpt_api/ch02_message_struct.py`
- **핵심**: API 호출의 무상태성(Stateless)을 대응하기 위해 매번 대화 히스토리 전체를 주입하는 메시지 리스트 아키텍처
- **자료형 및 구조**:
  - `messages`의 구조는 딕셔너리의 리스트(`list[dict[str, str]]`) 형식입니다.
  ```python
  messages = [
      {"role": "system", "content": "당신은 AI CCTV 보안 분석 시스템입니다."},
      {"role": "user", "content": "창고 출입구에서 사람 2명이 탐지됐습니다."},
      {"role": "assistant", "content": "위험도: 주의. 심야 시간대 2인 탐지."}
  ]
  ```
- **실무 주의사항 및 팁**:
  - API는 기억력이 없으므로 이전 대화 맥락을 누적한 전체 리스트를 전송해야 합니다.

### Skill 4: System Prompt 비교 실험 및 정체성 수립
- **파일**: `part01_llm_chatgpt_api/ch02_system_prompt_comparison.py`
- **핵심**: 페르소나 설정 유무에 따른 답변 일관성과 한글 구조화 응답성 성능 차이 증명
- **이 프로젝트 표준 System Prompt**:
  ```
  당신은 AI CCTV 보안 분석 시스템입니다.
  OpenCV로 탐지된 객체 정보를 입력받아 위험도를 분석합니다.
  답변 형식: 위험도(정상/주의/위험) + 판단 근거 + 권고 조치.
  한국어로만 답합니다.
  ```
- **실무 주의사항 및 팁**:
  - 페르소나 지정이 누락되면 불필요한 서술형 영어 답변 등이 발생하여 시스템 파이프라인의 후속 자동화 처리가 불가능해집니다.

### Skill 5: temperature 파라미터 제어를 통한 일관성 확보
- **파일**: `part01_llm_chatgpt_api/ch02_temperature_comparison.py`
- **핵심**: 무작위성 제어 매개변수를 통한 위험 판단의 신뢰성 극대화
- **자료형 및 범위**:
  - `temperature`: float 타입, 0.0 (결정적) ~ 2.0 (무작위) 범위.
- **실무 주의사항 및 팁**:
  - CCTV 위험 분석 및 보안 감사 업무에서는 절대적으로 `temperature = 0.0` 또는 극도로 낮은 값(`0.0 ~ 0.3`)을 고정해 사용해야 합니다. 일관되지 않은 위험 판단은 보안 시스템의 무력화를 유발합니다.

### Skill 6: max_tokens 파라미터와 finish_reason 분기
- **파일**: `part01_llm_chatgpt_api/ch02_maxtoken_comparison.py`
- **핵심**: 토큰 낭비 방지를 위한 상한선(max_tokens) 지정 및 정상 처리 여부 분기 분석
- **핵심 구현 코드**:
  ```python
  finish_reason = response.choices[0].finish_reason
  # "stop" -> 정상 완료 / "length" -> 토큰 초과 잘림 / "content_filter" -> 보안 필터 차단
  ```
- **실무 주의사항 및 팁**:
  - 단순 등급 분류는 `max_tokens = 100 ~ 200`, JSON 구조화 응답은 `max_tokens = 300 ~ 400`이 안전합니다. `"length"` 발생 시 프롬프트를 압축하거나 상한선 토큰 값을 높여야 합니다.

### Skill 7: JSON 응답 파싱 및 위험도별 자동 분기 구조
- **파일**: `part01_llm_chatgpt_api/ch02_jsonResponse_parsing.py`
- **핵심**: `response_format = {"type": "json_object"}` 활성화 및 파이썬 `json.loads` 파싱 에러 방어 처리 기법
- **핵심 구현 코드**:
  ```python
  response = client.chat.completions.create(
      model=model,
      messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_MSG}],
      temperature=0.0,
      response_format={"type": "json_object"}
  )
  try:
      result = json.loads(response.choices[0].message.content)
  except json.JSONDecodeError as e:
      result = None
  ```
- **실무 주의사항 및 팁**:
  - json_object 모드를 사용할 때에는 **반드시 System Prompt 내에 JSON으로 응답하라는 명시적 어구가 포함**되어야 에러가 발생하지 않습니다.
  - 파싱 완료 후 `risk_handlers` 딕셔너리 구조를 활용해 정상(로그 저장), 주의(경비 순찰), 위험(경찰 신고 + 비상 알람)에 매핑하는 자동 분기 테이블 패턴을 적용합니다.

### Skill 8: 멀티턴 대화 히스토리의 메모리 깊은 복사(Deep Copy) 제어
- **파일**: `part01_llm_chatgpt_api/ch02_multiTurnChat.py`
- **핵심**: 다회차(Multi-turn) 대화 구현 시 원본 리스트의 참조 훼손을 예방하는 메모리 복제 및 페이로드 연쇄 적재 패턴
- **핵심 구현 코드**:
  ```python
  def chat_with_history(client, history: list, user_message: str):
      updated_history = history.copy()  # [중요] 원본 보호를 위한 얕은 복사
      updated_history.append({"role": "user", "content": user_message})
      # API 호출 및 assistant 응답 추가
      return assistant_reply, updated_history
  ```
- **실무 주의사항 및 팁**:
  - 대화가 반복될수록 토큰 소모량이 누적되므로 장기 대화 파이프라인 설계 시 슬라이싱을 통한 과거 대화 소거 등이 요구됩니다.

### Skill 9: 실시간 API 비용 계산 유틸리티 구축
- **파일**: `part01_llm_chatgpt_api/ch02_dotenv_apicall.py`
- **핵심**: OpenAI Usage 응답 메타데이터를 파이썬 실시간 가격 연산 공식에 대입해 실시간 예산 감시
- **비용 계산 공식 (GPT-4o-mini 기준)**:
  ```python
  usage = response.usage
  input_cost = (usage.prompt_tokens / 1_000_000) * 0.150  # 백만 토큰당 $0.15
  output_cost = (usage.completion_tokens / 1_000_000) * 0.600  # 백만 토큰당 $0.60
  total_cost = input_cost + output_cost
  ```

---

## Part 02 - LangChain

### Skill 10: LangChain이 필요한 이유와 전처리 모듈 구조
- **파일**: `part02_langchain/ch01_whyLangChain.py`
- **핵심**: API 호출 중복, 프롬프트 파편화, 수동 JSON 파싱의 번거로움을 해결하기 위한 LCEL 체이닝의 구조적 당위성
- **전처리 모듈**: OpenCV의 복합 탐지 정보(`dict`)를 정제된 문자열로 구조화하여 랭체인 프롬프트 템플릿의 입구 부분에 공급합니다.

### Skill 11: LangChain LCEL 기본 체인 구성
- **파일**: `part02_langchain/ch01_langchian.py`
- **핵심**: `ChatOpenAI`, `ChatPromptTemplate`, `JsonOutputParser`를 선언적 파이프 연산자(`|`)로 결합
- **핵심 구현 코드**:
  ```python
  analysis_chain = prompt | llm | json_parser
  result = analysis_chain.invoke({"key": "value"})
  ```

### Skill 12: RunnableLambda를 통한 사용자 정의 함수 체이닝
- **파일**: `part02_langchain/ch01-1_runnableLamba.py`
- **핵심**: 일반 파이썬 함수를 랭체인 인터페이스에 부합하도록 래핑하여 파이프 연산자 흐름 내에 중간 가공 부품으로 활용
- **핵심 구현 코드**:
  ```python
  from langchain_core.runnables import RunnableLambda
  chain = RunnableLambda(preprocess_func) | prompt | llm | json_parser
  ```

### Skill 13: OpenCV 감지 데이터의 정밀 전처리 및 픽셀 연산
- **파일**: `part02_langchain/ch02_format_detection.py`
- **핵심**: Bounding Box(bbox) 좌표 정보를 실시간 픽셀 면적 크기(`width x height px`)로 연산 가공하여 텍스트 데이터의 입체성 보강
- **핵심 구현 코드**:
  ```python
  def format_detections(frame_data: dict) -> dict:
      detections = frame_data.get("detections", [])
      lines = []
      for d in detections:
          x1, y1, x2, y2 = d["bbox"]
          width, height = x2 - x1, y2 - y1
          lines.append(f"- {d['class']} ({d['confidence']:.0%}), 크기: {width}x{height}px")
      return {"frame_id": frame_data["frame_id"], "detections_text": "\n".join(lines)}
  ```

### Skill 14: ChatPromptTemplate의 메시지 분리 및 중괄호 이스케이프
- **파일**: `part02_langchain/ch02_prompt_template.py`
- **핵심**: 시스템 역할과 인간 질의 튜플 리스트 분리 및 템플릿 내 JSON 리터럴 중괄호(`{{`, `}}`) 처리 기법
- **핵심 구현 코드**:
  ```python
  prompt = ChatPromptTemplate.from_messages([
      ("system", "보안 전문가로서 아래 서식을 준수하십시오: {{'key': 'value'}}"),
      ("human", "현재 프레임 ID: {frame_id}\n내역: {detections_text}")
  ])
  ```

### Skill 15: JsonOutputParser의 Robustness 확보
- **파일**: `part02_langchain/ch02_output_parser.py`
- **핵심**: LLM이 반환하는 답변 텍스트 내 마크다운 펜스(```json ... ```)를 완벽하게 정제하고 유효한 Python 딕셔너리로 형변환 처리

### Skill 16: 탐지 신뢰도의 비판적 해독 프롬프팅
- **파일**: `part02_langchain/ch02_lcel_pipeline.py`
- **핵심**: "YOLO의 신뢰도는 클래스 가능성 수치일 뿐 절대적 객체 존재를 의미하지 않는다"는 가이드라인을 주입하여 LLM의 신중한 추론성 유도

### Skill 17: 최종 LCEL RAG/분석 파이프라인 통합
- **파일**: `part02_langchain/ch02_lcel_pipeline.py`
- **핵심**: 데이터 가공에서 모델 추론, 파싱까지 유기적으로 이어진 일관성 높은 최종 파이프라인
- **파이프라인 구조**:
  ```python
  analysis_chain = preprocess_lambda | prompt | llm | json_parser
  ```

### Skill 18: 대화 이력 보존을 위한 InMemoryChatMessageHistory
- **파일**: `part02_langchain/ch03_real_memory_chatbot.py`
- **핵심**: RAM(메모리) 상에 대화 객체인 `HumanMessage`와 `AIMessage`를 누적 보관하는 인메모리 관리 기법

### Skill 19: 객체지향형(OOP) 메모리 통합 챗봇 아키텍처
- **파일**: `part02_langchain/ch03_real_memory_chatbot.py`
- **핵심**: 메모리 관리, 프롬프트 조립, LLM 실행 및 수동 마크다운 펜스 파싱을 캡슐화한 종합 클래스 아키텍처
- **핵심 구현 코드**:
  ```python
  class CCTVOperatorChatbot:
      def __init__(self):
          self.history = InMemoryChatMessageHistory()
          self.frame_cache = {}
      
      def _build_messages(self, user_input: str) -> list:
          return [SystemMessage(content=SYSTEM_PROMPT), *self.history.messages, HumanMessage(content=user_input)]
      
      def analyze_frame(self, frame_id: int, detections: list, timestamp: str, location: str) -> dict:
          # 데이터 전처리 -> LLM 호출 -> history.add_user_message / add_ai_message -> 캐싱 및 JSON 반환
  ```

---

## Part 02 - LangChain (심화)

### Skill 20: 메모리 유무에 따른 지칭어 해독력 차이 실증
- **파일**: `part02_langchain/ch03_memoryCompare.py`
- **핵심**: "그거 왜 주의 등급이야?"와 같은 생략어/대명사가 포함된 질문에 대해 메모리가 확보되어 있을 때에만 정확한 분석이 가능하다는 차이를 시뮬레이션으로 규명

### Skill 21: SimpleBufferMemory 원리적 수동 구현
- **파일**: `part02_langchain/ch03_simple_buffer_memory.py`
- **핵심**: 랭체인의 `ConversationBufferMemory` 동작을 모방하여 `format_as_text()` 메서드를 통해 원문을 대화 이력 텍스트로 복합 변환하는 유틸리티 클래스 제작

### Skill 22: SimpleSummaryMemory 원리적 수동 구현
- **파일**: `part02_langchain/ch03_simple_summary_memory.py`
- **핵심**: 컨텍스트 누적으로 인한 토큰 팽창을 억제하기 위해 오래된 대화는 한 줄 요약으로 압축하고 최근 대화만 원문으로 유지하는 압축형 메모리 아키텍처 구현

---

## Part 02 - LangChain (Agent & Tools)

### Skill 23: Mock LLM 기법을 활용한 배치 파이프라인 시뮬레이션
- **파일**: `part02_langchain/ch04_langChain_pipeline.py`
- **핵심**: 인터넷 연결 및 API 키 잔액 유무와 관계없이 다중 프레임 연산을 원활하게 시뮬레이션하기 위해 `mock_llm_fn`을 파이프라인에 주입해 테스트 비용 절감

### Skill 24: @tool 데코레이터를 이용한 Agent 도구 메타데이터 등록
- **파일**: `part02_langchain/ch04_tool_decorator.py`
- **핵심**: LLM이 작업 수행 중 직접 상황을 판단해 호출할 수 있도록 함수의 docstring과 자료형 어노테이션 기반 툴 등록 기법 적용
- **핵심 구현 코드**:
  ```python
  from langchain_core.tools import tool
  @tool
  def filter_danger_frames(frames_json: str) -> str:
      """분석 결과 리스트에서 위험 프레임만 필터링합니다."""
      # 구현 코드
  ```
- **실무 주의사항 및 팁**:
  - 도구의 docstring 첫 번째 줄과 Args 타입 설명은 LLM이 도구를 올바르게 찾아 쓰기 위한 **라벨 메타데이터**로 활용되므로 정교하게 영문/국문 기술이 되어야 합니다.

### Skill 25: ReAct 추론 엔진 루프 수동 시뮬레이션
- **파일**: `part02_langchain/ch04_react_simulate.py`
- **핵심**: Thought -> Action -> Observation -> Final Answer로 이어지는 자율적 추론 단계를 `TOOL_REGISTRY` 매핑 테이블을 구현하여 완벽히 모의 수행하는 아키텍처

---

## Part 03 - RAG & VectorDB

### Skill 26: OpenAI Embedding 및 코사인 유사도 수학적 직접 구현
- **파일**: `part03_rag_vectordb/ch02_01_cosine_similarity.py`
- **핵심**: RAG 원천 기술의 수학적 원리를 해독하기 위해 넘파이(`numpy`) 벡터 점곱 및 노름 공식을 사용한 유사도 검색 모듈 제작
- **핵심 구현 코드**:
  ```python
  import numpy as np
  def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
      dot = np.dot(a, b)
      norm = np.linalg.norm(a) * np.linalg.norm(b)
      return dot / norm if norm != 0 else 0.0
  ```

### Skill 27: 디스크 기반 영속적 ChromaDB 적재 및 CRUD
- **파일**: `part03_rag_vectordb/ch02_02_chromadb_search.py`
- **핵심**: 메모리 증발 방지를 위해 영속성 디바이스를 구축하고 한글 콘솔 환경을 고려해 이모지가 배제된 정제 텍스트 기호 기반 데이터 입출력 패턴 적용

### Skill 28: ChromaDB의 인덱스 불일치(Index Inconsistency) 예방 및 고급 운용
- **파일**: `part03_rag_vectordb/ch02_03_chromadb_crud.py`
- **핵심**: PersistentClient를 통한 영속적 갱신, 코사인 측정 고정(`hnsw:space="cosine"`), delete_collection 초기화, 그리고 복합 `$and` 논리 필터 삭제 기능 운용
- **핵심 구현 코드**:
  ```python
  chroma_client = chromadb.PersistentClient(path="./chroma_db")
  collection = chroma_client.get_or_create_collection(
      name="logs", metadata={"hnsw:space": "cosine"}
  )
  # [치명적 주의]: 문서가 바뀌면 벡터도 같이 재생성해 넣어야 합니다.
  collection.update(
      ids=["log_001"],
      embeddings=[get_embedding(new_text)],
      documents=[new_text],
      metadatas=[{"risk_level": "위험", "resolved": True}]
  )
  # 복합 논리 필터 삭제
  collection.delete(where={"$and": [{"risk_level": "위험"}, {"location": "공장 외곽"}]})
  ```
- **실무 주의사항 및 팁**:
  - `collection.update()` 수행 시 텍스트 내용(`documents`)만 수정하고 임베딩 벡터(`embeddings`)를 누락하면, 데이터베이스 인덱스 상에는 구 텍스트의 벡터가 남게 되는 **인덱스 불일치**가 발생합니다. 이 경우 새로이 수정된 내용 기반의 의미 유사도 조회가 완벽히 차단됩니다.

### Skill 29: CSVLoader와 TextSplitter를 결합한 LCEL RAG 파이프라인
- **파일**: `part03_rag_vectordb/ch03_01_rag_pipeline.py`
- **핵심**: 메타데이터 출처를 추적하는 Loader와 대형 매뉴얼 문서를 분할하는 Splitter, 그리고 ChromaDB 검색 결과를 context 변수로 주입해 답변을 유도하는 종합 RAG 아키텍처
- **핵심 구현 코드**:
  ```python
  loader = CSVLoader(file_path="logs.csv", encoding="utf-8", source_column="timestamp")
  splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt | llm | StrOutputParser()
  )
  ```

### Skill 30: 외부 인프라스트럭처 연동을 위한 MCP(Model Context Protocol) 접목 아키텍처
- **개념**: Notion, Slack 등 비즈니스 애플리케이션의 API 단계를 RAG와 연동해 실시간 데이터 공급(Input) 및 자동 위험 상황 경보 발송(Output)을 지원하는 종합 연동 설계 개념

### Skill 31: MultiQueryRetriever - 다중 질의 파생을 통한 검색 재현율(Recall) 극대화
- **파일**: `part03_rag_vectordb/ch03_02_multi_query_retriever.py`
- **핵심**: 자연어 질의를 LLM을 활용해 다각도의 대체 질문으로 파생시키고, 병렬 검색을 통해 키워드 불일치에 따른 정보 누락 방지
- **핵심 구현 코드**:
  ```python
  import logging
  from langchain_classic.retrievers.multi_query import MultiQueryRetriever
  
  # INFO 레벨 활성화 시 생성된 다중 질의가 실시간으로 터미널 콘솔에 기록됩니다.
  logging.basicConfig()
  logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
  
  multi_query_retriever = MultiQueryRetriever.from_llm(
      retriever=base_retriever, llm=llm
  )
  ```
- **실무 주의사항 및 팁**:
  - 최신 랭체인 경량화 아키텍처에서는 기존 `langchain.retrievers`가 아닌 클래식 래퍼 모듈 경로인 `langchain_classic.retrievers.multi_query.MultiQueryRetriever`에서 임포트해야 구동 에러를 예방할 수 있습니다.
  - 실행 경로 독립성 확보를 위해 `os.path.dirname(os.path.abspath(__file__))` 기법으로 `.env` 및 `detection_logs.csv` 경로를 동적 매핑하여 실행 경로 이식성을 보강합니다.

---

## 알려진 이슈 및 대응 방안

| 항목 | 원인 및 해결방안 |
|------|-----------------|
| **환경변수 키 통일** | 소스 코드 전반에서 OpenAI 연동 키를 `OPENAI_API_KEY` 환경변수명 하나로 일관성 있게 통합 관리합니다. |
| **윈도우 터미널 인코딩** | Windows 기본 인코딩(CP949) 터미널 출력 환경에서 이모지 포함 문구 실행 시 `UnicodeEncodeError` 유발 현상이 발생합니다. |
| **이모지 전면 제거 정책** | 시스템 인코딩 충돌을 예방하고 프로젝트 스타일의 엄격한 가독성 확보를 위해, **모든 소스 코드 및 출력용 텍스트, 문서 마크다운에서 이모지를 전면 차단하고 대괄호 기호로 통일**합니다. |
| **파일 스트림 UTF-8 명시** | Windows 로컬 환경에서 텍스트 입출력 시 시스템 인코딩 오류가 발생하지 않도록 `open(..., encoding="utf-8")`을 코드 작성 시 필수로 선언합니다. |

---

## 코드 스타일 및 프로젝트 규칙
- 모든 주석, 출력 로그 메시지, 랭체인 최종 아웃풋, 그리고 대외용 문서(Artifact)는 **한국어**로만 작성합니다.
- 파일명 정의: 소문자 시작, 언더바 결합 및 목적 구체화 `ch{번호}_{영문설명}.py` 컨벤션을 따릅니다.
- **이모지 사용 절대 엄금**: 텍스트 가독성은 일반 대괄호 기호(`[TIP]`, `[WARNING]`, `[OK]`, `[ERROR]`) 등을 활용합니다.
- 교육적 설명 주석 및 자료형 어노테이션(`Type Hinting`)의 적극적 작성을 장려합니다.
- 결정적 구조 파싱이 필요한 JSON 파이프라인의 `temperature` 값은 `0.0`으로 고정하며, 그 외 시나리오도 `0.3` 미만으로 제어합니다.

---

## Part 04 - 멀티모달 (Multimodal)

### Skill 32: 순수 Python 기반 WAV 오디오 합성 및 16-bit PCM 바이너리 패킹
- **파일**: `part04_multimodal/ch03_01_whisper.py`
- **핵심**: 별도의 외부 오디오 라이브러리 없이 파이썬 내장 `wave`와 `struct` 모듈을 조합하여, Whisper API 권장 규격(16000Hz 주파수, 1채널 모노, 16-bit PCM 포맷)에 부합하는 물리적 가상 WAV 파일을 수학적 합성 기술로 제작합니다.
- **핵심 구현 코드 (스페이스 2칸 컨벤션 준수)**:
  ```python
  import wave
  import struct
  import math

  with wave.open(path, "w") as wf:
    wf.setnchannels(1)          # 모노 채널 (Whisper 권장)
    wf.setsampwidth(2)          # 16-bit PCM (2바이트 폭)
    wf.setframerate(sample_rate) # 샘플레이트 지정 (Whisper 권장 16000Hz)
    for i in range(n_samples):
      t = i / sample_rate
      val = 0.4 * math.sin(2 * math.pi * 200 * t)  # 오디오 파형 합성
      sample = int(val * 32767 * 0.8)
      wf.writeframes(struct.pack("<h", sample))    # 리틀엔디안 16비트 정수 패킹
  ```
- **자료형 및 파라미터 (Data Types & Params)**:
  - 입력: `path: str` (저장할 파일의 절대/상대 경로), `duration_sec: float` (음향 파일 재생 시간), `sample_rate: int` (주파수 헤르츠 수치)
  - 출력: 물리 디스크 상에 즉시 생성되는 바이너리 `.wav` 파일
  - 핵심 기법: `struct.pack("<h", sample)` (부동소수점 오디오 신호 값을 바이너리 16비트 signed short 형식으로 전환)
- **실무 주의사항 및 팁 (Warnings & Tips)**:
  - Whisper API를 활용해 음성 인식을 진행할 때 오디오 용량을 낭비하지 않도록 불필요한 스테레오 다중 채널을 지양하고 **1채널 모노 및 16000Hz 규격**으로 고정 가공해야 오버헤드를 막을 수 있습니다.

### Skill 33: winget 무인(Silent) 패키지 설치를 통한 멀티미디어 분석 인프라 구축
- **핵심**: 윈도우 패키지 관리자(`winget`)를 터미널 상에서 원격 제어하여 오디오 분석 및 파형 가공을 지원하는 Audacity 편집기를 확인 메시지(대화 상자) 대기 현상 없이 완벽 무인으로 초고속 자동 설치합니다.
- **핵심 구현 코드**:
  ```powershell
  winget install Audacity.Audacity --silent --accept-source-agreements --accept-package-agreements
  ```
- **실무 주의사항 및 팁 (Warnings & Tips)**:
  - 백그라운드 자동화 배치 스크립트나 CI/CD 파이프라인 상에서 사용자와의 시각적 대화 창이 생성되어 실행이 멈추는 행(Hang) 결함을 예방하기 위해, `--silent` 플래그 및 소스/패키지 라이선스 강제 서명 플래그를 필수로 함께 전달해야 합니다.

