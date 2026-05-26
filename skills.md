# AL-CCTV Platform — Skills 정리

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | AI CCTV 보안 분석 플랫폼 |
| **핵심 아키텍처** | OpenCV 1차 필터링 → 이상 프레임만 LLM 전달 → 위험도 분석 |
| **Python** | 3.14 (venv 가상환경) |
| **가상환경 활성화** | PowerShell: `.\venv\Scripts\Activate.ps1` <br> CMD: `.\venv\Scripts\activate.bat` |
| **LLM 모델** | GPT-4o-mini |
| **주요 패키지** | openai 2.36.0, python-dotenv 1.2.2, langchain 1.3.0, langchain-openai 1.2.1, langchain-community 0.4.1, chromadb, numpy |

---

## Part 01 — LLM 기초 & ChatGPT API

### Skill 1: LLM 개념 계층 이해
- **파일**: `part01_llm_chatgpt_api/ch01_llm개념이해.md`
- **핵심**: AI → ML → DL → LLM → ChatGPT 포함 관계
- **토큰**: API 비용의 기본 단위 (영어: 단어 일부, 한국어: 형태소 단위)
- **LLM 한계 4가지**:
  1. 지식 컷오프 → RAG로 해결 (Part 03)
  2. 환각(Hallucination) → 프롬프트 명시 + RAG (Part 03)
  3. 비용 → OpenCV 1차 필터링으로 해결
  4. 컨텍스트 길이 → LangChain 배치 처리 (Part 02)

### Skill 2: 환경 설정 (.env + dotenv)
- **파일**: `part01_llm_chatgpt_api/ch02_dotenv_apicall.py`
- **패턴**:
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPEN_AI_MODEL")  # gpt-4o-mini
```
- **보안**: API 키는 `.env`에 저장, `.gitignore`에 `.env` 추가 필수

### Skill 3: messages 구조 (3가지 역할)
- **파일**: `part01_llm_chatgpt_api/ch02_message_struct.py`
- **역할**:
  - `system` — AI의 정체성(페르소나) 설정, 대화 내내 유지
  - `user` — 사람이 AI에게 보내는 메시지
  - `assistant` — AI가 이전에 답변한 내용 기록 (멀티턴 대화용)
- **중요**: API는 '기억'이 없으므로 매번 전체 messages 리스트를 전달해야 함

### Skill 4: System Prompt 비교 실험
- **파일**: `part01_llm_chatgpt_api/ch02_system_prompt_comparison.py`
- **결론**:
  - ① 설정 없음 → 영어 답변, 형식 없음
  - ② 일반 어시스턴트 → 불규칙 답변
  - ③ CCTV 전문가 설정 → 한국어, 구조화된 분석, 실무 활용 가능
- **이 프로젝트 표준 System Prompt**:
```
당신은 AI CCTV 보안 분석 시스템입니다.
OpenCV로 탐지된 객체 정보를 입력받아 위험도를 분석합니다.
답변 형식: 위험도(정상/주의/위험) + 판단 근거 + 권고 조치.
한국어로만 답합니다.
```

### Skill 5: temperature 파라미터
- **파일**: `part01_llm_chatgpt_api/ch02_temperature_comparison.py`
- **범위**: 0.0 (결정적) ~ 2.0 (무작위), 실무에서는 0.0~1.0 범위만 사용
- **권장**: CCTV 위험도 분석에는 **temperature = 0.0 ~ 0.3**
- **이유**: 위험 판단이 매번 달라지면 신뢰할 수 없는 시스템이 됨

### Skill 6: max_tokens 파라미터
- **파일**: `part01_llm_chatgpt_api/ch02_maxtoken_comparison.py`
- **권장 설정**:
  - 단순 위험도 판단 → `max_tokens = 100~200`
  - 상세 분석 리포트 → `max_tokens = 300~500`
  - JSON 구조화 출력 → `max_tokens = 300~400`
- **`finish_reason`**: `"stop"` (정상완료) / `"length"` (잘림 → 토큰 늘려야 함)

### Skill 7: JSON 응답 파싱 + 위험도 자동 분기
- **파일**: `part01_llm_chatgpt_api/ch02_jsonResponse_parsing.py`
- **핵심 설정**: `response_format = {"type": "json_object"}`
- **필수 조건**: system prompt에 "JSON으로 답해"라고 반드시 명시
- **JSON 응답 스키마**:
```json
{
  "timestamp":    "탐지 시각",
  "location":     "탐지 위치",
  "person_count": 0,
  "risk_level":   "정상 | 주의 | 위험",
  "reason":       "판단 근거",
  "action":       "권고 조치"
}
```
- **위험도 자동 분기**:
  - 정상 → 로그 저장만
  - 주의 → 경비팀 알림 전송
  - 위험 → 경찰 즉시 신고 + 비상 알람

### Skill 8: 멀티턴 대화 히스토리 관리
- **파일**: `part01_llm_chatgpt_api/ch02_multiTurnChat.py`
- **패턴**: `chat_with_history(client, history, user_message)` 함수
  1. `history.copy()`로 원본 보호
  2. user 메시지 추가 → 전체 히스토리를 API에 전달
  3. assistant 답변을 히스토리에 추가하여 반환
- **주의**: 대화가 길어질수록 토큰 수 증가 → Part 02에서 LangChain이 자동 관리

### Skill 9: API 비용 계산
- **GPT-4o 기준**:
  - 입력: $2.50 / 1M 토큰
  - 출력: $10.00 / 1M 토큰
```python
input_cost  = (usage.prompt_tokens    / 1_000_000) * 2.50
output_cost = (usage.completion_tokens / 1_000_000) * 10.00
total_cost  = input_cost + output_cost
```
- **CCTV 비용 추정** (1분마다 1장, 프레임당 ~150토큰):
  - 하루: $0.54 / 한달: $16.20

---

## Part 02 — LangChain

### Skill 10: LangChain이 필요한 이유
- **파일**: `part02_langchain/ch01_whyLangChain.py`
- **LangChain 없이의 문제점** (Part 01 방식):
  1. 매번 프롬프트를 직접 조립 → 코드 중복
  2. 매번 API를 직접 호출 → 반복 코드
  3. 매번 응답을 수동 파싱 → 반복 코드
- **프레임 데이터 텍스트 변환 유틸리티 함수**: 탐지 결과를 텍스트로 변환하는 기초적인 형태

### Skill 11: LangChain LCEL 기본 체인 구성
- **파일**: `part02_langchain/ch01_langchian.py`
- **핵심 클래스**:
  - `ChatOpenAI`: OpenAI API를 랭체인 방식으로 호출하는 모델 클래스
  - `ChatPromptTemplate`: 메시지 템플릿 관리 (`{variable}` 사용)
  - `JsonOutputParser`: LLM의 JSON 응답을 파이썬 딕셔너리로 자동 변환
- **LCEL (LangChain Expression Language)**:
  - 파이프 연산자(|)를 사용하여 데이터 흐름 연결
  - 패턴: `analysis_chain = prompt | llm | json_parser`
- **실행**: `chain.invoke({"key": "value"})`

### Skill 12: RunnableLambda - 사용자 정의 함수 체이닝
- **파일**: `part02_langchain/ch01-1_runnableLamba.py`
- **핵심 클래스**: `RunnableLambda`
- **기능**:
  - 일반 파이썬 함수를 LangChain 체인에서 사용할 수 있는 객체로 변환
  - 딕셔너리 형태의 복합 입력 처리 가능
  - 파이프 연산자(|)를 이용해 여러 함수를 순차적으로 연결
- **코드 패턴**:
```python
chain = RunnableLambda(func1) | RunnableLambda(func2)
result = chain.invoke(input_data)
```

### Skill 13: ChatPromptTemplate 입력 최적화를 위한 데이터 변환
- **파일**: `part02_langchain/ch02_format_detection.py`
- **기능 및 구조**:
  - OpenCV 탐지 결과(dict)를 ChatPromptTemplate의 개별 입력 변수인 `frame_id`, `timestamp`, `location`, `detections_text`에 정확히 맞춘 구조화된 딕셔너리로 전처리합니다.
  - bbox의 가상 좌표를 가로/세로 크기로 계산(`width = x2 - x1`, `height = y2 - y1`)하여 실시간 픽셀 면적 크기(`크기: WxHpx`)를 출력하는 정교함을 갖추었습니다.
  - 전처리 완료된 헬퍼 함수를 `RunnableLambda`로 감싸 파이프라인의 입구 부품(`formatter`)으로 쉽게 패키징합니다.
- **코드 패턴**:
```python
def format_detections(frame_data: dict) -> dict:
    # 탐지 정보를 파싱하여 가로x세로 크기 및 좌표를 문자열화하고 변수 딕셔너리 반환
    ...
    return {"frame_id": ..., "timestamp": ..., "location": ..., "detections_text": ...}

formatter = RunnableLambda(format_detections)
```

### Skill 14: ChatPromptTemplate - System/Human 메시지 분리
- **파일**: `part02_langchain/ch02_prompt_template.py`
- **핵심 메서드**: `ChatPromptTemplate.from_messages()`
- **기능**:
  - `system` 역할과 `human` 역할을 명확히 분리하여 정의
  - 튜플 리스트 형태 `[("role", "content"), ...]` 사용
  - **이스케이프**: 템플릿 내에서 실제 중괄호를 출력하려면 `{{`, `}}` 처럼 두 번 사용 (리터럴 중괄호)
- **코드 패턴**:
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신의 역할은..."),
    ("human", "분석 요청: {input_var}")
])
```

### Skill 15: JsonOutputParser - 유연한 JSON 파싱 (Robustness)
- **파일**: `part02_langchain/ch02_output_parser.py`
- **핵심 클래스**: `JsonOutputParser`
- **기능**:
  - LLM이 반환한 텍스트에서 JSON 부분만 추출하여 파이썬 딕셔너리로 변환
  - **강점**: LLM이 JSON을 마크다운 코드 블록(```json ... ```)으로 감싸서 응답해도 에러 없이 자동으로 파싱함
- **코드 패턴**:
```python
json_parser = JsonOutputParser()
result = json_parser.parse(llm_markdown_response)
```

### Skill 16: 탐지 신뢰도(Confidence)의 비판적 해석 유도
- **파일**: `part02_langchain/ch02_lcel_pipeline.py`
- **프롬프트 기법**:
  - LLM에게 "YOLO의 신뢰도는 클래스 확률일 뿐, 객체의 실제 존재 여부에 대한 절대적 신뢰를 의미하지 않는다"는 점을 명시적으로 주입
  - 이를 통해 LLM이 신뢰도가 낮은 객체에 대해 "불확실성"을 인지하고 보수적인(신중한) 분석 결과를 내놓도록 유도

### Skill 17: 최종 랭체인 파이프라인 통합 (End-to-End Pipeline)
- **파일**: `part02_langchain/ch02_lcel_pipeline.py`
- **핵심**: 데이터 전처리부터 모델 호출, 파싱까지 선언적 구조로 연결
- **파이프라인 구조**:
```python
analysis_chain = (
    formatter      # 1. OpenCV 데이터를 프롬프트용 딕셔너리로 전처리 (RunnableLambda)
    | cctv_prompt  # 2. 전처리된 데이터를 시스템/휴먼 메시지 템플릿에 주입
    | llm          # 3. GPT-4o-mini 모델 호출 (temperature=0.0)
    | json_parser  # 4. LLM의 텍스트 응답을 JSON(dict)으로 파싱
)
```

### Skill 18: 대화 이력 관리 (InMemoryChatMessageHistory)
- **파일**: `part02_langchain/ch03_real_memory_chatbot.py`
- **핵심 클래스**: `InMemoryChatMessageHistory`
- **기능**:
  - LLM과의 대화 이력(`HumanMessage`, `AIMessage`)을 RAM(메모리)에 저장
  - 대화가 반복될 때마다 이전 메시지들을 포함하여 LLM에 전달함으로써 '맥락(Context)'을 유지

### Skill 19: 메모리 통합 챗봇 설계 (Stateful Chatbot)
- **파일**: `part02_langchain/ch03_real_memory_chatbot.py`
- **핵심**: 객체 지향 프로그래밍(OOP)을 통해 메모리와 분석 로직을 캡슐화
- **효과**: "3번 프레임 왜 위험이야?"와 같이 생략된 지칭어(대명사)가 포함된 질문에도 정확히 답변 가능
- **CCTV 탐지 데이터 구조 (OpenCV 출력 기준)**:
```python
{
    "frame_id": 1,
    "timestamp": "02:13",
    "location": "주차장 A구역",
    "detections": [
        {"class": "person", "bbox": [120, 80, 200, 350], "confidence": 0.91},
        {"class": "car",    "bbox": [50, 200, 280, 400],  "confidence": 0.95}
    ]
}
```

---

## Part 02 — LangChain (심화)

### Skill 20: 메모리 비교 실험 (기억 없는 LLM vs. 기억 있는 LLM)
- **파일**: `part02_langchain/ch03_memoryCompare.py`
- **목적**: 메모리 필요성을 체감하기 위한 시뮬레이션
- **케이스 A — 기억 없는 LLM**: 매번 독립적인 단일 메시지만 전달하여 후속 질문에 맥락 없이 답변 불가
- **케이스 B — 기억 있는 LLM**: `chat_history` 리스트에 이전 대화를 직접 누적 전달하여 맥락 유지

### Skill 21: SimpleBufferMemory - 버퍼 메모리 구현 원리
- **파일**: `part02_langchain/ch03_simple_buffer_memory.py`
- **목적**: `ConversationBufferMemory`의 내부 동작을 이해하기 위한 교육용 클래스
- **핵심 메서드**:
  - `add_user_message(text)` / `add_ai_message(text)` — 역할별 메시지 추가
  - `get_all_messages()` — 전체 이력 반환
  - `format_as_text()` — 프롬프트 삽입용 텍스트 변환 (`Human: ... / AI: ...`)
- **한계**: 대화가 길어질수록 토큰이 계속 증가

### Skill 22: SimpleSummaryMemory - 요약 메모리 구현 원리
- **파일**: `part02_langchain/ch03_simple_summary_memory.py`
- **목적**: `ConversationSummaryMemory`의 압축 원리를 이해하기 위한 교육용 클래스
- **동작 방식**:
  1. 최근 `max_recent`개까지는 원문 유지
  2. 초과 시 가장 오래된 대화를 한 줄 요약으로 압축
  3. LLM에는 `[이전 대화 요약] + [최근 대화 원문]`을 함께 전달

---

## Part 02 — LangChain (Agent & Tools)

### Skill 23: LCEL 배치 파이프라인 - 다수 프레임 일괄 분석
- **파일**: `part02_langchain/ch04_langChain_pipeline.py`
- **목적**: CH02 LCEL 파이프라인을 복습하고 다수 프레임을 루프로 일괄 처리
- **Mock LLM 패턴**: 실제 `ChatOpenAI` 대신 `mock_llm_fn` 함수를 `RunnableLambda`로 감싸서 오프라인 상태에서도 효율적으로 체인을 시뮬레이션하는 기법

### Skill 24: @tool 데코레이터 - LangChain Tool 등록
- **파일**: `part02_langchain/ch04_tool_decorator.py`
- **핵심**: `@tool` 데코레이터로 일반 파이썬 함수를 LLM이 호출 가능한 Tool로 등록
- **중요 규칙**: Tool 함수의 docstring이 LLM이 도구를 선택하는 기준이 되므로 상세히 기술
- **등록된 Tool 3종**:
  - `filter_danger_frames`: 위험 프레임만 필터링
  - `count_objects_in_zone`: 특정 구역 탐지 수 카운트
  - `get_risk_summary`: 전체 위험도 요약 통계
- **Tool 메타데이터 확인**: `tool.name`, `tool.description`, `tool.args`

### Skill 25: ReAct 패턴 - Thought → Action → Observation 루프
- **파일**: `part02_langchain/ch04_react_simulate.py`
- **개념**: LLM이 질문을 분석하고 Tool을 스스로 선택·호출·결과 확인 후 최종 답변을 생성하는 추론 패턴
- **`TOOL_REGISTRY` 패턴**: Tool 이름(str) → Tool 객체 매핑 딕셔너리를 직접 정의하여 호출
```python
TOOL_REGISTRY = {
    "filter_danger_frames":  filter_danger_frames,
    "count_objects_in_zone": count_objects_in_zone,
    "get_risk_summary":      get_risk_summary,
}
observation = TOOL_REGISTRY[action_name].invoke(action_input)
```

---

## Part 03 — RAG & VectorDB

### Skill 26: OpenAI Embedding + 코사인 유사도 직접 구현
- **파일**: `part03_rag_vectordb/ch02_01_cosine_similarity.py`
- **목적**: RAG의 핵심 원리인 "의미 기반 유사도 검색"을 직접 구현하여 이해
- **임베딩 모델**: `text-embedding-3-small` (1536차원)
- **코사인 유사도 공식**: `cos(θ) = (A · B) / (|A| × |B|)`
- **핵심 함수**:
```python
def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip()
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm != 0 else 0.0
```

### Skill 27: ChromaDB - 벡터 데이터베이스 구축
- **파일**: `part03_rag_vectordb/ch02_02_chromadb_search.py`
- **핵심**:
  - `collection.add(...)`를 통한 탐지 로그 데이터(ID, 문서 텍스트, 메타데이터) 등록
  - `collection.query(...)`를 사용해 현재 상황 쿼리와 유사도 거리가 가장 가까운 상위 N개 로그 추출
  - Windows 콘솔 인코딩(`cp949`) 호환을 위해 출력 이모지 제거 및 텍스트 기호 적용

### Skill 28: ChromaDB 영속성 관리 및 CRUD 운용법
- **파일**: `part03_rag_vectordb/ch02_03_chromadb_crud.py`
- **핵심 운용 기법**:
  - **영속적 클라이언트 구축**: `chromadb.PersistentClient(path="./chroma_db")`를 사용해 메모리가 아닌 디스크에 벡터 데이터를 영구 저장합니다.
  - **유사도 공간 설정**: `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`을 통해 임베딩 크기보다 방향을 중시하는 코사인 유사도 연산으로 거리 검색 방식을 고정합니다.
  - **문서 수정 시 필수 주의사항**: 
    > [!IMPORTANT]
    > `collection.update()`를 통해 문서 내용(`documents`)을 변경하는 경우, 반드시 새로운 텍스트의 벡터(`embeddings`)도 함께 생성해 전달해야 합니다. 그렇지 않으면 데이터베이스 내부 인덱스에 구 버전 텍스트의 벡터가 남아 있어, 갱신된 텍스트 내용으로 검색이 되지 않는 치명적인 인덱스 불일치가 발생합니다.
  - **정밀 삭제 제어**:
    - `collection.delete(ids=[...])` (ID 지정 개별 삭제)
    - `collection.delete(where={"resolved": True})` (단일 조건 필터링 삭제)
    - `collection.delete(where={"$and": [{"risk_level": "위험"}, {"location": "공장"}]})` (복합 조건 논리 삭제)
  - **컬렉션 초기화**: 데이터만 지우고 컬렉션 설정을 살리는 방식(`delete(ids=all_ids)`)과 클라이언트 수준에서 완전히 파괴 후 재생성하는 방식(`delete_collection`)의 구분 적용.

### Skill 29: LangChain LCEL RAG 파이프라인 구축 및 Retriever 구성
- **파일**: `part03_rag_vectordb/ch03_01_rag_pipeline.py`
- **구축 절차 및 구성 요소**:
  - **CSVLoader 연동**: `CSVLoader(file_path, encoding="utf-8", source_column="timestamp")`를 통해 구조화된 로우 데이터를 개별 `Document` 객체로 자동 파싱합니다.
  - **CharacterTextSplitter 분할**: 장문의 텍스트 및 보안 매뉴얼을 처리하기 위해 청크 크기(`chunk_size=500`), 오버랩(`chunk_overlap=50`), 줄바꿈 구분자(`separator="\n"`)를 적용하여 텍스트 분할 파이프라인을 구축합니다.
  - **Chroma 자동 인덱싱**: `OpenAIEmbeddings(model="text-embedding-3-small")`와 `Chroma.from_documents()`를 연동하여 청크 텍스트 벡터화 및 ChromaDB 적재 작업을 한 줄의 코드로 통합 처리합니다.
  - **Retriever 변환**: 적재된 데이터스토어를 `vectorstore.as_retriever(search_kwargs={"k": 3})`를 호출하여 의미 유사도가 가장 높은 상위 3개의 맥락 문서를 실시간 질의할 수 있는 정보 검색 모듈로 가공합니다.

### Skill 30: RAG 파이프라인과 MCP(Model Context Protocol) 접목 아키텍처
- **개념**: LangChain RAG 파이프라인의 데이터 수집(Input) 및 최종 결과 조치(Output/Action) 단계를 외부 도구들과 유기적으로 연결합니다.
- **접목 포인트**:
  1. **입력단 (Data Ingestion)**: Notion MCP, Slack MCP 등을 사용하여 실시간 원시 데이터를 가져와 Document로 변환합니다.
  2. **출력단 (Action Execution)**: 최종 분석 리포트가 생성되면, Notion MCP를 통해 페이지에 문서를 자동 기록하거나, 위험 판정 시 Slack MCP를 통해 경보 메시지를 즉시 발송하는 조치를 취합니다.
- **의의**: LLM이 데이터를 조회하는 통로이자, 분석 결과에 따라 비즈니스 도구들을 직접 실행하게 만드는 자동화된 '손발' 역할을 수행합니다.

### Skill 31: MultiQueryRetriever - 다중 질의를 통한 검색 재현율(Recall) 극대화
- **파일**: `part03_rag_vectordb/ch03_02_multi_query_retriever.py`
- **핵심**: 사용자의 모호한 자연어 질문에 대해 LLM이 의미상 동일한 다각도의 대체 질문들을 생성하고, 각 질문으로 검색을 병렬 수행한 뒤 그 결과 문서들의 고유한 합집합을 추출합니다.
- **주요 기법**:
  1. **임포트 경로 주의**: 최신 랭체인 아키텍처에 맞추어 `langchain_classic.retrievers.multi_query.MultiQueryRetriever` 경로에서 임포트합니다.
  2. **동적 대체 질문 확인 (로깅)**: 아래 설정을 통해 LLM이 생성한 대체 질문들을 콘솔 터미널에 실시간으로 출력합니다.
     ```python
     import logging
     logging.basicConfig()
     logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
     ```
  3. **검색기 구성**:
     ```python
     multi_query_retriever = MultiQueryRetriever.from_llm(
         retriever=base_retriever,
         llm=llm
     )
     ```
  4. **강건한 파일 경로 관리**: 실행 디렉토리에 영향받지 않고 일관되게 동작하도록 `os.path.abspath(__file__)` 기준의 절대 경로 탐색 기법을 적용합니다.
- **의의**: 단어가 정확히 일치하지 않아도 의미적으로 연관된 과거 사례들을 풍부하게 수집하여 RAG의 한계인 검색 누락(False Negative)을 방지합니다.

---

## 알려진 이슈

| 항목 | 원인 및 해결방안 |
|------|-----------------|
| **환경변수 키 통일** | 모든 파일에서 `OPENAI_API_KEY`를 사용하도록 완전히 통일됨 |
| **윈도우 터미널 인코딩** | 윈도우 기본 터미널(CP949)에서 이모지 출력 시 `UnicodeEncodeError` 발생 |
| **이모지 제거 정책** | 인코딩 오류 예방을 위해 모든 코드 및 출력 메시지에서 이모지를 제거하고 텍스트 기호로 대체함 |
| **파일 스트림 인코딩** | 윈도우 환경에서 파일 입출력 시 인코딩 오류가 발생하지 않도록 `open(..., encoding="utf-8")`을 명시해야 함 |

---

## 코드 스타일 컨벤션
- 모든 주석·출력·**문서(Artifact)** 및 **AI 답변**은 **한국어**로 작성
- 파일명: `ch{번호}_{영문설명}.py` 형식
- **이모지 사용 절대 금지**: 터미널 출력 오류(`UnicodeEncodeError`) 방지 및 프로젝트 통일성 확보
- 교육적 설명 주석을 상세히 포함
- `temperature = 0.0` (JSON 분석) / `0.3` (일반 분석)
- **프롬프트 내 지시**: 모델에게 항상 "한국어로 답변할 것"을 명시적으로 지시
- **실행 환경**: 반드시 가상환경(`venv`)을 활성화하거나 가상환경 내 파이썬 인터프리터 경로를 사용하여 실행합니다.
  - 활성화 방법:
    - PowerShell: `.\venv\Scripts\Activate.ps1`
    - CMD: `.\venv\Scripts\activate.bat`
  - 직접 실행: `& [경로]/venv/Scripts/python.exe [파일]` (PowerShell)
