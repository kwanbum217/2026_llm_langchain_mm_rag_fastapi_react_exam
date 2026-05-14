# AL-CCTV Platform — Skills 정리

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | AI CCTV 보안 분석 플랫폼 |
| **핵심 아키텍처** | OpenCV 1차 필터링 → 이상 프레임만 LLM 전달 → 위험도 분석 |
| **Python** | 3.14 (venv 가상환경) |
| **LLM 모델** | GPT-4o-mini |
| **주요 패키지** | openai 2.36.0, python-dotenv 1.2.2, langchain 1.3.0, langchain-openai 1.2.1, langchain-community 0.4.1 |

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
- **`format_detections()` 유틸리티 함수**: 탐지 결과를 텍스트로 변환
```python
def format_detections(frame_data):
    lines = [f"[{frame_data['timestamp']}] 프레임 #{frame_data['frame_id']}"]
    for d in frame_data['detections']:
        lines.append(
            f"- {d['class']} 탐지 (신뢰도 {d['confidence']:.0%}), "
            f"위치: 좌상단({d['bbox'][0]},{d['bbox'][1]}) "
            f"우하단({d['bbox'][2]},{d['bbox'][3]})"
        )
    return "\n".join(lines)
```

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
  - 파이프 연산자(|)를 이용해 여러 함수를 순차적으로 연결 (함수형 프로그래밍 스타일)
- **코드 패턴**:
```python
chain = RunnableLambda(func1) | RunnableLambda(func2)
result = chain.invoke(input_data)
```
### Skill 13: ChatPromptTemplate - System/Human 메시지 분리
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

### Skill 14: JsonOutputParser - 유연한 JSON 파싱 (Robustness)
- **파일**: `part02_langchain/ch02_output_parser.py`
- **핵심 클래스**: `JsonOutputParser`
- **기능**:
  - LLM이 반환한 텍스트에서 JSON 부분만 추출하여 파이썬 딕셔너리로 변환
  - **강점**: LLM이 JSON을 마크다운 코드 블록(```json ... ```)으로 감싸서 응답해도 에러 없이 자동으로 파싱함
  - 파이썬 내장 `json.loads()`는 마크다운 형식이 포함되면 파싱 에러가 발생하므로, 랭체인의 파서를 쓰는 것이 훨씬 안정적임
- **코드 패턴**:
```python
json_parser = JsonOutputParser()
result = json_parser.parse(llm_markdown_response)
```

### Skill 15: 탐지 신뢰도(Confidence)의 비판적 해석 유도
- **파일**: `part02_langchain/ch02_lcel_pipeline.py`
- **프롬프트 기법**:
  - LLM에게 "YOLO의 신뢰도는 클래스 확률일 뿐, 객체의 실제 존재 여부에 대한 절대적 신뢰를 의미하지 않는다"는 점을 명시적으로 교육
  - 이를 통해 LLM이 신뢰도가 낮은 객체에 대해 "불확실성"을 인지하고 보수적인(신중한) 분석 결과를 내놓도록 유도
- **효과**: 신뢰도가 낮은 탐지 건에 대해 무조건적인 위험 판정 대신 "불확실함에 따른 추가 모니터링 필요"와 같은 합리적인 조치 사항을 제안하게 됨

### Skill 16: 최종 랭체인 파이프라인 통합 (End-to-End Pipeline)
- **파일**: `part02_langchain/ch02_lcel_pipeline.py`
- **핵심**: 지금까지 만든 모든 컴포넌트를 하나로 연결하여 완성된 AI 보안 분석 엔진 구축
- **파이프라인 구조**:
  ```python
  analysis_chain = (
      formatter      # 1. OpenCV 데이터를 프롬프트용 딕셔너리로 전처리 (RunnableLambda)
      | cctv_prompt  # 2. 전처리된 데이터를 시스템/휴먼 메시지 템플릿에 주입
      | llm          # 3. GPT-4o-mini 모델 호출 (temperature=0.0)
      | json_parser  # 4. LLM의 텍스트 응답을 JSON(dict)으로 파싱
  )
  ```
- **의의**: raw 데이터 입력부터 최종 구조화된 결과 출력까지 코드 한 줄(`invoke`)로 실행 가능한 선언적 아키텍처 완성

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

## 알려진 이슈

| 이슈 | 설명 |
|------|------|
| **환경변수 키 통일** | 모든 파일에서 `OPENAI_API_KEY`를 사용하도록 통일됨 |

---

## 코드 스타일 컨벤션
- 모든 주석·출력·**문서(Artifact)** 및 **AI 답변**은 **한국어**로 작성
- 파일명: `ch{번호}_{영문설명}.py` 형식
- 이모지 사용 금지
- 교육적 설명 주석을 상세히 포함
- `temperature = 0.0` (JSON 분석) / `0.3` (일반 분석)
- **프롬프트 내 지시**: 모델에게 항상 "한국어로 답변할 것"을 명시적으로 지시
