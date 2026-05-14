# 지금까지 만든 세 부품을 | 연산자로 연결하여 최종 랭체인 파이프 라인을 구성합니다

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# .env 파일 로드
load_dotenv()


def format_detections(frame_data: dict) -> dict:
    """OpenCV 탐지 결과 → PromptTemplate 입력 딕셔너리 (step03과 동일)"""
    detections = frame_data.get("detections", [])
    lines = []
    if not detections:
        lines.append("- 탐지된 객체 없음")
    else:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            lines.append(
                f"- {d['class']} 탐지 (신뢰도 {d['confidence']:.0%}), "
                f"위치: 좌상단({x1},{y1}) 우하단({x2},{y2}), "
                f"크기: {x2-x1}×{y2-y1}px"
            )
    return {
        "frame_id": frame_data["frame_id"],
        "timestamp": frame_data["timestamp"],
        "location": frame_data.get("location", "미지정"),
        "detections_text": "\n".join(lines),
    }


formatter = RunnableLambda(format_detections)

# PromptTemplate 컴포넌트
cctv_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # System: LLM에게 역할과 출력 규칙을 알려줍니다
            # {{중괄호 두 개}}는 리터럴 중괄호 문자입니다 (변수가 아님)
            """당신은 AI CCTV 보안 관제 전문가입니다.
탐지 결과를 분석하여 위험도를 판단하고 조치사항을 제안하세요.

반드시 한국어로만 답변하세요.
반드시 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력하세요.
{{"risk_level": "정상|주의|위험", "reason": "판단 이유", "action": "조치사항"}}""",
        ),
        (
            "human",
            # Human: 실제 분석 요청 — {중괄호 하나}는 나중에 실제 값으로 교체됩니다
            """── CCTV 탐지 보고 ──
프레임 번호: {frame_id}번
촬영 시각  : {timestamp}
촬영 위치  : {location}

탐지된 객체:
{detections_text}

단, 위에서 신뢰도는 YOLO가 판단한 클래스 일 확률이지, 신뢰있는 객체를 뜻하지는 않는다. 이를 염두에 두고 위험도를 판단해야한다
위 탐지 결과를 분석하고 위험도를 JSON으로 반환하세요.""",
        ),
    ]
)

# Model 컴포넌트
llm = ChatOpenAI(
  model="gpt-4o-mini", temperature=0.0
)

# Json Output Parser 컴포넌트
json_parser = JsonOutputParser()

# chain 컴포넌트
# | 연산자로 파이프라인 조립
analysis_chain = ( 
  formatter  # yolo가 준 문자열을  프롬프트로 만들기 전 전처리
  | cctv_prompt # llm에게 전송할 프롬프트
  | llm  # llm 호출
  | json_parser # 응답 데이터 파싱
)

frame_data = {
    "frame_id" : 1,
    "timestamp": "02:13",           # 새벽 2시 13분 → 심야 시간대
    "location" : "주차장 A구역",
    "detections": [
        # OpenCV YOLO 탐지 결과 그대로
        {"class": "person", "bbox": [120, 80, 200, 350], "confidence": 0.91},
        {"class": "person", "bbox": [310, 95, 390, 360], "confidence": 0.87},
        {"class": "car",    "bbox": [50, 200, 280, 400],  "confidence": 0.95},
    ],
}

result = analysis_chain.invoke(frame_data)
print("반환 타입:", type(result))   # <class 'dict'>
print()
print("분석 결과:")
print(f"  위험도: {result['risk_level']}")
print(f"  판단 이유: {result['reason']}")
print(f"  조치사항: {result['action']}")