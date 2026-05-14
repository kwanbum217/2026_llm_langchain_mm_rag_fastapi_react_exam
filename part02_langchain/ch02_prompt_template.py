from langchain_core.prompts import ChatPromptTemplate

# ChatPromptTemplate.from_messages() 로 System / Human 두 역할 함께 정의
cctv_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # System: LLM에게 역할과 출력 규칙을 알려줍니다
            # {{중괄호 두 개}}는 리터럴 중괄호 문자입니다 (변수가 아님)
            """당신은 AI CCTV 보안 관제 전문가입니다.
탐지 결과를 분석하여 위험도를 판단하고 조치사항을 제안하세요.

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

위 탐지 결과를 분석하고 위험도를 JSON으로 반환하세요.""",
        ),
    ]
)

#  위 템플릿에 실제 값을 채워서 메시지를 생성해 보자
test_messages = cctv_prompt.format_messages(
    frame_id=42,  # 프레임 번호
    timestamp="02:13",  # 촬영 시각
    location="주차장 A구역",  # 촬영 위치
    detections_text=(  # 탐지 결과 (여러 줄 문자열)
        "- person 탐지 (신뢰도 91%), 위치: 좌상단(120,80) 우하단(200,350)\n"
        "- car    탐지 (신뢰도 95%), 위치: 좌상단(50,200) 우하단(280,400)"
    ),
)

# 어떻게 생겼는지 확인
for msg in test_messages:
    print(f"[{msg.type.upper()}]")
    print(msg.content)
    print("─" * 50)