# ─────────────────────────────────────────────────────────────────
# 예제 07: response_format=json_object 로 JSON 출력받기
# - LLM 답변을 파이썬 딕셔너리로 바로 활용하는 핵심 패턴입니다
# - Part 01 실습 최종 예제와 동일한 구조입니다
# ─────────────────────────────────────────────────────────────────

# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json          # JSON 파싱 라이브러리 (파이썬 내장)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPEN_AI_MODEL")

# ── System Prompt: JSON 출력 형식을 명시합니다 ────────────────────
# response_format=json_object 를 쓸 때는
# system prompt에 "JSON으로 답해"라고 반드시 명시해야 합니다
SYSTEM = """당신은 AI CCTV 보안 분석 시스템입니다.
탐지 결과를 분석하여 반드시 아래 JSON 형식으로만 답하세요.
다른 텍스트는 절대 포함하지 마세요.

{
  "timestamp":    "탐지 시각",
  "location":     "탐지 위치",
  "person_count": 탐지된 사람 수 (숫자),
  "risk_level":   "정상" 또는 "주의" 또는 "위험",
  "reason":       "판단 근거 (한 문장)",
  "action":       "권고 조치 (한 문장)"
}"""

USER = "2025-04-04 02:13, 주차장 A구역에서 사람 2명, 차량 1대가 탐지됐습니다."

# ── API 호출 ──────────────────────────────────────────────────────
response = client.chat.completions.create(
    model       = model,
    messages    = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER},
    ],
    max_tokens  = 300,
    temperature = 0.0,   # JSON 분석은 항상 0.0 — 일관된 형식 보장
    response_format = {"type": "json_object"},   # ← 핵심 설정!
    # json_object 모드: 응답이 반드시 유효한 JSON임을 보장합니다
)

raw_text = response.choices[0].message.content
print(raw_text)

#  json 파싱
try:
  result = json.loads(raw_text)

except json.JSONDecodeError as e:
  # 예외 처리: JSON 형식이 아닌 응답이 왔을 때
  # temperature가 0.0이고 system에 명시했다면 거의 발생하지 않음
  print(f"JSON 파싱 실패: {e}")
  print("→ system prompt에 JSON 형식을 더 명확하게 명시하세요.")
  result = None
  
#  파싱된 결과 (딕셔너리)
for key, value in result.items():
  print(f'{key}: {value}')

# ── 결과 활용 — 위험도에 따른 자동 분기 ──────────────────────────
if result:
    print("\n=== 위험도 기반 자동 조치 ===")

    # 위험도별 처리 방법 정의
    risk_handlers = {
        "정상": {
            "icon":   "🟢",
            "action": "로그 저장만 (추가 조치 없음)",
            "notify": False,
        },
        "주의": {
            "icon":   "🟡",
            "action": "경비팀 알림 전송",
            "notify": True,
        },
        "위험": {
            "icon":   "[ALERT]",
            "action": "경찰 즉시 신고 + 비상 알람",
            "notify": True,
        },
    }

    level = result.get("risk_level", "정상")
    handler = risk_handlers.get(level, risk_handlers["정상"])

    print(f"  {handler['icon']} 위험도 : {level}")
    print(f"  [ACTION] 자동 조치: {handler['action']}")

    if handler["notify"]:
        # 실제로는 여기서 경비팀 알림 API를 호출하거나
        # DB에 저장하거나 이메일을 보내는 코드를 넣습니다
        # (Part 05 FastAPI에서 구현합니다)
        print(f"  [SEND] 알림 전송: {result.get('action', '')}")

    print(f"\n  [SAVE] 분석 결과를 DB에 저장...")
    print(f"  [OK] 처리 완료 — {result.get('timestamp', '')}")