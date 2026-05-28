# ─────────────────────────────────────────────────────────────────
# 예제 05: temperature 파라미터 실험
# - 같은 질문을 temperature만 바꿔서 3번 호출합니다
# - 숫자가 클수록 "다양하고 창의적", 작을수록 "일관되고 안정적"
# ─────────────────────────────────────────────────────────────────

# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPEN_AI_MODEL")

SYSTEM = (
    "당신은 AI CCTV 보안 분석 시스템입니다. "
    "위험도(정상/주의/위험)와 이유를 간결하게 답합니다."
)
USER = "새벽 2시에 주차장에서 사람 2명이 탐지됐습니다. 위험도를 분석해주세요."

# 실험할 temperature 값과 설명
# temperature 범위: 0.0(완전 결정적) ~ 2.0(매우 무작위)
# 실무에서는 0.0~1.0 범위만 사용합니다
temperature_tests = [
    (0.0, "완전 결정적 — 항상 같은 답 (보안 분석에 적합)"),
    (0.7, "중간 — 약간의 변화 허용"),
    (1.5, "높음 — 불안정, 실무 비권장"),
]

print("=== temperature 파라미터 실험 ===\n")
print(f"[질문] {USER}\n")

for temp, desc in temperature_tests:
    print(f"\n{'─' * 55}")
    print(f"[TEMP]  temperature = {temp}  ({desc})")
    print("    같은 질문을 3번 호출:")

    # 동일한 temperature로 3번 호출해서 일관성을 확인합니다
    for call_num in range(1, 4):
        response = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": USER},
            ],
            max_tokens  = 80,
            temperature = temp,   # 실험 대상 파라미터
        )
        answer = response.choices[0].message.content
        # 긴 답변은 한 줄로 압축해서 비교하기 쉽게 표시
        one_line = answer.replace("\n", " / ")
        print(f"    호출 {call_num}: {one_line[:70]}")

print(f"\n{'─' * 55}")
print("\n[TIP] 결론:")
print("   CCTV 위험도 분석에는 temperature=0.0~0.3 사용을 권장합니다.")
print("   위험 판단이 매번 달라지면 신뢰할 수 없는 시스템이 됩니다.")