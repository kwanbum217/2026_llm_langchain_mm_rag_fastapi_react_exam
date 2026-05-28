# ─────────────────────────────────────────────────────────────────
# 예제 06: max_tokens 파라미터 실험
# - max_tokens: 출력의 최대 토큰 수를 제한합니다
# - 너무 작으면 답변이 잘리고, 너무 크면 비용이 늘어납니다
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

SYSTEM = "당신은 CCTV 보안 분석 전문가입니다. 한국어로 상세히 답합니다."
USER   = "새벽 2시에 창고에서 사람 2명, 차량 1대가 탐지됐습니다. 위험도를 분석해주세요."

# 다양한 max_tokens 설정 실험
token_tests = [30, 100, 300]

print("=== max_tokens 파라미터 실험 ===\n")
print(f"[질문] {USER}\n")

for max_tok in token_tests:
    print(f"\n{'─' * 55}")
    print(f"[SIZE] max_tokens = {max_tok}")

    response = client.chat.completions.create(
        model       = model,
        messages    = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER},
        ],
        max_tokens  = max_tok,
        temperature = 0.3,
    )

    answer        = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    out_tokens    = response.usage.completion_tokens

    print(f"   완료 이유: {finish_reason}")
    print(f"   실제 출력 토큰: {out_tokens}")
    print(f"   응답:\n   {answer}")

    # 잘렸다면 경고 표시
    if finish_reason == "length":
        print("\n   [WARNING]  답변이 잘렸습니다! max_tokens를 늘려야 합니다.")

print(f"\n{'─' * 55}")
print("\n[TIP] 이 강의 권장 설정:")
print("   단순 위험도 판단  → max_tokens = 100~200")
print("   상세 분석 리포트 → max_tokens = 300~500")
print("   JSON 구조화 출력 → max_tokens = 300~400")