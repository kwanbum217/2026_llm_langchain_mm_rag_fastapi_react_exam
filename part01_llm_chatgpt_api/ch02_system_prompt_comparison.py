# ─────────────────────────────────────────────────────────────────
# 예제 04: System Prompt 비교 실험
# - 동일한 user 메시지에 system prompt만 바꿔서 3번 호출합니다
# - "system이 얼마나 중요한지"를 직접 체감하는 실습입니다
# ─────────────────────────────────────────────────────────────────

# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# 윈도우 터미널 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPEN_AI_MODEL")

# 모든 실험에서 동일한 user 메시지를 씁니다
USER_MESSAGE = "새벽 2시에 창고 출입구에서 사람 2명이 탐지됐습니다."

# 비교할 system prompt 세 가지
# 딕셔너리 형태로 묶어서 반복 처리 — 코드 중복을 줄이는 패턴
experiments = [
    {
        "name":   "① 설정 없음",
        "system": None,   # system 없이 호출하면 어떻게 되는지 확인
    },
    {
        "name":   "② 일반 어시스턴트",
        "system": "You are a helpful assistant.",
    },
    {
        "name":   "③ CCTV 전문가 (이 강의 버전)",
        "system": (
            "당신은 AI CCTV 보안 분석 시스템입니다. "
            "OpenCV로 탐지된 객체 정보를 입력받아 위험도를 분석합니다. "
            "답변 형식: 위험도(정상/주의/위험) + 판단 근거 + 권고 조치. "
            "한국어로만 답합니다."
        ),
    },
]

print("=== System Prompt 비교 실험 ===")
print(f"\n[공통 질문] {USER_MESSAGE}\n")

for exp in experiments:
    print(f"\n{'─' * 55}")
    print(f"[CONFIG] {exp['name']}")

    # messages 구성 — system이 None이면 user만 포함
    if exp["system"] is not None:
        # system prompt가 있으면 messages의 첫 번째에 추가
        messages = [
            {"role": "system", "content": exp["system"]},
            {"role": "user",   "content": USER_MESSAGE},
        ]
    else:
        # system 없이 user 메시지만 전달
        messages = [
            {"role": "user", "content": USER_MESSAGE},
        ]

    response = client.chat.completions.create(
        model      = model,
        messages   = messages,
        max_tokens = 150,
        temperature = 0.3,   # 실험 간 비교를 위해 낮게 고정
    )

    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens

    print(f"[ASSISTANT] 응답 ({tokens} 토큰):")
    print(f"   {answer}")

print(f"\n{'─' * 55}")
print("\n[INFO] 정리:")
print("   ① 설정 없음  → 영어로 답변, 형식 없음")
print("   ② 일반 AI    → 영어 또는 불규칙 한국어, 형식 없음")
print("   ③ 전문가 설정 → 한국어, 구조화된 분석, 실무 활용 가능")
print("\n   → 이 강의에서는 항상 ③ 방식으로 system을 설정합니다!")