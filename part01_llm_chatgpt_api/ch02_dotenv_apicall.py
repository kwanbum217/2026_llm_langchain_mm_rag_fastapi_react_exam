# ─────────────────────────────────────────────────────────────────
# 예제 02: .env 파일 읽기 + OpenAI 클라이언트 생성
# - python-dotenv 라이브러리로 .env 파일을 자동으로 읽어옵니다
# ─────────────────────────────────────────────────────────────────

# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from dotenv import load_dotenv   # python-dotenv 패키지에서 가져오기
from openai import OpenAI

# .env 파일을 읽어 환경변수 등록
load_dotenv()

# api key 읽기
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPEN_AI_MODEL")

if not api_key:
  raise ValueError(
    "OPENAI_API_KEY가 환경변수가 없습니다."
  )

#  OPENAI 클라이언트 객체 생성
client = OpenAI(api_key = api_key)

#  API_KEY 일부만 출력
masked_key = api_key[:12] + "..." + api_key[-4:]

print(f"API key load complete: {masked_key}")
print(f"   Client type: {type(client)}")

# API 호출 ------------------------------------
response = client.chat.completions.create(
    model = model,
    messages = [
        # ── 역할 1: system ───────────────────────────────────────────
        # AI의 정체성(페르소나)을 설정합니다
        # 대화 내내 유지되는 "배경 설명"입니다
        # → "넌 이런 AI야, 이렇게 행동해"
        {
            "role": "system",
            "content": "당신은 AI CCTV 보안 분석 시스템입니다. "
                   "답변은 항상 한국어로, 간결하게 작성합니다."
        },
        # ── 역할 2: user ─────────────────────────────────────────────
        # 사람이 AI에게 보내는 메시지입니다
          # → "내가 물어보는 내용"
        {
            "role": "user",
            "content": "안녕하세요 .잘 작동하나요?."
        },
    ],
    max_tokens = 100
)

# ── 응답 구조 파헤치기 ───────────────────────────────────────────
print("=== API 첫 번째 호출 성공! ===\n")

# 핵심 답변 꺼내기
# response.choices: 여러 개의 응답 후보 (보통 1개)
# [0]: 첫 번째(기본) 후보
# .message.content: 실제 텍스트 답변
answer = response.choices[0].message.content
print(f"GPT 답변 : {answer}")

# 종료 이유 확인(finish_reason)
# "stop"    → 정상 완료
# "length"  → max_tokens에 잘림 (max_tokens 늘려야 함)
# "content_filter" → 콘텐츠 정책 위반으로 차단됨
finish_reason = response.choices[0].finish_reason
print(f"종료 사유 : {finish_reason}")

# 토큰 사용량 확인
usage = response.usage
print(f"\n[STATUS] 토큰 사용량:")
print(f"   입력 (prompt)    : {usage.prompt_tokens:>5} 토큰")
print(f"   출력 (completion): {usage.completion_tokens:>5} 토큰")
print(f"   합계 (total)     : {usage.total_tokens:>5} 토큰")

# GPT-4o 기준 비용 계산
# 입력: $2.50 / 1M 토큰,  출력: $10.00 / 1M 토큰
input_cost  = (usage.prompt_tokens    / 1_000_000) * 2.50
output_cost = (usage.completion_tokens / 1_000_000) * 10.00
total_cost  = input_cost + output_cost
print(f"\n[COST] 이번 호출 비용: ${total_cost:.6f}  (약 {total_cost * 1400:.4f}원)")