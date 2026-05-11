# ─────────────────────────────────────────────────────────────────
# 예제 08: 멀티턴 대화 — 대화 히스토리 직접 관리
# - API는 "기억"이 없기 때문에 매번 전체 대화를 전달합니다
# - 이 패턴을 Part 02에서 LangChain Memory가 자동화해줍니다
# ─────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPEN_AI_MODEL")

def chat_with_history(client, history:list, user_message:str) -> tuple[str, list]:
  """
    대화 히스토리를 유지하면서 API를 호출합니다.

    Args:
        client       : OpenAI 클라이언트
        history      : 이전 대화 내역 (messages 리스트)
        user_message : 이번 차례 사용자 메시지

    Returns:
        (assistant_reply, updated_history)
        - assistant_reply  : AI 답변 문자열
        - updated_history  : 이번 대화가 추가된 히스토리
  """
  updated_history = history.copy() # 원본 보호
  updated_history.append({"role" : "user", "content" : user_message})  # 새 메세지를 히스토리에 추가

  # 2. 전체 히스토리를 API에 전달
  response = client.chat.completions.create(
        model       = model,
        messages    = updated_history,   # ← 전체 히스토리 전달
        max_tokens  = 200,
        temperature = 0.3,
  )  

  assistant_reply = response.choices[0].message.content

  updated_history.append({"role" : "assistant", "content" : assistant_reply}) # AI 답변을 히스토리에 추가

  return assistant_reply, updated_history

# 시나리오: 3단계 상황 전개
scenario = [
    "새벽 2시에 창고 앞에서 사람 1명이 탐지됐어요.",
    "10분 뒤에 같은 장소에서 또 탐지됐는데, 이번엔 2명이에요.",
    "방금 전에 분석한 상황 두 개를 요약해줘서 최종 보고서를 만들어줘.",
]

history = [
    {
        "role":    "system",
        "content": "당신은 AI CCTV 보안 분석 시스템입니다. 이전 대화 맥락을 기억하며 분석합니다."
    }
]

print("=== 멀티턴 대화 (3턴) ===\n")

total_tokens = 0

for turn_num, user_msg in enumerate(scenario, start=1):
    print(f"[{turn_num}턴] 🧑 운영자: {user_msg}")

    reply, history = chat_with_history(client, history, user_msg)

    print(f"      🤖 시스템: {reply}")
    print(f"      📊 현재 히스토리 길이: {len(history)}개 메시지\n")

print("─" * 55)
print(f"최종 히스토리 구조:")
for i, msg in enumerate(history):
    role = msg["role"]
    preview = msg["content"][:35] + "..." if len(msg["content"]) > 35 else msg["content"]
    print(f"  [{i}] {role:9s}: {preview}")

print()
print("💡 3턴이 지나자 히스토리가 7개 메시지(system + 3×user/assistant)로 늘었습니다.")
print("   대화가 길어질수록 토큰이 늘어납니다.")
print("   → Part 02에서 LangChain이 이걸 자동 관리합니다.")  