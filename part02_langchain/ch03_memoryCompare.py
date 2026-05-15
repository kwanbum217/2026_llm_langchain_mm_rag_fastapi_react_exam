# ─────────────────────────────────────────────────────────────
# 예제 1: 기억 없는 LLM vs. 기억 있는 LLM 비교
#
# 이 예제는 실제 API 없이 실행되는 시뮬레이션입니다.
# 실제 수업에서는 mock_llm 부분을 ChatOpenAI로 교체하면 됩니다.
# ─────────────────────────────────────────────────────────────


def mock_llm(messages: list) -> str:
    """
    실제 LLM 대신 동작을 시뮬레이션하는 함수.

    이 함수는 메시지 이력이 있을 때와 없을 때
    다르게 동작하여, Memory의 필요성을 보여줍니다.

    실제 수업에서는 이 함수를 ChatOpenAI로 교체합니다:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(messages)
        return response.content
    """
    last_msg = messages[-1]["content"]

    # 대화 이력에서 3번 프레임 관련 내용이 있는지 확인
    has_frame3_context = any(
        "frame_id: 3" in m["content"] or "3번 프레임" in m["content"]
        for m in messages
    )

    if "왜 위험" in last_msg:
        if has_frame3_context:
            # 이력이 있으면: 맥락을 이해한 답변
            return "3번 프레임(02:13)은 창고 출입구에서 새벽에 person 2명이 탐지되어 '위험'으로 분류했습니다."
        else:
            # 이력이 없으면: 맥락 없이 답변 불가
            return "죄송합니다. 어떤 프레임에 대해 말씀하시는지 알 수 없습니다."
    elif "분석해" in last_msg:
        return '{"frame_id": 3, "risk_level": "위험", "reason": "새벽 2시 창고 2인 탐지"}'
    return "네, 말씀하세요."


# ────────────────────────────────────────────────────────
# 케이스 A: 기억 없는 경우 — 매번 독립적인 메시지만 전달
# ────────────────────────────────────────────────────────
print("【케이스 A: 기억 없는 LLM】")

# 첫 번째 호출: 분석 요청
response_1 = mock_llm([{"role": "user", "content": "3번 프레임 분석해줘"}])
print(f"사용자: 3번 프레임 분석해줘")
print(f"LLM   : {response_1}")

# 두 번째 호출: 이전 대화 없이 새로 시작
# → LLM은 response_1을 전혀 모릅니다
response_2 = mock_llm([{"role": "user", "content": "왜 위험으로 분류했어?"}])
print(f"\\n사용자: 왜 위험으로 분류했어?")
print(f"LLM   : {response_2}")
print("→ ❌ 이전 분석을 기억하지 못합니다!\\n")


# ────────────────────────────────────────────────────────
# 케이스 B: 기억 있는 경우 — 이전 대화를 함께 전달
# ────────────────────────────────────────────────────────
print("【케이스 B: 기억 있는 LLM】")

# 대화 이력을 직접 관리하는 리스트
# LangChain Memory는 이 역할을 자동으로 해줍니다
chat_history = []

# 첫 번째 교환: 분석 요청 + 응답을 이력에 저장
user_msg_1 = "3번 프레임 분석해줘. person 2명, 새벽 02:13, 창고 출입구"
chat_history.append({"role": "user", "content": user_msg_1})  # 이력 저장

ai_msg_1 = mock_llm(chat_history)
chat_history.append({"role": "assistant", "content": ai_msg_1})  # 이력 저장

print(f"사용자: {user_msg_1}")
print(f"LLM   : {ai_msg_1}")

# 두 번째 교환: 이전 이력 + 새 질문을 함께 전달
user_msg_2 = "왜 위험으로 분류했어?"
chat_history.append({"role": "user", "content": user_msg_2})  # 이력 저장

ai_msg_2 = mock_llm(chat_history)  # ← 전체 이력을 포함해 전달
chat_history.append({"role": "assistant", "content": ai_msg_2})  # 이력 저장

print(f"\\n사용자: {user_msg_2}")
print(f"LLM   : {ai_msg_2}")
print("→ ✅ 이전 분석 내용을 기억합니다!")

# 현재 이력 상태 확인
print(f"\\n현재 이력 길이: {len(chat_history)}개 메시지")