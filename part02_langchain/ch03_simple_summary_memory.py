# ─────────────────────────────────────────────────────────────
# SummaryMemory 동작 원리를 이해하기 위한 교육용 구현
# 실제 LangChain에서는 요약을 LLM이 자동으로 수행(InMemoryChatMessageHistory)합니다
# ─────────────────────────────────────────────────────────────

class SimpleSummaryMemory:
    """
    오래된 대화를 요약으로 압축하는 메모리 클래스.

    동작 방식:
    1. 최근 max_recent개까지는 원문 유지
    2. 그 이상이 되면 가장 오래된 것을 요약으로 압축
    3. LLM에는 [요약 + 최근 원문]을 함께 전달

    실제 LangChain ConversationSummaryMemory는
    압축 요약도 LLM이 자동으로 수행합니다.
    """

    def __init__(self, max_recent: int = 4):
        # 최근 대화 원문 (최대 max_recent개 보관)
        self.recent: list = []
        # 오래된 대화의 요약본 (한 줄로 압축)
        self.summary: str = ""
        # 최대 유지할 최근 대화 수
        self.max_recent = max_recent

    def add_exchange(self, user_msg: str, ai_msg: str):
        """
        대화 1회(사용자 질문 + AI 응답)를 메모리에 추가.

        max_recent 초과 시 가장 오래된 대화를 요약으로 압축합니다.
        """
        # 새 대화 추가
        self.recent.append({"human": user_msg, "ai": ai_msg})

        # 최대 개수 초과 시 → 가장 오래된 것을 요약으로 압축
        if len(self.recent) > self.max_recent:
            oldest = self.recent.pop(0)  # 가장 오래된 대화 꺼내기

            # 실제 LangChain에서는 LLM이 요약합니다:
            #   summary_chain.invoke({"conversation": oldest})
            # 여기서는 단순 압축으로 대체
            compressed = f"[{oldest['human'][:20]}] → [{oldest['ai'][:25]}]"

            # 기존 요약에 붙이기
            self.summary = (self.summary + " | " + compressed
                            if self.summary else compressed)

    def get_context(self) -> str:
        """
        LLM에 전달할 전체 컨텍스트 반환.

        구조: [이전 대화 요약] + [최근 대화 원문]
        """
        parts = []
        if self.summary:
            parts.append(f"[이전 대화 요약]\n{self.summary}\n")
        if self.recent:
            parts.append("[최근 대화]")
            for exchange in self.recent:
                parts.append(f"Human: {exchange['human']}")
                parts.append(f"AI: {exchange['ai']}")
        return "\n".join(parts)

# ── 시연: 4개 프레임 분석 후 상태 확인 ──────────────────────
summary_mem = SimpleSummaryMemory(max_recent=2)

exchanges = [
    ("1번 프레임 분석", '{"risk_level":"정상"}'),
    ("2번 프레임 분석", '{"risk_level":"주의"}'),
    ("3번 프레임 분석", '{"risk_level":"위험"}'),
    ("3번 프레임 왜 위험이야?", "새벽 2시 창고 2인 탐지 - 침입 가능성"),
]

for user, ai in exchanges:
    summary_mem.add_exchange(user, ai)

print("[SummaryMemory 상태 - 4회 대화 후]")
print(summary_mem.get_context())