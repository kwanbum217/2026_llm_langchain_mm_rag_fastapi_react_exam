class SimpleBufferMemory:
    """
    가장 단순한 형태의 대화 메모리 구현체.
    랭체인의 ConversationBufferMemory가 내부적으로 어떻게 작동하는지 이해하기 위한 교육용 클래스입니다.
    """

    def __init__(self):
        # 대화 이력을 저장하는 리스트
        # 각 항목: {"role": "human" 또는 "ai", "content": "내용"}
        self.messages: list = []

    def add_user_message(self, text: str):
        """사용자(운영자)의 메시지를 이력에 추가"""
        self.messages.append({"role": "human", "content": text})

    def add_ai_message(self, text: str):
        """AI의 응답을 이력에 추가"""
        self.messages.append({"role": "ai", "content": text})

    def get_all_messages(self) -> list:
        """저장된 전체 대화 이력 반환 — LLM 호출 시 사용"""
        return self.messages

    def format_as_text(self) -> str:
        """
        대화 이력을 사람이 읽기 좋은 텍스트로 변환.
        프롬프트에 직접 삽입할 때 이 형태를 사용합니다.

        출력 형태:
            Human: ...
            AI: ...
        """
        lines = []
        for msg in self.messages:
            prefix = "Human" if msg["role"] == "human" else "AI"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        """대화 이력 초기화 — 새 운영자 세션 시작 시 사용"""
        self.messages = []

    def __len__(self):
        """저장된 메시지 수 반환 (len(memory) 형태로 사용)"""
        return len(self.messages)


# ── 동작 확인 (테스트 코드) ──────────────────────────────────────────────────
if __name__ == "__main__":
    memory = SimpleBufferMemory()

    # 1. 메시지 추가
    memory.add_user_message("3번 프레임 분석해줘.")
    memory.add_ai_message("3번 프레임은 '위험'으로 판독되었습니다.")
    memory.add_user_message("왜 위험이야?")

    # 2. 이력 확인
    print("【전체 메시지 객체】")
    print(memory.get_all_messages())
    print()

    print("【프롬프트용 텍스트 변환】")
    print(memory.format_as_text())
    print()

    print(f"저장된 메시지 개수: {len(memory)}개")

    # 3. 초기화 테스트
    memory.clear()
    print(f"초기화 후 메시지 개수: {len(memory)}개")