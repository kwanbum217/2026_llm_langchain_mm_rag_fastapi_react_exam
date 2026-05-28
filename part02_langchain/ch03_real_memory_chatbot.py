# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json

from dotenv import load_dotenv  # .env 파일 로드
from langchain_openai import ChatOpenAI  # 실제 LLM
from langchain_core.messages import (
    SystemMessage,  # LLM에게 페르소나 주입용으로 만들어진 클래스
    HumanMessage,  # 유저가 LLM에게 전달할 프롬프트를 만들 클래스
    AIMessage,  # LLM이 응답한 메시지
)
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
)  # 메모리 + 요약기능이 구현된 클래스

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
# api_key가 없다면
if not api_key:
    raise EnvironmentError(
        "\n[오류] OPENAI_API_KEY 가 설정되지 않았습니다.\n"
        "  방법 A: 프로젝트 폴더에 .env 파일을 만들고\n"
        "          OPENAI_API_KEY=sk-proj-... 를 입력하세요.\n"
        "  방법 B: 터미널에서 export OPENAI_API_KEY=sk-proj-... 를 실행하세요."
    )

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# ── 시스템 프롬프트 ──────────────────────────────────────────────────────────
# LLM 에게 항상 알려줄 역할과 규칙
# 모든 대화 요청에 첫 번째 메시지로 포함됩니다
SYSTEM_PROMPT = """당신은 AI CCTV 보안 분석 어시스턴트입니다.
OpenCV로 탐지된 결과를 분석하고 위험도를 판단합니다.
이전 분석 결과를 기억하여 운영자의 후속 질문에 정확히 답변합니다.
위험도는 반드시 '정상' / '주의' / '위험' 중 하나로만 분류하세요.

[분석 요청 시 응답 형식]
반드시 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력하세요.
{{"risk_level": "정상|주의|위험", "reason": "판단 이유", "action": "조치사항"}}

[후속 질문 시]
이전 분석 이력을 참고하여 자연스러운 한국어로 답변하세요."""


class CCTVOperatorChatbot:
    """
    CCTV 운영자를 위한 Memory를 챗봇( 실제 ChatOpenAI 버전)

    주요 메서드:
      analyze_frame(frame_id, description) : 프레임 분석 및 대화 이력 저장
      ask: 후속 질문 처리 및 답변 생성
      show_memory_state() : 현재 메모리 상태 확인(디버깅용)
    """

    def __init__(self):
        # 대화 이력 저장소 - 프로그램 실행중 (현재 세션)에만 유지 (RAM 저장)
        self.history = InMemoryChatMessageHistory()

        self.frame_cache: dict = {}

    def _build_messages(self, user_input: str) -> list:
        """
        LLM 에 전달할 메시지 리스트를 구성합니다.

        구조:
            [SystemMessage]  역할 정의 (항상 첫 번째)
            [HumanMessage]   1번 대화  ← 이력
            [AIMessage]      1번 응답  ← 이력
            [HumanMessage]   2번 대화  ← 이력
            ...
            [HumanMessage]   현재 질문 (항상 마지막)

        이 구조 덕분에 LLM 이 이전 대화 맥락을 모두 볼 수 있습니다.
        """
        return [
            SystemMessage(content=SYSTEM_PROMPT),  # 역할 정의
            *self.history.messages,  # 전체 이전 대화 (언팩 : history.messages가 list의 아이템 꺼냄)
            HumanMessage(content=user_input),  # 현재 질문
        ]

    def _call_llm(self, user_input: str) -> str:
        """
        실제 ChatOpenAI 를 호출하고 응답 문자열을 반환합니다.

        step04 의 mock_llm() 과 완전히 동일한 인터페이스입니다.
        내부적으로만 사용합니다 (analyze_frame, ask 에서 호출).
        """
        messages = self._build_messages(user_input)

        # 실제 API 호출 — 인터넷 연결 + API 키 필요
        response = llm.invoke(messages)

        # response.content 가 LLM 이 반환한 텍스트 문자열입니다
        return response.content

    # JsonOutputParser는 랭체인 전용 컴포넌트. 이 예제는 프레임분석을 요청하는 프롬프트와
    # 일반적인 질문하는 (자연어 질문) 프롬프트가 섞여 있기 때문에...
    # 자연어로 질문할 때는 LLM이 text로 답변할 확률이 높다...
    def _safe_json_parse(self, text: str, frame_id: int) -> dict:
        """
        LLM 응답 문자열을 JSON 으로 파싱합니다.

        LLM 이 간혹 JSON 을 ```json ... ``` 마크다운으로 감싸서 줄 때도
        정상 처리합니다.

        파싱 실패 시 raw_response 를 포함한 딕셔너리를 반환합니다.
        """
        # 마크다운 코드블록 제거
        text = text.strip()
        if text.startswith("```"):
            # ```json\n...\n``` 또는 ```\n...\n``` 형태 처리
            lines = text.split("\n")
            # 첫 줄(```json)과 마지막 줄(```) 제거
            text = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # JSON 파싱 실패 — 원문 그대로 저장
            print(f"  파싱 실패: 프레임 {frame_id} JSON 파싱 실패. 원문 저장.")
            return {"frame_id": frame_id, "raw_response": text}

    def analyze_frame(
        self,
        frame_id: int,
        detections: list,
        timestamp: str,
        location: str,
    ) -> dict:
        """
        OpenCV 탐지 결과를 LLM 으로 분석하고 결과를 메모리에 저장합니다.

        Parameters
        ----------
        frame_id   : 프레임 번호 (예: 3)
        detections : OpenCV 탐지 결과 리스트
                     [{"class": "person", "bbox": [x1,y1,x2,y2], "confidence": 0.91}, ...]
        timestamp  : 탐지 시각 문자열 (예: "02:13")
        location   : 위치 설명 (예: "창고 출입구")

        Returns
        -------
        dict : LLM 이 반환한 위험도 분석 결과
               {"risk_level": "위험", "reason": "...", "action": "..."}
        """
        # ① 탐지 결과 → LLM 이 읽기 좋은 텍스트로 변환
        #    Part 01 format_detections 패턴 재활용
        det_lines = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            det_lines.append(
                f"  - {d['class']} (신뢰도 {d['confidence']:.0%}), "
                f"위치: 좌상단({x1},{y1}) 우하단({x2},{y2}), "
                f"크기: {x2-x1}×{y2-y1}px"
            )

        user_msg = (
            f"frame_id: {frame_id} 분석 요청\n"
            f"시각: {timestamp} | 위치: {location}\n"
            f"탐지 결과:\n" + "\n".join(det_lines)
        )

        # ② 실제 LLM 호출
        print(f"  [API 호출] 프레임 {frame_id} 분석 중...", end=" ", flush=True)
        ai_response = self._call_llm(user_msg)
        print("완료")

        # ③ 대화 이력에 저장 ← Memory 의 핵심
        # exchange (사용자 프롬프트 + 그 프롬프트에 대한 LLM의 응답)의 쌍으로 저장
        self.history.add_user_message(user_msg)
        self.history.add_ai_message(ai_response)

        # ④ JSON 파싱 + 캐시 저장
        result = self._safe_json_parse(ai_response, frame_id)
        result["frame_id"] = frame_id  # frame_id 가 없으면 추가
        self.frame_cache[frame_id] = result

        return result

    def ask(self, question: str) -> str:
        """
        운영자의 자유 질문을 처리합니다.
        이전 모든 분석 대화가 자동으로 컨텍스트에 포함됩니다.
        """
        print(f"  [API 호출] 질문 처리 중...", end=" ", flush=True)
        ai_response = self._call_llm(question)
        print("완료")

        # 후속 질문도 이력에 저장 (다음 질문의 맥락이 됩니다)
        self.history.add_user_message(question)
        self.history.add_ai_message(ai_response)

        return ai_response


    def show_memory_state(self):
        """현재 메모리 상태를 출력합니다. 디버깅 및 교육용."""
        danger_frames = [
            k
            for k, v in self.frame_cache.items()
            if isinstance(v, dict) and v.get("risk_level") == "위험"
        ]
        caution_frames = [
            k
            for k, v in self.frame_cache.items()
            if isinstance(v, dict) and v.get("risk_level") == "주의"
        ]

        print("\n" + "-" * 45)
        print("  계급별 프레임 :")
        print(f"  - 위험 프레임 : {danger_frames}")
        print(f"  - 주의 프레임 : {caution_frames}")
        print("-" * 45)

# ─────────────────────────────────────────────────────────────────────────────
# 실행 시나리오
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("   CCTV 운영자 챗봇 (실제 LLM 버전) 시작")
    print("=" * 50 + "\n")

    bot = CCTVOperatorChatbot()

    # ── 1번 프레임 분석 (정상 예상) ──────────────────────────────────────────
    print("[1번 프레임 분석]")
    r1 = bot.analyze_frame(
        frame_id=1,
        detections=[
            {"class": "person", "bbox": [100, 80, 180, 320], "confidence": 0.89},
        ],
        timestamp="13:25",
        location="주차장 A구역",
    )
    print(f"  결과: {r1}\n")

    # ── 3번 프레임 분석 (위험 예상) ──────────────────────────────────────────
    print("[3번 프레임 분석]")
    r3 = bot.analyze_frame(
        frame_id=3,
        detections=[
            {"class": "person", "bbox": [120, 80,  200, 350], "confidence": 0.91},
            {"class": "person", "bbox": [310, 95,  390, 360], "confidence": 0.87},
            {"class": "car",    "bbox": [50,  200, 280, 400], "confidence": 0.95},
        ],
        timestamp="02:13",
        location="창고 출입구",
    )
    print(f"  결과: {r3}\n")

    # ── 5번 프레임 분석 (주의 예상) ──────────────────────────────────────────
    print("[5번 프레임 분석]")
    r5 = bot.analyze_frame(
        frame_id=5,
        detections=[
            {"class": "person", "bbox": [200, 100, 300, 400], "confidence": 0.78},
        ],
        timestamp="23:45",
        location="공장 외곽",
    )
    print(f"  결과: {r5}\n")

    # ── 후속 질문 처리 ── Memory 가 핵심 ────────────────────────────────────
    print("-" * 50)
    print("질문 1: '3번 프레임 왜 위험으로 분류했어?'")
    ans1 = bot.ask("3번 프레임 왜 위험으로 분류했어?")
    print(f"  챗봇: {ans1}\n")

    print("질문 2: '지금까지 위험 등급 프레임 몇 개야?'")
    ans2 = bot.ask("지금까지 위험 등급 프레임 몇 개야?")
    print(f"  챗봇: {ans2}\n")

    print("질문 3: '오늘 분석 결과 요약해줘'")
    ans3 = bot.ask("오늘 분석 결과 요약해줘")
    print(f"  챗봇: {ans3}\n")

    # ── 메모리 상태 확인 ─────────────────────────────────────────────────────
    bot.show_memory_state()

    print("\n실행 완료")