# ─────────────────────────────────────────────────────────────
# 예제 1: LCEL 배치 파이프라인으로 10개 프레임 일괄 분석
#
# 이 예제는 CH02 내용의 복습입니다.
# API 키 불필요 (Mock LLM 사용)
# ─────────────────────────────────────────────────────────────

import json, re, random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

# random.seed(42)

# ── 공통 데이터: 10개 프레임 시뮬레이션 ───────────────────────
LOCATIONS = ["주차장 A", "창고 출입구", "로비", "비상구 복도", "옥상"]

def make_frame_results(n: int = 10) -> list:
    """
    OpenCV가 영상에서 추출했을 프레임 탐지 결과를 시뮬레이션합니다.
    실제 수업에서는 OpenCV cv2.VideoCapture로 생성한 결과를 사용합니다.
    """
    frames = []
    for i in range(1, n + 1):
        n_persons = random.randint(0, 5)
        frames.append({
            "frame_id":  i,
            # 3의 배수 프레임은 새벽 시간 (02:xx) → 위험도 높음
            "timestamp": f"0{2 if i % 3 == 0 else 1}:{i // 60:02d}:{i % 60:02d}",
            "location":  LOCATIONS[i % len(LOCATIONS)],
            "detections": [
                {
                    "class":      "person",
                    "bbox":       [random.randint(0, 400), random.randint(0, 300),
                                   random.randint(100, 500), random.randint(100, 400)],
                    "confidence": round(random.uniform(0.75, 0.98), 2),
                }
                for _ in range(n_persons)
            ],
        })
    return frames


def format_detections(frame: dict) -> str:
    """
    OpenCV 탐지 결과 딕셔너리 → LLM 읽기 좋은 텍스트로 변환.

    Part 01에서 만든 format_detections 함수를 그대로 재활용합니다.
    이처럼 Part 01 코드는 이 강의 전체에서 반복 사용됩니다.
    """
    lines = [f"[{frame['timestamp']}] {frame['location']}"]
    for d in frame["detections"]:
        lines.append(
            f"  - {d['class']} (신뢰도 {d['confidence']:.0%}), "
            f"bbox={d['bbox']}"
        )
    if not frame["detections"]:
        lines.append("  - 탐지된 객체 없음")
    return "\n".join(lines)


# ── Mock LLM (API 키 불필요) ───────────────────────────────────
def mock_llm_fn(prompt_value) -> object:
    """
    실제 ChatOpenAI 대신 사용하는 Mock 함수.

    ChatOpenAI로 교체 시:
    ─────────────────────────────────────────────
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 아래 mock_llm_fn 자리에 llm을 그대로 사용하면 됩니다
    analysis_chain = prompt | llm | json_parser
    ─────────────────────────────────────────────
    """
    # prompt가 렌더링한 메시지에서 human 내용 추출
    human_content = ""
    messages = prompt_value.to_messages()
    for m in messages:
        if hasattr(m, "content") and "frame_id" in str(m.content):
            human_content = str(m.content)
            break

    # frame_id 추출 (정규식)
    fid_match = re.search(r"frame_id:\\s*(\\d+)", human_content)
    frame_id  = int(fid_match.group(1)) if fid_match else 0

    # 위험도 판단 로직 (실제 LLM 역할)
    person_count  = human_content.count("person")
    has_night     = bool(re.search(r"0[123]:\d{2}", human_content))   # 새벽 01~03시
    has_sensitive = "창고" in human_content or "비상구" in human_content

    if person_count >= 2 and has_night:
        risk, reason = "위험", f"심야 {person_count}인 탐지"
    elif person_count >= 1 and has_sensitive:
        risk, reason = "주의", "민감 구역 인원 탐지"
    elif person_count == 0:
        risk, reason = "정상", "탐지 객체 없음"
    else:
        risk, reason = "정상", "일반 탐지"

    response_text = json.dumps({
        "frame_id":     frame_id,
        "risk_level":   risk,
        "person_count": person_count,
        "reason":       reason,
        "action": ("경비팀 즉시 출동" if risk == "위험"
                   else "모니터링 강화" if risk == "주의"
                   else "이상 없음"),
    }, ensure_ascii=False)

    class FakeResponse:
        """ChatOpenAI 응답 객체를 흉내 내는 클래스 (.content 속성 필요)"""
        def __init__(self, text):
            self.content = text

    return FakeResponse(response_text)


# ── LCEL 파이프라인 구성 ────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "CCTV 탐지 결과를 분석해 위험도를 JSON으로 반환하세요. "
     "risk_level은 '정상'/'주의'/'위험' 중 하나."),
    ("human",
     "frame_id: {frame_id}\\n탐지 결과:\\n{detection_text}"),
])

json_parser = JsonOutputParser()

# 파이프라인: prompt → mock_llm → .content 추출 → JSON 파싱
analysis_chain = (
    prompt
    | RunnableLambda(mock_llm_fn)           # Mock LLM (→ 실제는 ChatOpenAI)
    | RunnableLambda(lambda r: r.content)   # .content 문자열 추출
    | json_parser                           # JSON 문자열 → dict
)


# ── 배치 실행 ────────────────────────────────────────────────
frame_results = make_frame_results(10)
print("배치 분석 실행 중...")

results = []
for frame in frame_results:
    result = analysis_chain.invoke({
        "frame_id":       frame["frame_id"],
        "detection_text": format_detections(frame),
    })
    # frame_id 보정 (파이프라인 내에서 손실 방지)
    result["frame_id"] = frame["frame_id"]
    results.append(result)

# 결과 요약
danger  = [r for r in results if r["risk_level"] == "위험"]
caution = [r for r in results if r["risk_level"] == "주의"]
normal  = [r for r in results if r["risk_level"] == "정상"]

print(f"\\n분석 완료: 총 {len(results)}개 프레임")
print(f"  정상 {len(normal)}건 | 주의 {len(caution)}건 | 위험 {len(danger)}건")
if danger:
    print(f"  위험 프레임 ID: {[r['frame_id'] for r in danger]}")

print("\\n첫 번째 결과 샘플:")
print(json.dumps(results[0], ensure_ascii=False, indent=2))