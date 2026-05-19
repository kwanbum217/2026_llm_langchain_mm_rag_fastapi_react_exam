# ─────────────────────────────────────────────────────────────
# 예제 2: @tool 데코레이터로 세 가지 Tool 등록 및 직접 호출
#
# Tool은 LLM이 호출하지만, .invoke()로 직접 호출해서
# 동작을 먼저 확인할 수 있습니다.
# ─────────────────────────────────────────────────────────────

import json, random, re
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

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


# Tool 1 : 위험한 프레임을 필터링
@tool  #LLM 스스로가 판단하여 이 함수를 호출 할 수 있도록...
def filter_danger_frames(frames_json: str) -> str:
    """
    분석 결과 리스트에서 위험(risk_level='위험') 프레임만 필터링합니다.

    CCTV 분석 결과에서 즉각 조치가 필요한 위험 이벤트만 추출할 때 사용합니다.

    Args:
        frames_json: 분석 결과 리스트를 JSON 문자열로 전달
                     예: '[{"frame_id":3,"risk_level":"위험",...}]'
    Returns:
        위험 프레임만 포함한 JSON 문자열 (빈 리스트 가능)
    """
    frames = json.loads(frames_json)  # JSON 파싱
    danger = [f for f in frames if f.get("risk_level") == "위험"]  # 위험만 필터링
    return json.dumps(danger, ensure_ascii=False, indent=2)  # JSON 문자열로 반환


# ── Tool 2: 특정 구역 객체 카운트 ────────────────────────────
@tool
def count_objects_in_zone(frames_json: str, zone: str) -> str:
    """
    특정 구역(zone)에서 탐지된 객체 수를 카운트합니다.

    구역별 보안 밀도를 확인하거나 특정 위치의 이상 여부를 판단할 때 사용합니다.

    Args:
        frames_json: 프레임 탐지 결과 리스트 JSON 문자열
                     (frame_results의 원본 데이터 사용)
        zone: 카운트할 구역 이름 (예: '창고 출입구', '주차장 A', '로비')
    Returns:
        구역명, 총 탐지 수, 프레임 수, 평균을 담은 JSON 문자열
    """
    frames = json.loads(frames_json)

    # zone과 일치하는 프레임만 추출
    matched = [f for f in frames if f.get("location", "") == zone]

    # 해당 구역의 전체 탐지 객체 수 합산
    total = sum(len(f.get("detections", [])) for f in matched)

    return json.dumps({
        "zone":            zone,
        "total_detections": total,
        "frame_count":      len(matched),
        # 프레임이 없으면 0 (ZeroDivisionError 방지)
        "avg_per_frame":    round(total / len(matched), 2) if matched else 0,
    }, ensure_ascii=False)
  
# ── Tool 3: 전체 위험도 요약 ─────────────────────────────────
@tool
def get_risk_summary(results_json: str) -> str:
    """
    전체 분석 결과의 위험도 요약 통계를 반환합니다.

    운영자에게 현재 상황을 한눈에 보여줄 때 사용합니다.

    Args:
        results_json: 분석 결과 리스트 JSON 문자열
    Returns:
        정상/주의/위험 카운트와 위험 프레임 ID 목록을 담은 JSON 문자열
    """
    data    = json.loads(results_json)
    summary = {"정상": 0, "주의": 0, "위험": 0, "위험_프레임_ids": []}

    for r in data:
        lvl = r.get("risk_level", "정상")
        summary[lvl] = summary.get(lvl, 0) + 1       # 위험도별 카운트
        if lvl == "위험":
            summary["위험_프레임_ids"].append(r.get("frame_id"))  # 위험 프레임 ID 수집

    return json.dumps(summary, ensure_ascii=False)

# ── Tool 메타데이터 확인 ─────────────────────────────────────
tools = [filter_danger_frames, count_objects_in_zone, get_risk_summary]

print("[등록된 Tool 목록]")
for t in tools:
    # Tool 이름과 설명 첫 줄 출력
    first_line = t.description.splitlines()[0]
    print(f"  [{t.name}]")
    print(f"    설명: {first_line}")
    print(f"    입력: {list(t.args.keys())}")


results_json = json.dumps(results, ensure_ascii=False, indent=2)
frames_json = json.dumps(frame_results, ensure_ascii=False, indent=2)

print('\n Tool 직접 호출 테스트 =================================' )

danger_list = json.loads(
  filter_danger_frames.invoke({"frames_json" : results_json})
)

print(f"\n1. filter_danger_frames")
print(f"   위험 프레임 {len(danger_list)}건: {[r['frame_id'] for r in danger_list]}")

# tool2 호출 - 창고 출입구 카운트
zone_stat = json.loads(
    count_objects_in_zone.invoke({
        "frames_json": frames_json, 
        "zone":        "창고 출입구"
    })
)

print(f"\n2. count_objects_in_zone ('창고 출입구')")
print(f"   총 탐지: {zone_stat['total_detections']}건 "
      f"(프레임 {zone_stat['frame_count']}개, 평균 {zone_stat['avg_per_frame']}/프레임)")

# tool 3 호출 - 요약 통계
summary_stat = json.loads(
  get_risk_summary.invoke({"results_json" : results_json})
)

print(f"\n3. get_risk_summary")
print(f"   정상: {summary_stat['정상']}건 | "
      f"주의: {summary_stat['주의']}건 | "
      f"위험: {summary_stat['위험']}건")
print(f"   위험 ID: {summary_stat['위험_프레임_ids']}")
