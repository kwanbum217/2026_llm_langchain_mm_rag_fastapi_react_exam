# ─────────────────────────────────────────────────────────────
# 예제 3: ReAct 패턴 수동 시뮬레이션
#
# 질문: "위험 프레임 요약 + 창고 출입구 탐지 현황 같이 알려줘"
#
# LLM이 실제로 하는 일:
#   1. 질문 분석 → 어떤 Tool이 필요한지 판단 (Thought)
#   2. Tool 호출 (Action)
#   3. 결과 확인 (Observation)
#   4. 충분한지 판단 → 더 필요하면 다시 1번으로
#   5. 최종 답변 생성 (Final Answer)
# ─────────────────────────────────────────────────────────────

import json, random, re
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser


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


# 예제 2에서 만든 Tool 레지스트리 사용
TOOL_REGISTRY = {
    "filter_danger_frames":    filter_danger_frames,
    "count_objects_in_zone":   count_objects_in_zone,
    "get_risk_summary":        get_risk_summary,
}

def react_step(
  thought: str, #LLM 추론 
  action: str, # 호출해야할 tool 이름
  action_input : dict # tool 에 전달할 파라미터 
) -> str :
  """
    ReAct 한 사이클(Thought → Action → Observation)을 실행합니다.

    실제 Agent에서는 LLM이 thought와 action_name을 자동으로 생성하지만,
    여기서는 우리가 직접 입력해서 흐름을 눈으로 확인합니다.
  """
  print(f"\n  💭 Thought     : {thought}")
  print(f"  🔧 Action      : {action_name}({str(action_input)[:50]}...)") 

  observation = TOOL_REGISTRY[action_name].invoke(action_input)

  
  

  
