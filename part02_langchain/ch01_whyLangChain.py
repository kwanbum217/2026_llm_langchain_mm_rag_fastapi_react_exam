# ── 방법 1: LangChain 없이 반복 호출 ───────────────────────────────────────
# > ❌ **이 방법의 문제점 3가지**
# > 
# > 1. **프롬프트 수정의 어려움**: 프롬프트 형식을 바꾸려면 for 루프 안을 찾아서 수정해야 합니다. 프레임이 100개라면 100번 영향을 받습니다.
# > 2. **파싱 코드 중복**: `json.loads(response.choices[0].message.content)` 같은 코드를 매번 써야 합니다.
# > 3. **재사용 불가**: 이 코드는 "CCTV 분석"에만 쓸 수 있습니다. 다른 분석에 쓰려면 처음부터 다시 작성해야 합니다.
# Part 01에서 만든 format_detections 함수 (재사용)
def format_detections(frame_data):
    lines = [f"[{frame_data['timestamp']}] 프레임 #{frame_data['frame_id']}"]
    if not frame_data['detections']:
        lines.append("- 탐지된 객체 없음")
    for d in frame_data['detections']:
        lines.append(
            f"- {d['class']} 탐지 (신뢰도 {d['confidence']:.0%}), "
            f"위치: 좌상단({d['bbox'][0]},{d['bbox'][1]}) "
            f"우하단({d['bbox'][2]},{d['bbox'][3]})"
        )
    return "\n".join(lines)

# 분석할 프레임 목록 (OpenCV로 추출한 결과라고 가정)
frame_results = [
    {
        "frame_id": 1,
        "timestamp": "02:05",
        "detections": [
            {"class": "person", "bbox": [120, 80, 200, 350], "confidence": 0.91},
            {"class": "person", "bbox": [310, 95, 390, 360], "confidence": 0.87},
        ],
    },
    {
        "frame_id": 2,
        "timestamp": "14:30",
        "detections": [
            {"class": "person", "bbox": [150, 90, 230, 360], "confidence": 0.85},
        ],
    },
    # ... 실제로는 100개 이상
]

# ❌ 문제: 프레임마다 수동으로 같은 작업을 반복해야 한다
results = []

for frame in frame_results:
    # 매번 프롬프트를 직접 조립 (코드 중복)
    detections_text = format_detections(frame)
    prompt = f"""당신은 보안 전문가입니다.
다음 CCTV 탐지 결과를 분석하고 위험도를 JSON으로 반환하세요.

{detections_text}
"""
    # 매번 API를 직접 호출 (반복 코드)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    # 매번 응답을 수동으로 파싱 (반복 코드)
    import json
    result = json.loads(response.choices[0].message.content)
    results.append(result)