from langchain_core.runnables import RunnableLambda


def format_detections(frame_data: dict) -> dict:
    """
    OpenCV 탐지 결과 딕셔너리를 ChatPromptTemplate 입력용 딕셔너리로 변환합니다.

      변경 이유:
          Part 01에서는 문자열 하나를 반환했습니다.
          Part 02 PromptTemplate은 4개 변수(frame_id, timestamp, location,
          detections_text)를 각각 받아야 하므로 딕셔너리를 반환하도록 수정합니다.

      Args:
          frame_data (dict): OpenCV 탐지 결과
              {
                  "frame_id"  : 프레임 번호 (int),
                  "timestamp" : 촬영 시각 문자열 "HH:MM",
                  "location"  : 촬영 위치 문자열,
                  "detections": [
                      {"class": "person", "bbox": [x1, y1, x2, y2], "confidence": 0.91},
                      ...
                  ]
              }

      Returns:
          dict: ChatPromptTemplate의 4개 변수에 맞는 딕셔너리
              {
                  "frame_id"       : 프레임 번호,
                  "timestamp"      : 촬영 시각,
                  "location"       : 촬영 위치,
                  "detections_text": 탐지 결과 여러 줄 문자열
              }
    """
    detections = frame_data.get("detections", [])
    lines = []
    if not detections:
        lines.append("- 탐지된 객체 없음")
    else:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            width = x2 - x1  # bbox의 가로 크기
            height = y2 - y1  # bbox의 세로 크기
            lines.append(
                f"- {d['class']} 탐지 (신뢰도 {d['confidence']:.0%}), "
                f"위치: 좌상단({x1},{y1}) 우하단({x2},{y2}), "
                f"크기: {width}×{height}px"
            )
    return {
        "frame_id": frame_data["frame_id"],
        "timestamp": frame_data["timestamp"],
        "location": frame_data.get("location", "미지정"),
        "detections_text": "\n".join(lines),
    }
# ── 동작 확인 ─────────────────────────────────────────────────────────────
sample_frame = {
    "frame_id" : 1,
    "timestamp": "02:13",
    "location" : "주차장 A구역",
    "detections": [
        {"class": "person", "bbox": [120, 80, 200, 350], "confidence": 0.91},
        {"class": "car",    "bbox": [50, 200, 280, 400], "confidence": 0.95},
    ],
} 

result = format_detections(sample_frame)

print("반환 타입:", type(result))          # <class 'dict'>
print("키 목록:",  list(result.keys()))    # ['frame_id', 'timestamp', 'location', 'detections_text']
print()
print("detections_text 내용:")
print(result["detections_text"])


# ── RunnableLambda로 랭체인 파이프라인 부품과
formatter = RunnableLambda(format_detections)

# 입력이 딕셔너리여야 함을 명시 (type hint)
test_output = formatter.invoke(sample_frame)
print("test_output:", test_output)
print("keys:", list(test_output.keys()))