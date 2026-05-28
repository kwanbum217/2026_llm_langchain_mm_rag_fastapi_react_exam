# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# ── 상황 1: LLM이 깔끔한 JSON을 반환했을 때 ────────────────────────────────
llm_response_clean = '{"risk_level": "위험", "reason": "심야 2인 탐지", "action": "경비팀 출동"}'

# Patser없이 직접 파싱한다면...
result_manual = json.loads(llm_response_clean)

print("수동 파싱 타입:", type(result_manual))    # <class 'dict'>
print("수동 파싱 결과:", result_manual)

# ── 상황 2: LLM이 마크다운 코드블록으로 감싸서 반환했을 때 ──────────────────
# GPT-4o는 간혹 JSON을 ```json ... ``` 형식으로 감싸서 줍니다
llm_response_markdown = """```json
{"risk_level": "주의", "reason": "낮 시간대 1인", "action": "모니터링"}
```"""
# 이 코드에서는 에러가 날 수밖에 없는데 문자열에 백틱을 비롯한 json이라는 문자열이 포함 되어 있기 때문이다.
# 일반 json 내장 모듈로 파싱하면 에러가 난다
try :
  result_manual = json.loads(llm_response_markdown)
except json.JSONDecodeError as e:
  print(f"수동 JSON 파싱 에러: {e}")


# 위 md 형식의 응답을 JsonOutputParser로 파싱
json_parser = JsonOutputParser()
result_auto = json_parser.parse(llm_response_markdown)  # 파이썬 딕셔너리 객체로 변환
print(f"JsonOutputParser 파싱 타입: {type(result_auto)}")
print(f"위험레벨: {result_auto['risk_level']}")