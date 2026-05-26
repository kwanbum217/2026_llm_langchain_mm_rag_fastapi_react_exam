# ch03_02_whisper_local.py
# 로컬 Whisper 엔진을 활용한 고속 음성인식(STT) 실습 스크립트
# 오디오 바이너리 로드 -> 로컬 모델 연산 (CPU 기반) -> 텍스트 복원 및 분석

import os
import sys
import shutil
import logging

# Windows 콘솔 한글 깨짐 방지 및 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
if hasattr(sys.stderr, "reconfigure"):
  sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

# [중요 튜닝]: Windows 환경변수 지연 반영 해결 기법
# winget으로 ffmpeg를 갓 설치했을 때, 부모 프로세스를 재시작하지 않으면 환경변수(PATH)가 연동되지 않습니다.
# 이를 방지하기 위해 winget 표준 설치 경로를 동적으로 스캔하여 PATH에 선제 주입해 줍니다.
if not shutil.which("ffmpeg"):
  winget_ffmpeg_dir = r"C:\Users\lucian\AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
  if os.path.exists(winget_ffmpeg_dir):
    # 하위 ffmpeg 폴더 탐색
    for sub in os.listdir(winget_ffmpeg_dir):
      sub_path = os.path.join(winget_ffmpeg_dir, sub)
      if os.path.isdir(sub_path) and sub.startswith("ffmpeg-"):
        bin_path = os.path.join(sub_path, "bin")
        if os.path.exists(bin_path):
          os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
          break

# [중요 경고]: openai-whisper는 내부적으로 ffmpeg 시스템 명령어를 호출하므로,
# 반드시 시스템 환경변수(PATH)에 ffmpeg가 완벽하게 설치 및 등록되어 있어야 구동 오류를 피할 수 있습니다.
try:
  import whisper  # type: ignore
except ModuleNotFoundError as e:
  raise ModuleNotFoundError(
    "[ERROR] 가상환경에 openai-whisper가 설치되어 있지 않습니다. "
    "먼저 pip install openai-whisper를 완료해 주십시오."
  ) from e

# 파일 경로 독립성 확보를 위해 실행 위치 기준 절대 경로 계산
current_dir = os.path.dirname(os.path.abspath(__file__))
# 에러에 지목되었던 5초 테스트용 WAV 오디오 경로 지정
# 만약 파일이 존재하지 않는 경우를 대비해 기존의 생성된 radio_normal.wav를 백업으로 설정
AUDIO_PATH = os.path.join(current_dir, "20260526_\u201cAll_units (1).wav")
if not os.path.exists(AUDIO_PATH):
  AUDIO_PATH = os.path.join(current_dir, "radio_normal.wav")

print("[STEP 1] 로컬 Whisper 모델 로딩 시작 (CPU 전용 모드)")
# GeForce 940MX의 2GB VRAM 하드웨어 한계를 고려하여 VRAM 오버헤드(OOM)가 전무하고
# CPU 연산에서도 고속 구동되는 초경량 'tiny' 모델을 로드합니다.
# 인위적인 장치 지정을 통해 GPU 메모리 부족 경고를 원천 차단합니다.
model = whisper.load_model("tiny", device="cpu")
print("로컬 Whisper 'tiny' 모델 로딩 완료")

print(f"\n[STEP 2] 음성 텍스트 변환(STT) 시작: {os.path.basename(AUDIO_PATH)}")
print("  (주의: 시스템 내부적으로 ffmpeg를 구동해 오디오 디코딩을 전개합니다)")

try:
  result = model.transcribe(
    AUDIO_PATH,
    temperature=0.0,
    language="en",    # 한국어 음성인식 강제 지정
    fp16=False        # CPU 연산 환경이므로 float16 정밀도 가속 비활성화 (False 강제)
  )
except FileNotFoundError as e:
  if "ffmpeg" in str(e).lower() or "[winerror 2]" in str(e).lower():
    raise RuntimeError(
      "[인프라 결함] Windows 시스템에 ffmpeg가 설치되어 있지 않아 오디오를 읽을 수 없습니다.\n"
      "  -> 조치 방법: 관리자 권한 터미널에서 아래 winget 명령으로 ffmpeg를 신속히 자동 설치하십시오:\n"
      "     winget install BtbN.FFmpeg.GPL.Shared --silent --accept-source-agreements --accept-package-agreements\n"
      "     설치 완료 후 터미널 창을 완전히 새로 열어야 환경변수(PATH)가 시스템에 갱신 적용됩니다."
    ) from e
  raise e

print("\n[STEP 3] 음성 인식 텍스트 추출 성공")
print("=" * 60)
print(f"인식된 텍스트:\n{result.get('text', '').strip()}")
print("=" * 60)

# 세부 시간별 자막(Segment) 데이터 매핑 루프
print("\n[STEP 4] 시간 정보(Timeline) 세부 분석")
segments = result.get("segments", [])
for s in segments:
  start = s.get("start", 0.0)
  end = s.get("end", 0.0)
  text = s.get("text", "").strip()
  print(f"  [{start:05.2f}s -> {end:05.2f}s] {text}")

# 세그먼트별 상세 결과
# 세그먼트: Whisper가 문장 단위로 끊어서 반환하는 타임스탬프 정보
print("[세그먼트별 타임스탬프]")
for seg in result["segments"]:
  start_t = seg["start"]
  end_t   = seg["end"]
  text    = seg["text"].strip()
  prob    = seg["avg_logprob"]    # 신뢰도 (낮을수록 불확실)
  no_sp   = seg["no_speech_prob"] # 무음 확률

# avg_logprob 기준 신뢰도 표시
# -0.5 이상  → 신뢰할 수 있는 인식 결과
# -0.5 ~ -1.0 → 주의 필요
# -1.0 이하  → 노이즈나 불명확한 발화일 가능성 높음
  confidence = "✅" if prob > -0.5 else ("⚠️" if prob > -1.0 else "❌")
  print(f"  {confidence} [{start_t:5.1f}s → {end_t:5.1f}s] {text}")
  print(f"       신뢰도: {prob:.3f} | 무음확률: {no_sp:.3f}")

print()

# ─────────────────────────────────────────────────────────
# STEP 4: LangChain으로 텍스트 요약
# 변환된 텍스트를 GPT-4o에게 넘겨 보안 보고 형식으로 요약
# ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("=" * 60)
print("🤖 GPT-4o 무전 내용 요약")
print("=" * 60)

prompt = PromptTemplate.from_template(
    """당신은 보안 관제 요원입니다.
아래 무전 교신 내용을 보안 보고서 형식으로 요약하세요.

[무전 교신 원문]
{transcription}

다음 항목을 포함해서 요약하세요:
- 상황 요약 (1~2문장)
- 탐지 내용 (인원, 위치, 시간)
- 조치 사항
- 후속 대응 필요 여부"""
)

llm   = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | llm | StrOutputParser()

summary = chain.invoke({"transcription": result["text"]})
print(f"\n{summary}")
print("\n✅ Whisper 로컬 변환 + LangChain 요약 완료!")