import wave  # wave 파일 읽기/쓰기
import struct  # 숫자 -> 바이너리 변환
import math
import os
from openai import OpenAI


def make_test_wav(
  path: str,
  duration_sec: float = 2.0,
  sample_rate: int = 16000  
):
  """
  순수 Python으로 실습용 WAV 파일 생성.

  Args:
      path         : 저장 경로 (예: "radio_transmission.wav")
      duration_sec : 파일 길이 (초)
      sample_rate  : 샘플레이트 (Whisper 권장: 16000Hz)

  Returns:
      생성된 파일 경로
  """
  # 전체 샘플 수 = 초당 샘플 수 × 길이
  # 16000Hz, 3초 -> 48,000개 샘플
  n_samples = int(sample_rate * duration_sec)

  with wave.open(path, "w") as wf:
    wf.setnchannels(1)          # 모노 (1채널) — Whisper는 모노 권장
    wf.setsampwidth(2)          # 16-bit PCM (샘플 1개당 2바이트)
    wf.setframerate(sample_rate) # 초당 샘플 수

    for i in range(n_samples):
      t = i / sample_rate     # 현재 시간 (초)

      # 여러 주파수를 합성해 사람 목소리와 유사한 파형 생성
      # 실제 목소리는 100~300Hz 기본 주파수 + 배음(harmonic) 구조
      val = (
        0.4 * math.sin(2 * math.pi * 200  * t) +  # 기본 주파수
        0.3 * math.sin(2 * math.pi * 400  * t) +  # 2배음
        0.2 * math.sin(2 * math.pi * 800  * t) +  # 4배음
        0.1 * math.sin(2 * math.pi * 1600 * t)    # 8배음
      )

      # 부동소수점 [-1, 1] -> 16-bit PCM [-32768, 32767] 범위로 변환
      sample = int(val * 32767 * 0.8)             # 0.8: 약간 여유 두기
      sample = max(-32768, min(32767, sample))    # 범위 클리핑

      # "<h": 리틀엔디안 16-bit 정수 형식으로 패킹
      wf.writeframes(struct.pack("<h", sample))

  size_bytes = os.path.getsize(path)
  print(f"✅ 파일 생성 완료: {path}")
  print(f"   샘플레이트: {sample_rate}Hz")
  print(f"   길이: {duration_sec}초")
  print(f"   크기: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
  return path


# ── 실행 ─────────────────────────────────────────────────────────
# 실습용 파일 3가지 생성 (시나리오별)
make_test_wav("radio_normal.wav",    duration_sec=2.0)  # 정상 상황
make_test_wav("radio_warning.wav",   duration_sec=3.0)  # 주의 상황
make_test_wav("radio_emergency.wav", duration_sec=4.0)  # 긴급 상황
