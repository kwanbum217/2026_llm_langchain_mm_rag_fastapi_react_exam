import sys
import os
import json
import argparse
import subprocess
import time
import warnings

# Windows 인코딩(CP949) 환경에서 이모지/한글 출력 에러(UnicodeEncodeError) 방지
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

# 절대경로 설정으로 실행 경로에 구애받지 않도록 견고화 (공백 포함 정확한 파일명 지정)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(_SCRIPT_DIR, "waves", "20260526_All_units .wav")
STT_JSON_PATH = os.path.join(_SCRIPT_DIR, "stt_temp.json")
DIARIZATION_JSON_PATH = os.path.join(_SCRIPT_DIR, "diarization_temp.json")

# 설정
MODEL_NAME = "tiny.en"            # 영어 무전 전용 초경량(39MB) 영어 모델로 OOM 완벽 방지 튜닝

# 화자 이름 매핑
# pyannote는 SPEAKER_00, SPEAKER_01 ... 으로 반환합니다.
# 코드 실행 후 "감지된 화자 구간" 출력을 보고 순서에 맞게 조정하세요.
SPEAKER_NAMES = {
    "SPEAKER_00": "민욱 (Control)",
    "SPEAKER_01": "주완 (Unit 3)",
    "SPEAKER_02": "대진 (Unit 2)",
    "SPEAKER_03": "다은 (Unit 5)",
}

# ─────────────────────────────────────────────────────────────
# 1단계: 독립된 Whisper STT 프로세스 실행 함수
# ─────────────────────────────────────────────────────────────
def run_stt_step():
  print("=" * 60)
  print("SUBPROCESS STEP 1: Whisper STT (독립 프로세스)")
  print("=" * 60)

  # 지연 임포트(Lazy Import)로 메인 메모리 적재 최소화
  import torch
  import whisper  # type: ignore

  # OpenBLAS 메모리 할당 폭증 방지를 위해 스레드 수를 2개로 조정하여 가용량 확보
  torch.set_num_threads(2)
  print("[INFO] PyTorch CPU 스레드 개수가 2개로 최적화되었습니다 (메모리 가용성 극대화).")

  print("[INFO] Whisper STT 모델 로딩 중...")
  whisper_model = whisper.load_model(MODEL_NAME)

  print("[INFO] Whisper STT 음성 인식 시작...")
  with torch.no_grad():
    result = whisper_model.transcribe(
        AUDIO_PATH,
        language="en",
        task="transcribe",
        initial_prompt=(
            "Security radio communication. "
            "Units responding to suspicious activity at parking structure."
        ),
        verbose=False,
        fp16=False,
    )

  # 결과 중 세그먼트와 텍스트 매핑에 필요한 핵심 정보만 추출하여 저장
  saved_result = {
      "text": result.get("text", ""),
      "segments": []
  }
  for seg in result.get("segments", []):
    saved_result["segments"].append({
        "start": seg.get("start", 0.0),
        "end": seg.get("end", 0.0),
        "text": seg.get("text", "")
    })

  with open(STT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(saved_result, f, ensure_ascii=False, indent=2)

  print(f"[OK] STT 완료 — 임시 결과 저장됨: {STT_JSON_PATH}")


# ─────────────────────────────────────────────────────────────
# 2단계: 독립된 pyannote 화자 분리 프로세스 실행 함수
# ─────────────────────────────────────────────────────────────
def run_diarization_step():
  print("=" * 60)
  print("SUBPROCESS STEP 2: pyannote 화자 분리 (독립 프로세스)")
  print("=" * 60)

  import gc
  import traceback
  from dotenv import load_dotenv
  load_dotenv()
  HF_TOKEN = os.getenv("HF_TOKEN")

  import torch
  import soundfile as sf  # type: ignore
  from pyannote.audio import Pipeline  # type: ignore

  # OpenBLAS 메모리 할당 폭증 방지를 위해 스레드 수를 2개로 조정
  torch.set_num_threads(2)
  print("[INFO] PyTorch CPU 스레드 개수가 2개로 최적화되었습니다 (메모리 가용성 극대화).")

  print("[INFO] pyannote 화자 분리 파이프라인 로딩 중...")
  diarization_pipeline = Pipeline.from_pretrained(
      "pyannote/speaker-diarization-3.1",
      token=HF_TOKEN
  )

  # [밸런스 튜닝] 배치 연산 크기를 8로 조정하여 메모리 안정성과 속도 밸런스 유지
  diarization_pipeline.embedding_batch_size = 8
  diarization_pipeline.segmentation_batch_size = 8
  print("[INFO] pyannote 배치 사이즈 밸런스 튜닝 완료 (Embedding/Segmentation Batch: 8)")

  # 파이프라인 로딩 후 메모리 단편화 명시 해제
  gc.collect()

  # ── 오디오 로딩: numpy 배열 해제로 메모리 최적화 ──────────────────
  # torch.tensor()는 numpy 배열과 메모리를 공유하지 않는 독립 복사본을 생성합니다.
  # 변환 즉시 numpy 배열을 del + gc.collect()하여 메모리를 절반으로 줄입니다.
  print("오디오 파일 로딩 및 리샘플링 중...")
  waveform_np, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
  print(f"[INFO] 오디오 로드 완료: {sample_rate}Hz, {waveform_np.shape}")

  # numpy → torch 독립 복사본 생성 후 numpy 즉시 해제
  waveform_tensor = torch.tensor(waveform_np)   # from_numpy와 달리 독립 복사
  del waveform_np
  gc.collect()

  # (channel, time) 형태로 정렬
  if waveform_tensor.dim() == 1:
    waveform_tensor = waveform_tensor.unsqueeze(0)   # 모노: (time,) → (1, time)
  elif waveform_tensor.shape[0] > waveform_tensor.shape[1]:
    waveform_tensor = waveform_tensor.T              # (time, ch) → (ch, time)

  # 16000Hz 다운샘플링 (pyannote 최적 주파수)
  if sample_rate != 16000:
    print(f"[INFO] 리샘플링: {sample_rate}Hz → 16000Hz")
    new_len = int(waveform_tensor.shape[1] * (16000 / sample_rate))
    resampled = torch.nn.functional.interpolate(
        waveform_tensor.unsqueeze(0),   # (ch, time) → (1, ch, time)
        size=new_len,
        mode="linear",
        align_corners=False
    ).squeeze(0)                         # (1, ch, time) → (ch, time)
    del waveform_tensor
    gc.collect()
    waveform_tensor = resampled
    sample_rate = 16000

  pyannote_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

  # 화자 분리 알고리즘 실행
  print("화자 분리(Diarization) 연산을 수행하고 있습니다 (약 20~40초 소요)...")
  try:
    with warnings.catch_warnings():
      warnings.filterwarnings(
          "ignore",
          message=r"std\(\): degrees of freedom is <= 0",
          category=UserWarning
      )
      with torch.no_grad():
        diarization_output = diarization_pipeline(  # type: ignore[misc]
            pyannote_input,
            min_speakers=2,
            max_speakers=4
        )
  except Exception:
    print("\n[ERROR] 화자 분리 연산 중 예외 발생 — 상세 에러:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

  # 텐서 해제
  del waveform_tensor, pyannote_input
  gc.collect()

  # pyannote 4.x / 3.x 라이브러리 반환 객체 호환성 바인딩
  if hasattr(diarization_output, "speaker_diarization"):
    annotation = diarization_output.speaker_diarization
  else:
    annotation = diarization_output

  # 타임라인 정보를 직렬화 가능한 리스트로 추출
  turns_list = []
  for turn, _, speaker in annotation.itertracks(yield_label=True):
    turns_list.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

  with open(DIARIZATION_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(turns_list, f, ensure_ascii=False, indent=2)

  print(f"[OK] 화자 분리 완료 — 임시 결과 저장됨: {DIARIZATION_JSON_PATH}")



# ─────────────────────────────────────────────────────────────
# 3단계: Whisper 타임스탬프 보정 함수
# ─────────────────────────────────────────────────────────────
def correct_whisper_timestamps(segments: list, audio_path: str) -> tuple:
    """
    Whisper 타임스탬프 보정 함수.

    Windows에서 ffmpeg 없이 실행하면 Whisper가 오디오를
    16000Hz로 리샘플링하지 못해서 타임스탬프가 뻥튀기됩니다.
    예) 44100Hz WAV → 타임스탬프가 실제의 2.75배 (44100/16000)

    실제 오디오 길이 기준으로 전체 비율을 맞춰 보정합니다.
    실제 길이와 1초 이내 차이면 보정 없이 원본 반환합니다.

    반환값: (보정된 segments, 보정 비율)
    pyannote annotation 보정에도 동일 비율을 사용합니다.
    """
    import soundfile as sf  # type: ignore

    # 실제 오디오 길이를 soundfile로 측정 (ffmpeg 불필요)
    info = sf.info(audio_path)
    actual_duration = info.duration  # 초 단위 실제 재생 시간

    if not segments:
        return segments, 1.0

    # Whisper가 인식한 마지막 타임스탬프 (뻥튀기된 값)
    whisper_last_ts = max(float(seg.get("end", 0.0)) for seg in segments)

    if whisper_last_ts <= 0:
        return segments, 1.0

    # 실제 길이와 Whisper 인식 길이 차이가 1초 이내면 보정 불필요
    if abs(whisper_last_ts - actual_duration) <= 1.0:
        print(f"[INFO] 타임스탬프 보정 불필요 (Whisper: {whisper_last_ts:.2f}s ≈ 실제: {actual_duration:.2f}s)")
        return segments, 1.0

    # 보정 비율 계산: actual / whisper (< 1이면 타임스탬프 축소 보정)
    ratio = actual_duration / whisper_last_ts
    print(f"[INFO] Whisper 타임스탬프 보정 수행")
    print(f"       실제 오디오 길이 : {actual_duration:.2f}s")
    print(f"       Whisper 인식 길이: {whisper_last_ts:.2f}s")
    print(f"       보정 비율        : {ratio:.4f} (÷{1/ratio:.4f})")

    corrected = []
    for seg in segments:
        corrected.append({
            **seg,
            "start": float(seg.get("start", 0.0)) * ratio,
            "end":   float(seg.get("end",   0.0)) * ratio,
        })
    return corrected, ratio


# ─────────────────────────────────────────────────────────────
# 4단계: 세그먼트 중간값 기반 화자 조회 함수
# ─────────────────────────────────────────────────────────────
def get_speaker_at(seg_start: float, seg_end: float,
                   diarization_data: list) -> str:
    """
    Whisper 세그먼트의 중간 시점(mid)이 속하는 화자를 반환합니다.

    중간 시점을 쓰는 이유:
    Whisper 세그먼트와 pyannote 구간의 경계가 정확히 일치하지 않습니다.
    중간값을 쓰면 경계 오류를 줄일 수 있습니다.

    끝점을 열린 구간(<)으로 처리하는 이유:
    [0.0, 4.5), [4.5, 8.2) 처럼 경계에서 맞닿을 때
    mid == 4.5 이면 두 구간 모두 매칭될 수 있습니다.
    < 로 처리하면 중복 매핑을 방지합니다.
    """
    mid = (seg_start + seg_end) / 2.0

    # 1차: mid 가 속하는 pyannote 구간 탐색 — 열린 끝점 [start, end) 적용
    for turn in diarization_data:
        t_start = float(turn.get("start", 0.0))
        t_end   = float(turn.get("end",   0.0))
        if t_start <= mid < t_end:
            return str(turn.get("speaker", "UNKNOWN"))

    # 2차 폴백: mid 가 어느 구간에도 속하지 않으면 (무음·갭)
    # 세그먼트 중간값과 각 구간 중심 사이의 거리가 가장 가까운 화자 반환
    best_speaker = "UNKNOWN"
    min_dist = float("inf")
    for turn in diarization_data:
        t_start  = float(turn.get("start", 0.0))
        t_end    = float(turn.get("end",   0.0))
        t_center = (t_start + t_end) / 2.0
        dist = abs(mid - t_center)
        if dist < min_dist:
            min_dist = dist
            best_speaker = str(turn.get("speaker", "UNKNOWN"))

    return best_speaker


# ─────────────────────────────────────────────────────────────
# 메인 제어 프로세스 (인자 분기 및 최종 정렬 매핑)
# ─────────────────────────────────────────────────────────────
def main():
  parser = argparse.ArgumentParser(description="Subprocess 기반 메모리 최적화 화자 분리 파이프라인")
  parser.add_argument("--step", type=str, choices=["stt", "diarization"], default=None,
                      help="특정 서브 프로세스 단계 강제 실행용 내부 인자")
  args = parser.parse_args()

  if args.step == "stt":
    run_stt_step()
    return
  elif args.step == "diarization":
    run_diarization_step()
    return

  # ─── 메인 엮기 코디네이터 시작 ───
  print("=" * 60)
  print("[SYSTEM] Subprocess 기반 화자 분리 파이프라인 통합 제어 시작")
  print("=" * 60)
  print("[INFO] 대용량 AI 모듈을 별개의 격리된 프로세스로 순차 구동하여")
  print("       물리 메모리(8GB) 고갈 현상을 원천 방지합니다.\n")

  # 1. 1단계: Whisper STT 서브프로세스 실행
  start_time = time.time()
  print("[SYSTEM] STEP 1: Whisper STT 서브프로세스를 가동합니다...")
  stt_proc = subprocess.run(
      [sys.executable, "-Xutf8", __file__, "--step", "stt"],
      capture_output=False
  )
  if stt_proc.returncode != 0:
    print("[ERROR] STT 서브프로세스 실행 중 오류가 발생했습니다.", file=sys.stderr)
    sys.exit(stt_proc.returncode)
  
  # 2. 2단계: Pyannote 화자 분리 서브프로세스 실행
  # 앞의 프로세스가 죽었으므로 Whisper 메모리는 OS에 의해 100% 반환된 완벽한 상태입니다.
  print("\n[SYSTEM] STEP 2: pyannote 화자 분리 서브프로세스를 가동합니다...")
  diar_proc = subprocess.run(
      [sys.executable, "-Xutf8", __file__, "--step", "diarization"],
      capture_output=False
  )
  if diar_proc.returncode != 0:
    print("[ERROR] 화자 분리 서브프로세스 실행 중 오류가 발생했습니다.", file=sys.stderr)
    sys.exit(diar_proc.returncode)


  # 3. 3단계: 임시 결과 파일 복원 및 최종 정렬 매핑 (Alignment)
  print("\n" + "=" * 60)
  print("STEP 3: 화자별 전사 텍스트 매핑 (Alignment)")
  print("=" * 60)

  if not os.path.exists(STT_JSON_PATH) or not os.path.exists(DIARIZATION_JSON_PATH):
    print("[ERROR] 임시 결과 파일이 존재하지 않아 최종 매핑을 수행할 수 없습니다.", file=sys.stderr)
    sys.exit(1)

  with open(STT_JSON_PATH, "r", encoding="utf-8") as f:
    stt_data = json.load(f)
  with open(DIARIZATION_JSON_PATH, "r", encoding="utf-8") as f:
    diarization_data = json.load(f)

  # ── Whisper 타임스탬프 보정 (ffmpeg 없는 Windows 환경의 Hz 팽창 교정) ──
  corrected_segments, correction_ratio = correct_whisper_timestamps(
      stt_data["segments"], AUDIO_PATH
  )
  stt_data["segments"] = corrected_segments

  # pyannote 타임스탬프도 동일 비율로 보정 (이미 16000Hz 기준이지만 safety guard)
  if abs(correction_ratio - 1.0) > 0.001:
    print(f"[INFO] pyannote 타임스탬프도 동일 비율({correction_ratio:.4f})로 보정합니다.")
    diarization_data = [
        {
            **turn,
            "start": float(turn.get("start", 0.0)) * correction_ratio,
            "end":   float(turn.get("end",   0.0)) * correction_ratio,
        }
        for turn in diarization_data
    ]

  print(f"[OK] STT 완료 — 세그먼트 수: {len(stt_data['segments'])}개")
  print(f"   전체 텍스트 미리보기: {stt_data['text'][:80]}...\n")

  # [DEBUG] pyannote 원본 화자 구간 출력 (뭉개짐 여부 진단용)
  print(f"[DEBUG] pyannote 화자 구간 총 {len(diarization_data)}개:")
  for t in diarization_data:
    spk_name = SPEAKER_NAMES.get(t['speaker'], t['speaker'])
    print(f"  [{t['start']:.2f}s -> {t['end']:.2f}s] {spk_name}")
  print()

  # 중간값(mid) 기반 화자 매핑: get_speaker_at() 사용
  # 경계 중복 매핑 방지(열린 끝점)와 갭 구간 폴백이 함수 내부에 포함되어 있습니다.
  speaker_logs = []
  for segment in stt_data["segments"]:
    seg_start = float(segment.get("start", 0.0))
    seg_end   = float(segment.get("end",   0.0))
    seg_text  = str(segment.get("text",   "")).strip()

    best_speaker = get_speaker_at(seg_start, seg_end, diarization_data)
    speaker_name = SPEAKER_NAMES.get(best_speaker, best_speaker)

    speaker_logs.append({
        "start":   seg_start,
        "end":     seg_end,
        "speaker": speaker_name,
        "text":    seg_text
    })

  print("[OK] 화자별 전사 텍스트 매핑 완료:")
  for log in speaker_logs:
    print(f"  [{log['start']:.1f}s -> {log['end']:.1f}s] {log['speaker']}: {log['text']}")
  print()

  # 4. 연속된 같은 화자의 발화를 하나로 병합
  merged_segments = merge_consecutive_segments(speaker_logs)

  print("[OK] 화자별 전사 텍스트 병합 완료:")
  for seg in merged_segments:
    print(f"  [{seg['speaker']}] ({seg['start']:.1f}s ~ {seg['end']:.1f}s)\n  {seg['text']}")
  print()

  # 5. 전사 결과를 텍스트 파일로 저장
  transcript_path = AUDIO_PATH.replace(".wav", "_transcript.txt")
  try:
    with open(transcript_path, "w", encoding="utf-8") as f:
      f.write("=== 화자별 대화 전사 ===\n\n")
      for seg in merged_segments:
        f.write(f"[{seg['speaker']}] ({seg['start']:.1f}s ~ {seg['end']:.1f}s)\n")
        f.write(f"{seg['text']}\n\n")
    print(f"[FILE] 전사 파일 저장: {transcript_path}\n")
  except Exception as e:
    print(f"[WARNING] 전사 파일 저장 실패: {e}")

  # 6. 임시 파일 삭제 및 메모리 청소
  try:
    if os.path.exists(STT_JSON_PATH):
      os.remove(STT_JSON_PATH)
    if os.path.exists(DIARIZATION_JSON_PATH):
      os.remove(DIARIZATION_JSON_PATH)
    print("[SYSTEM] 임시 결과 파일 캐시를 안전하게 폐기했습니다.")
  except Exception as e:
    print(f"[WARNING] 임시 캐시 삭제 실패: {e}")

  print(f"[SYSTEM] 전체 연산이 완료되었습니다! (총 소요 시간: {time.time() - start_time:.1f}초)")

def merge_consecutive_segments(segment_with_speaker: list) -> list:
  """
    연속된 같은 화자의 발화를 하나로 합칩니다.
    gap_threshold: 0.8초 이내 같은 화자 → 연속 발화로 판단
  """
  merged_list = []

  # 리스트가 비어있으면 그대로 반환
  if not segment_with_speaker:
    return merged_list

  # 현재 합쳐지고 있는 구간 초기화
  current_speaker = None
  current_start = 0.0
  current_end = 0.0
  current_text_parts = []

  # 순회하며 합치기
  for entry in segment_with_speaker:
    spk = entry.get("speaker")
    start = entry.get("start", 0.0)
    end = entry.get("end", 0.0)
    text = entry.get("text", "")

    # 화자가 동일하고, 시간적으로 바로 연속되거나 0.8초 이내 갭인 경우
    if spk == current_speaker and start <= current_end + 0.8:
      current_end = max(current_end, end)
      if text:
        current_text_parts.append(text)
    else:
      # 새로운 화자 등장 또는 구간 분리
      # 이전 구간이 있다면 먼저 저장
      if current_speaker is not None:
        merged_list.append({
            "start": current_start,
            "end": current_end,
            "speaker": current_speaker,
            "text": " ".join(current_text_parts).strip()
        })
      
      # 현재 구간 갱신
      current_speaker = spk
      current_start = start
      current_end = end
      current_text_parts = [text] if text else []

  # 마지막 구간 저장
  if current_speaker is not None:
    merged_list.append({
        "start": current_start,
        "end": current_end,
        "speaker": current_speaker,
        "text": " ".join(current_text_parts).strip()
    })

  return merged_list

if __name__ == "__main__":
  main()



def merge_consecutive_segments(segment_with_speaker : list) -> list:
  """
    연속된 같은 화자의 발화를 하나로 합칩니다.

    예시:
    [민욱] "Subject attempted access."  7.8s ~ 10.0s
    [민욱] "Access denied three times." 10.1s ~ 12.0s
    →
    [민욱] "Subject attempted access. Access denied three times." 7.8s ~ 12.0s

    gap_threshold: 0.8초 이내 같은 화자 → 연속 발화로 판단
  """

  # 만일  스피커0 의 화자와 스키퍼 1의 화자의 이름이 같다면 합치고, 아니면 스피커1화자로 분류한다 -> 조건은 0.8초 이내 같은 화자로  연속 발화로 판단
  # if not segment_with_speaker :
  #   return []
  
  # gap_threshold = 0.8
  # merged = [segment_with_speaker[0].copy()]

  # for seg in segment_with_speaker[1:] :
  #   prev = merged[-1]
  #   gap = seg['start'] - prev['end'] 
  #   if seg["speaker"] == prev["speaker"] and gap < gap_threshold :
  #     prev["text"] += " " + seg["text"].strip()
  #     prev["end"] = seg["end"]
  #   else:
  #     merged.append(seg.copy())

  # return merged



  merged_list = []

  # 리스트가 비어있으면 그대로 반환
  if not speaker_logs:
    return merged_list

  # 현재 합쳐지고 있는 구간 초기화
  current_speaker = None
  current_start = 0.0
  current_end = 0.0
  current_text_parts = []

  # 순회하며 합치기
  for entry in speaker_logs:
    spk = entry.get("speaker")
    start = entry.get("start", 0.0)
    end = entry.get("end", 0.0)
    text = entry.get("text", "")

    # 화자가 동일하고, 시간적으로 바로 연속되거나 살짝 겹치는 경우
    # 즉, 끊김이 없는 자연스러운 말의 흐름인 경우
    if spk == current_speaker and start <= current_end + 0.1:
      # 시간 연장
      current_end = max(current_end, end)
      # 텍스트 추가 (띄어쓰기 유지)
      if current_text_parts and text:
        current_text_parts.append(text)
      elif text:
        current_text_parts.append(text)
    else:
      # 새로운 화자 등장 또는 구간 분리
      # 이전 구간이 있다면 먼저 저장
      if current_speaker is not None:
        merged_list.append({
            "start": current_start,
            "end": current_end,
            "speaker": current_speaker,
            "text": " ".join(current_text_parts).strip()
        })
      
      # 현재 구간 갱신
      current_speaker = spk
      current_start = start
      current_end = end
      current_text_parts = [text] if text else []

  # 마지막 구간 저장
  if current_speaker is not None:
    merged_list.append({
        "start": current_start,
        "end": current_end,
        "speaker": current_speaker,
        "text": " ".join(current_text_parts).strip()
    })

  return merged_list
  