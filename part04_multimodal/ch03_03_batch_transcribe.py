# 여러개의 wav를 배치 처리 하는 방법 습득
# 파일마다 load_model()을 호출하면 매번 모델로딩하는데 수십초가 걸린다.

# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import whisper

def batch_transcribe(audio_dir:str,model_name:str = "base") -> list : 
    """
    폴더 안의 오디오 파일을 모두 변환합니다.
    모델을 한 번만 로드하고 재사용합니다.

    사용 예시:
        results = batch_transcribe("./audio_files", model_name="base")
        for r in results:
            print(r["file"], "→", r["text"][:50])

    반환값 구조:
        [
            {
                "file":     "radio_001.wav",
                "text":     "전체 변환 텍스트",
                "language": "ko",
                "segments": [...]
            },
            ...
        ]
    """
    # 모델을 함수 밖에서 한 번만 로드
    model = whisper.load_model(model_name)
    results = []

    # WAV, MP3, MP4, M4A, FLAC 파일만 필터링
    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac"))
    ]

    for filename in sorted(audio_files):
        filepath = os.path.join(audio_dir, filename)
        print(f"변환 중: {filename}")

        result = model.transcribe(
            filepath,
            fp16=False,
        )

        results.append({
            "file":     filename,
            "text":     result["text"],
            "language": result["language"],
            "segments": result["segments"],
        })
        print(f"  → {result['text'][:60]}")

    return results

batch_transcribe("./waves", model_name="base")