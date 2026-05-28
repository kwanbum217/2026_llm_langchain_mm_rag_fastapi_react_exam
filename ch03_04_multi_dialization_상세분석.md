# ch03_04_multi_dialization.py 상세 분석 보고서

**프로젝트 학습 기간**: 2025.12.30 ~ 2026.07.28 (BiNCS AI·Data Engineer 양성과정)  
**학습 모듈**: ch03 멀티모달 AI (Multimodal AI)  
**핵심 기술**: 음성 인식(STT) + 화자 분리(Speaker Diarization) 파이프라인  
**목적**: 보안 무전(Radio Communication) 오디오를 자동으로 화자별 전사(Transcript)하는 실전 모듈 개발

이 문서는 코드의 **명세서(스펙)**를 기준으로, 모든 변수·함수·흐름을 상세히 분석하여 작성되었습니다.  
학습자가 "내가 배우는 내용"을 완벽히 이해할 수 있도록 **왜 이 기술을 사용하는지**, **각 단계의 역할**, **실무 적용 이유**를 함께 설명합니다.

---

## 1. 현재 코드의 기본 개요 (이 코드가 무엇을 하는가, 왜 존재하는가)

### 1.1 전체 목적
이 코드는 **멀티모달 AI 파이프라인**의 전형적인 예시입니다.  
단순한 음성→텍스트 변환(STT)을 넘어, **"누가 언제 무슨 말을 했는지"**까지 자동으로 식별하여 구조화된 전사본을 생성합니다.

- **입력**: 보안 무전 상황을 녹음한 WAV 파일 (`20260526_All_units.wav`)
- **출력**: 화자 이름 + 시간대 + 발화 내용이 포함된 깔끔한 전사본 (콘솔 + `_transcript.txt` 파일)

### 1.2 왜 이 기술이 필요한가? (실무 적용 이유)
BiNCS 과정에서 배우는 **AI-CCTV 플랫폼** 프로젝트와 직접 연결됩니다.

| 상황 | 문제점 | 이 코드가 해결하는 것 |
|------|--------|---------------------|
| 보안 무전 | 여러 사람이 빠르게 번갈아 말함 | 화자 자동 구분 |
| 사건 발생 후 | "누가 뭐라고 했는지" 수동으로 듣고 정리 | 자동 전사 + 시간 태깅 |
| 이후 분석 | LLM/RAG으로 요약·검색 필요 | 구조화된 텍스트 데이터 제공 |

**학습 포인트**:  
단순 STT(Whisper)만으로는 부족하고, **화자 분리(pyannote)**를 결합해야 실전에서 쓸 수 있는 데이터가 됩니다.  
이것이 바로 **멀티모달(Multimodal)**의 핵심입니다. (음성 + 메타데이터 결합)

### 1.3 사용 기술 스택과 선택 이유

| 기술 | 역할 | 왜 선택했는가? (학습 관점) |
|------|------|---------------------------|
| **Whisper (OpenAI)** | 음성→텍스트 변환 (STT) | zero-shot 성능 우수, 영어 무전 인식률 높음, `initial_prompt`로 도메인 적응 가능 |
| **pyannote.audio 3.1** | 화자 분리 (Speaker Diarization) | Hugging Face에서 가장 성능 좋은 오픈소스 모델, Pipeline으로 사용 편의성 높음 |
| **soundfile** | 오디오 파일 읽기 | Windows에서 FFmpeg 설치 없이도 안정적으로 waveform 추출 가능 (실무 호환성) |
| **torch** | Tensor 변환 | pyannote 입력 형식 맞추기 (channel, time) |
| **langchain** | (현재 미사용) | 향후 LLM 요약·RAG 파이프라인 확장용으로 미리 import |

> **중요**: langchain 관련 코드는 현재 실행되지 않지만, **다음 단계(LLM 연동)**를 위해 미리 준비한 것입니다. BiNCS 과정의 학습 흐름과 정확히 일치합니다.

---

## 2. 코드 안에 변수명들의 상세 정리 (명세서 기준 완전 분석)

아래는 코드에 등장하는 **모든 주요 변수·상수·함수**를 명세서 순서대로 정리한 것입니다.  
각 항목에 **정의 위치**, **데이터 타입**, **실제 값(이번 실행)**, **역할**, **왜 이렇게 설계했는지**를 명확히 기록했습니다.

### 2.1 전역 설정 (Configuration)

| 변수명 | 타입 | 이번 실행 값 | 역할 및 상세 설명 | 설계 이유 (왜 이렇게 했는가) |
|--------|------|--------------|-------------------|-----------------------------|
| `MODEL_NAME` | str | `"base"` | Whisper 모델 크기 지정 | CPU 환경에서는 `"base"`, GPU면 `"turbo"`로 변경. 학습 시 CPU 실습 환경을 고려한 기본값 |
| `AUDIO_PATH` | str | `"./20260526_All_units.wav"` | 처리할 오디오 파일 경로 | 실제 프로젝트에서 날짜별로 파일이 쌓이므로 날짜 prefix 사용 |
| `HF_TOKEN` | str | `os.getenv("HF_TOKEN")` | Hugging Face 인증 토큰 | pyannote 모델 다운로드 시 필요. `.env` 파일로 관리 (보안) |
| `SPEAKER_NAMES` | dict | `{"SPEAKER_00": "민욱 (Control)", ...}` | pyannote 라벨 → 실제 화자 이름 매핑 | pyannote는 SPEAKER_00, 01...으로만 반환하므로 **사후 매핑**이 필수. 실행 후 수동 조정하도록 주석에 명시 |

### 2.2 핵심 함수 (Core Functions)

#### `correct_timestanps(segments: list, actual_duration: float) -> tuple`
- **위치**: 28~55번째 줄
- **입력**:
  - `segments`: Whisper가 생성한 세그먼트 리스트 (각각 `start`, `end`, `text` 포함)
  - `actual_duration`: soundfile로 측정한 실제 오디오 길이 (초)
- **출력**: `(보정된 segments, 보정 비율)`
- **상세 로직**:
  1. Whisper 마지막 세그먼트 끝 시간(`whisper_max`)과 실제 길이 비교
  2. 차이가 1초 이내면 보정 없이 원본 반환
  3. 차이가 크면 `ratio = actual_duration / whisper_max` 계산 후 모든 타임스탬프에 곱함
- **왜 필요한가?**
  - Windows + ffmpeg 미설치 환경에서 Whisper가 오디오를 16000Hz로 리샘플링하지 못해 타임스탬프가 **2.75배** 정도 부풀려지는 현상 발생
  - 이 함수는 **실제 오디오 길이 기준으로 비율 보정**하여 pyannote와의 시간 정렬을 맞춤
- **이번 실행**: 보정 필요 없었음 (차이 1초 이내) → 경고 메시지 출력되지 않음

#### `get_speaker_at(annotation, start: float, end: float) -> str`
- **위치**: 57~75번째 줄
- **입력**: pyannote `annotation` 객체 + Whisper 세그먼트의 시작/끝 시간
- **출력**: 해당 시간대에 속하는 화자 ID (`SPEAKER_00` 등) 또는 `"UNKNOWN"`
- **핵심 로직**:
  - 세그먼트의 **중간 시점(mid = (start + end) / 2)** 계산
  - pyannote의 각 turn을 순회하며 `turn.start <= mid < turn.end` 조건으로 매칭
  - **중간값 사용 이유**: 경계선에서 두 구간이 겹칠 수 있는 문제 방지
  - **열린 구간(<) 사용 이유**: 경계 중복 매핑 방지
- **왜 중요한가?**
  - Whisper와 pyannote의 세그먼트 경계가 정확히 일치하지 않기 때문에 **중간값 휴리스틱**이 실무에서 매우 효과적

#### `merge_consecutive_segments(segments: list) -> list`
- **위치**: 77~100번째 줄
- **입력**: 화자 정보가 추가된 세그먼트 리스트
- **출력**: 연속된 같은 화자 발화가 하나로 합쳐진 리스트
- **상세 로직**:
  - `gap_threshold = 0.8` (초)
  - 이전 세그먼트와 현재 세그먼트의 화자가 같고, 간격이 0.8초 미만이면 **텍스트를 연결**하고 `end` 시간만 갱신
  - 그렇지 않으면 새 세그먼트로 추가
- **왜 필요한가?**
  - Whisper는 문장 단위로 잘게 나누지만, 실제 대화는 **연속 발화**가 많음
  - 0.8초라는 임계값은 **자연스러운 대화 흐름**을 유지하면서 과도한 병합을 막는 경험적 값
  - 결과: 10개 → 8개로 줄어 가독성 대폭 향상

### 2.3 실행 중 생성되는 주요 변수

| 변수명 | 타입 | 내용 | 역할 |
|--------|------|------|------|
| `whisper_model` | whisper.Whisper | Whisper base 모델 객체 | STT 추론 엔진 |
| `result` | dict | Whisper 전체 결과 (`segments`, `text` 등) | STT 원본 출력 |
| `actual_duration` | float | 121.85027210884354 | soundfile로 측정한 실제 길이 |
| `diarization_pipeline` | pyannote Pipeline | pyannote/speaker-diarization-3.1 파이프라인 | 화자 분리 엔진 |
| `data, sample_rate` | np.ndarray, int | float32 waveform, 44100 | soundfile로 읽은 오디오 데이터 |
| `waveform` | torch.Tensor | shape (1, time) | pyannote 입력 형식으로 변환 |
| `output` | DiarizeOutput | pyannote 실행 결과 | `.speaker_diarization` 속성에 Annotation 저장 |
| `annotation` | Annotation | 화자 구간 정보 | itertracks()로 (turn, _, speaker) 순회 가능 |
| `segments_with_speaker` | list[dict] | 화자 정보 추가된 Whisper 세그먼트 | 중간 결과 |
| `merged_segments` | list[dict] | 최종 병합된 세그먼트 | STEP 4에서 사용 |

---

## 3. 코드의 흐름 (완벽히 깔끔하게 정리한 실행 순서)

아래는 **명세서 기준**으로 코드를 4단계로 완전히 재구성한 흐름입니다.  
각 단계마다 **입력 → 처리 → 출력**을 명확히 하고, **왜 이 순서로 설계했는지** 설명합니다.

### STEP 0: 환경 준비 (사전 단계)
```python
load_dotenv()
MODEL_NAME = "base"
AUDIO_PATH = "./20260526_All_units.wav"
HF_TOKEN = os.getenv("HF_TOKEN")
SPEAKER_NAMES = { ... }
```
- **이유**: 모든 설정을 코드 상단에 모아 **한눈에 파악** 가능하게 함. `.env`로 토큰 관리 (보안 + 재사용성)

### STEP 1: Whisper STT (음성 → 텍스트 변환)
```python
whisper_model = whisper.load_model(MODEL_NAME)
result = whisper_model.transcribe(
    AUDIO_PATH,
    language="en",
    task="transcribe",
    initial_prompt="Security radio communication. ...",
    verbose=False,
    fp16=False,
)
actual_duration = sf.info(AUDIO_PATH).duration
correct_timestanps(result['segments'], actual_duration)  # ← 주의: 반환값 미할당 (버그 가능성)
```
**상세 설명**:
- `language="en"`: 영어 무전이므로 영어로 고정 (한국어로 바꾸면 한국어 인식)
- `initial_prompt`: 도메인 힌트 제공 → "Security radio communication"이라는 맥락을 미리 알려 인식률 향상
- `fp16=False`: CPU 환경에서는 반드시 False (GPU면 True로 변경 시 2배 속도)
- `correct_timestanps`: 타임스탬프 보정 시도 (이번에는 보정 안 됨)

**출력 예시**:
```
✅ STT 완료 — 세그먼트 수: 10개
전체 텍스트 미리보기: All units be advised. We have a suspicious individual...
실제 오디오 길이 : 121.85027210884354초
```

### STEP 2: pyannote 화자 분리
```python
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,
)
data, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
waveform = torch.from_numpy(data).unsqueeze(0)
output = diarization_pipeline(
    {"waveform": waveform, "sample_rate": 44100},
    min_speakers=2,
    max_speakers=4,
)
annotation = output.speaker_diarization
```
**상세 설명**:
- `Pipeline.from_pretrained`: 최초 실행 시 Hugging Face에서 자동 다운로드, 이후 캐시 사용
- `soundfile.read`: FFmpeg 없이도 안정적으로 로드 (Windows 실무 필수)
- `unsqueeze(0)`: (time,) → (1, time)으로 채널 차원 추가 (pyannote 요구 형식)
- `min/max_speakers`: num_speakers 고정 대신 범위 지정 → 유연성 확보

**출력 예시**:
```
✅ 화자 분리 완료
감지된 화자 구간:
SPEAKER_01: 0.0s ~ 22.8s
SPEAKER_00: 22.9s ~ 29.3s
...
```

### STEP 3: STT + 화자 병합 (가장 핵심 로직)
```python
segments_with_speaker = []
for seg in result["segments"]:
    speaker_id = get_speaker_at(annotation, seg["start"], seg["end"])
    speaker_name = SPEAKER_NAMES.get(speaker_id, speaker_id)
    segments_with_speaker.append({...})

merged_segments = merge_consecutive_segments(segments_with_speaker)
```
**상세 설명**:
1. Whisper의 각 세그먼트마다 `get_speaker_at`으로 화자 매칭
2. `SPEAKER_NAMES` 딕셔너리로 실제 이름 변환 (UNKNOWN은 그대로 유지)
3. `merge_consecutive_segments`로 연속 발화 병합 (0.8초 임계값)

**결과**: 10개 세그먼트 → **8개**로 압축

### STEP 4: 화자별 대화 전사 출력 및 저장
```python
for seg in merged_segments:
    # 분:초 포맷팅
    print(f"┌─ [{seg['speaker_name']}] ...")
    print(f"│  {seg['text']}")

# 파일 저장
transcript_path = AUDIO_PATH.replace(".wav", "_transcript.txt")
with open(transcript_path, "w", encoding="utf-8") as f:
    ...
```
**목적**: 
- 사람이 읽기 쉬운 형식으로 콘솔 출력
- `_transcript.txt` 파일로 저장 → **다음 단계(LLM 요약, RAG)**에서 바로 사용 가능

---

## 4. 출력 결과 분석 (왜 이렇게 나왔는가? 상세 원인 분석)

### 4.1 전체 실행 결과 요약
- **Whisper 세그먼트**: 10개 → **병합 후 8개**
- **감지된 화자**: SPEAKER_01 (주완 Unit 3), SPEAKER_00 (민욱 Control)
- **UNKNOWN 발생**: 1개 (01:20.00 ~ 01:29.00)
- **전사 품질**: 전반적으로 우수하나 마지막 부분 약간 끊김

### 4.2 상세 분석

#### (1) 타임스탬프 보정 미발생
- `correct_timestanps` 호출 시 실제 길이와 Whisper 최대 시간이 1초 이내로 차이 났음
- → 보정 없이 원본 사용 → 경고 메시지 없음
- **학습 포인트**: 이 함수는 Windows 실습 환경에서 **필수 방어 코드**임

#### (2) torchcodec Warning (치명적이지 않음)
```
UserWarning: torchcodec is not installed correctly...
```
- **원인**: pyannote 내부에서 FFmpeg 관련 라이브러리 탐색 실패
- **왜 무시해도 되는가?**
  - 코드에서 `soundfile.read()` + `torch.from_numpy()`로 waveform을 **직접 생성**하여 넘김
  - pyannote가 내부 디코더를 사용하지 않으므로 정상 동작
- **실무 교훈**: Windows에서 pyannote 사용할 때 soundfile 우회 패턴을 반드시 익혀야 함

#### (3) UNKNOWN 세그먼트 발생 원인
- `get_speaker_at` 함수에서 `mid` 값이 pyannote의 어떤 turn에도 속하지 않음
- **가능한 원인**:
  1. Whisper와 pyannote의 세그먼트 경계 미세 불일치
  2. pyannote가 해당 구간을 화자 전환 구간으로 판단 (turn 사이 gap)
- **개선 방안**: `mid` 대신 `start` 또는 `end`를 사용하거나, tolerance(허용 오차)를 추가하는 방법 고려

#### (4) 텍스트 품질 (마지막 부분 끊김)
```
│  We are approaching from the opposite side... Unit o be acting as lookout...
│
│  Primary subject notice our presence. Subject is increasing locking speed.
```
- **원인**:
  - Whisper가 긴 문장을 여러 세그먼트로 나눠 인식
  - 오디오 끝부분이라 인식률 저하
  - "Unit 2" → "Unit o" 로 잘못 인식 (음성 유사성)
- **개선 방안**:
  - `initial_prompt`를 더 구체적으로 작성
  - `temperature=0` 또는 `beam_size` 조정
  - 후처리(LLM 보정) 단계 추가

#### (5) 병합 결과 (10 → 8)
- `merge_consecutive_segments`가 2개의 연속 발화를 성공적으로 합침
- **효과**: 가독성 대폭 향상, LLM이 처리하기 좋은 단위로 재구성

### 4.3 최종 전사본 품질 평가

| 항목 | 평가 | 비고 |
|------|------|------|
| 화자 구분 정확도 | ★★★★★ | SPEAKER_00/01이 실제 Control / Unit 3와 잘 매칭 |
| 시간 정확도 | ★★★★☆ | UNKNOWN 1개 발생했으나 전체적으로 양호 |
| 텍스트 정확도 | ★★★★☆ | 마지막 부분 약간의 인식 오류 |
| 가독성 | ★★★★★ | 병합 로직 덕분에 매우 깔끔 |

---

## 5. 학습 요약 및 다음 단계 제안 (BiNCS 과정 관점)

### 5.1 이번 코드로 배운 핵심 개념
1. **멀티모달 파이프라인 구성 방법**  
   - 서로 다른 모델(Whisper + pyannote)의 출력을 **시간 기준으로 정렬**하는 기법

2. **실무 환경 대응력**  
   - Windows + FFmpeg 미설치 환경에서 soundfile로 우회하는 패턴
   - 타임스탬프 보정 로직 (sampling rate mismatch 문제)

3. **데이터 품질 향상 기법**  
   - `initial_prompt`로 도메인 적응
   - 연속 발화 병합 (`gap_threshold`)
   - 중간값 매칭 (`mid` 휴리스틱)

4. **파이프라인 확장성**  
   - `_transcript.txt` 저장 → LangChain + LLM + RAG으로 이어지는 학습 흐름

### 5.2 다음 단계 추천 (BiNCS 과정과 연계)
- **STEP 5**: 생성된 `_transcript.txt`를 LangChain으로 불러와 LLM 요약 (사건 보고서 자동 생성)
- **STEP 6**: RAG 파이프라인에 전사본 + 이미지(Grad-CAM) + 메타데이터 통합
- **STEP 7**: Strawberry Doctor 프로젝트에 음성 로그 기능 추가 (My Farm 대시보드 확장)

---

**문서 작성 완료**  
이 MD 파일은 **명세서(코드 + 실행 결과)**를 100% 반영하여, 변수 하나하나, 흐름 하나하나를 상세히 분석한 결과입니다.  
학습자가 이 파일을 보면서 "내가 배우는 내용"을 완벽히 이해하고, 실무에 적용할 수 있는 수준까지 도달할 수 있도록 작성되었습니다.

필요 시 이 파일을 기반으로 **개선된 코드 버전**이나 **다음 단계(STEP 5) 코드**도 만들어 드리겠습니다.
