# 로컬 Whisper 배치 STT 분석 데이터 기반 ChromaDB 연동 RAG 시스템 구축
# 
# 작동 개요:
# 1. waves/ 폴더에 위치한 오디오 파일 리스트 추출 및 Whisper 로컬 배치 STT 수행 (language=None 자동 판독)
# 2. 세그먼트 데이터로부터 텍스트, 시작/종료 시간, 신뢰도를 포함하는 LangChain Document 객체 생성
# 3. OpenAI text-embedding-3-small 임베딩 모델을 활용해 메타데이터와 함께 ChromaDB 저장
# 4. 출처 정보(파일명, 재생 시간 구간)가 포함된 RAG 컨텍스트 결합 후 GPT-4o 질의응답 및 보안 보고서 생성
#
# 규칙 준수: 모든 콘솔 출력 로그에서 이모지는 완전히 제거되었으며 대괄호 기호로 통일되었습니다.

import os
import sys
import shutil
import time
import whisper  # type: ignore
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Windows CP949 콘솔 한글 깨짐 방지 및 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
if hasattr(sys.stderr, "reconfigure"):
  sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

# [중요 튜닝]: Windows 환경변수 지연 반영 해결 기법 (ffmpeg 핫로딩)
if not shutil.which("ffmpeg"):
  winget_ffmpeg_dir = r"C:\Users\lucian\AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
  if os.path.exists(winget_ffmpeg_dir):
    for sub in os.listdir(winget_ffmpeg_dir):
      sub_path = os.path.join(winget_ffmpeg_dir, sub)
      if os.path.isdir(sub_path) and sub.startswith("ffmpeg-"):
        bin_path = os.path.join(sub_path, "bin")
        if os.path.exists(bin_path):
          os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
          break

# API 환경 변수 로드
load_dotenv()

# ─── 설정 ─────────────────────────────────────────────────────────────
MODEL_NAME = "tiny"  # CPU 환경에서의 원활한 구동을 위해 초경량 tiny 모델로 지정 (RAM 절약)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "waves")
# ──────────────────────────────────────────────────────────────────────

def get_audio_documents():
  """
  waves/ 폴더에서 오디오 파일들을 스캔하고, Whisper 로컬 CPU 모델을 통해 
  전사된 세그먼트 데이터를 구조화된 Document 객체 리스트로 생성합니다.
  """
  import gc  # 메모리 OOM 방지를 위한 가비지 컬렉션 모듈
  import torch

  if not os.path.exists(AUDIO_DIR):
    print(f"[ERROR] 오디오 디렉토리가 없습니다: {AUDIO_DIR}")
    return []

  audio_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac"))
  ]

  if not audio_files:
    print(f"[WARNING] 변환할 오디오 파일이 waves 폴더 내에 없습니다.")
    return []

  # CPU 연산에 집중하도록 디바이스 'cpu' 고정
  device = "cpu"
  print(f"[INFO] CPU 모드 가동 - 모델 로딩 중: {MODEL_NAME}")
  start_time = time.time()
  model = whisper.load_model(MODEL_NAME, device=device)
  print(f"[OK] 모델 로딩 완료: {time.time() - start_time:.1f}초\n")

  documents = []

  for filename in sorted(audio_files):
    filepath = os.path.join(AUDIO_DIR, filename)
    print(f"[INFO] 변환 중 (CPU): {filename}")
    
    start_trans = time.time()
    # language=None 으로 자동 감지 지원, CPU 연산이므로 fp16=False 고정
    result = model.transcribe(
      filepath,
      language=None,
      fp16=False
    )
    print(f"  [OK] 변환 완료: {time.time() - start_trans:.1f}초 (감지 언어: {result['language']})")

    for seg in result["segments"]:
      if not isinstance(seg, dict):
        continue
      text = str(seg.get("text", "")).strip()
      if not text:
        continue
      
      start_sec = float(seg.get("start", 0.0))
      end_sec = float(seg.get("end", 0.0))
      prob = float(seg.get("avg_logprob", 0.0))
      
      # avg_logprob 신뢰도 정형화
      confidence = "OK" if prob > -0.5 else ("주의" if prob > -1.0 else "불량")

      # LangChain Document 객체화 및 메타데이터 바인딩
      doc = Document(
        page_content=text,
        metadata={
          "source": filename,
          "start_time": f"{start_sec:.1f}s",
          "end_time": f"{end_sec:.1f}s",
          "timestamp": f"[{start_sec:.1f}s -> {end_sec:.1f}s]",
          "confidence": confidence
        }
      )
      documents.append(doc)
      print(f"    - {doc.metadata['timestamp']} {text} (신뢰도: {doc.metadata['confidence']})")
    
    # ─── CPU 메모리 누수 및 OOM 완화 조치 ───
    # 매 파일 변환 작업이 완료될 때마다 리소스 강제 해제 및 캐시 비우기 수행
    del result
    gc.collect()
    print("-" * 60)

  print(f"[OK] 총 {len(documents)}개의 문장/세그먼트 추출 완료\n")
  return documents

def main():
  print("=" * 60)
  print("[START] Whisper RAG 파이프라인 빌드 및 가동")
  print("=" * 60)

  # 1. 오디오 문장 및 메타데이터 추출
  docs = get_audio_documents()
  if not docs:
    print("[ERROR] 처리할 오디오 문서 데이터가 없어 시스템을 종료합니다.")
    return

  # 2. 임베딩 모델 로드 및 ChromaDB 적재 (로컬 메모리형)
  print("[INFO] OpenAI text-embedding-3-small 모델 로드 및 Chroma DB 빌드 중...")
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
  )
  print("[OK] 벡터 데이터베이스 빌딩 완료\n")

  # 3. Retriever 및 RAG QA Chain 구성
  # 유사도 최상위 4개 문서 참조
  retriever = vector_db.as_retriever(search_kwargs={"k": 4})
  
  # GPT-4o 연동 프롬프트 템플릿 설계
  prompt_template = PromptTemplate.from_template(
    """당신은 무전 보안 관제 지원 AI 비서입니다.
반드시 아래의 [관제 무전 정보] 컨텍스트만을 철저히 근거로 삼아 질문에 답변하십시오.
제공된 정보로 답을 유추하거나 알 수 없을 경우, 거짓 정보나 추측성 내용을 지어내지 말고 정직하게 "제시된 무전 정보에서 답변할 수 있는 근거를 찾을 수 없습니다"라고 답하십시오.

답변 시 각 정보의 출처 파일명과 발화 구간(타임스탬프) 정보를 대괄호 안에 명시하여 답변의 신뢰도를 높이십시오.
예: "용의자가 3층으로 도주 중이라는 교신이 있었습니다. [source: radio_sample.wav, timestamp: [12.4s -> 15.6s]]"

[관제 무전 정보]
{context}

사용자 질문: {question}
AI 답변:"""
  )

  llm = ChatOpenAI(model="gpt-4o", temperature=0)
  
  # 랭체인 헬퍼 함수를 사용하여 텍스트 형태로 컨텍스트 결합
  def format_docs(docs_list):
    formatted = []
    for d in docs_list:
      formatted.append(
        f"출처: {d.metadata['source']} | 구간: {d.metadata['timestamp']} | 신뢰도: {d.metadata['confidence']}\n"
        f"무전 텍스트: {d.page_content}\n"
        f"---"
      )
    return "\n".join(formatted)

  # LangChain Expression Language (LCEL) 체인 연동
  from langchain_core.runnables import RunnablePassthrough
  
  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
  )

  # 4. 무전 통합 RAG 질의응답 시나리오 실행
  query_1 = "무전에서 식별되거나 침입한 인원 수와 현재 대응 상태를 요약하고, 관련 출처 시간도 함께 표시해줘."
  
  print("=" * 60)
  print(f"[질문 1] {query_1}")
  print("=" * 60)
  
  ans_1 = rag_chain.invoke(query_1)
  print(f"\n{ans_1}\n")

  query_2 = "무전기에서 'All units'를 찾고, 해당 무전이 송신된 오디오 소스 파일명과 타임스탬프 범위를 상세히 알려줘."
  
  print("=" * 60)
  print(f"[질문 2] {query_2}")
  print("=" * 60)

  ans_2 = rag_chain.invoke(query_2)
  print(f"\n{ans_2}\n")

  print("=" * 60)
  print("[SUCCESS] RAG 연동 검증 및 보안 질문 생성 테스트 완료")
  print("=" * 60)

if __name__ == "__main__":
  main()
