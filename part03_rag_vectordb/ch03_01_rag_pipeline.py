# ch03_01_rag_pipeline.py
# LangChain LCEL RAG 파이프라인
# 탐지 로그 CSV → ChromaDB → Retriever → GPT-4o → 대응 방안 생성

# raw_docs의 구조: List[Document]
# [
#   Document(page_content="timestamp: ...\nlocation: ...", metadata={...}),
#   Document(page_content="timestamp: ...\nlocation: ...", metadata={...}),
#   ...
# ]

# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv
from dotenv import load_dotenv
from typing import List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CSV_PATH = "detection_logs.csv"

sample_rows = [
    ["timestamp",        "location",   "classes_detected",       "risk_level", "action_taken",   "note"],
    ["2024-11-20 02:31", "창고 출입구", "person:2/car:1",         "위험",       "경보 후 도주",    "야간 침입 시도"],
    ["2025-01-08 03:15", "공장 외곽",  "person:3/car:2/truck:1", "위험",       "경찰 출동",       "복수 인원 침입"],
    ["2024-12-05 14:22", "주차장 A",   "person:1",               "정상",       "없음",            "정상 이용객"],
    ["2024-10-15 23:45", "창고 주변",  "person:1",               "주의",       "경비 순찰",       "배회 감지"],
    ["2024-09-03 12:00", "정문",       "person:5",               "정상",       "없음",            "회의 참석자"],
    ["2025-02-11 01:05", "주차장 B",   "person:2",               "위험",       "CCTV 확대/경보",  "차량 접근 반복"],
    ["2024-08-22 22:10", "후문",       "person:1/car:1",         "주의",       "경비 확인",       "야간 차량 정차"],
    ["2025-03-01 03:40", "창고 출입구","person:1",               "위험",       "경비 출동",       "새벽 단독 침입"],
]

# with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
#     csv.writer(f).writerows(sample_rows)
# print(f"[OK] STEP 1: CSV 생성 완료 ({len(sample_rows)-1}건)\n")

# ─────────────────────────────────────────────────────────────
# STEP 2: DocumentLoader — CSV 파일을 Document 객체로 읽기
# CSVLoader는 각 행(row) = Document 1개로 변환합니다.
# ─────────────────────────────────────────────────────────────
loader = CSVLoader(
  file_path=CSV_PATH,
  encoding='utf-8',
  source_column="timestamp", # 어떤 컬럼을 문서의 출처로 표시할지
)

raw_docs = loader.load()

print(f"[OK] STEP 2: {len(raw_docs)}개 Document 로드")
print(f"   첫 번째 Document 내용:")
print(f"   {raw_docs[0].page_content[:100]}")
print(f"   메타데이터: {raw_docs[0].metadata}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: TextSplitter — 문서를 청크로 분할
# CSV 행은 짧아서 분할이 거의 일어나지 않지만,
# 보안 매뉴얼 PDF 같은 긴 문서에서는 이 설정이 핵심!
# ─────────────────────────────────────────────────────────────
splitter = CharacterTextSplitter(
  chunk_size=500,  # 청크 하나단위의 최대 글자수
  chunk_overlap=50, # 청크 앞뒤가 50자씩 겹치도록 일부로  -> 맥락이 끊기지 않도록
  separator="\n"  # 청크와 청크를 구분하는 구분기호
)

docs =splitter.split_documents(raw_docs)

print(f"[OK] STEP 3: {len(raw_docs)}개 → {len(docs)}개 청크로 분할\n")

# ─────────────────────────────────────────────────────────────
# STEP 4: OpenAI Embedding + ChromaDB에 저장
# from_documents()가 임베딩 + DB 저장을 한 번에 처리합니다.
# ─────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
print(f"[OK] STEP 4: ChromaDB에 {len(docs)}개 청크 저장 완료\n")

# ─────────────────────────────────────────────────────────────
# STEP 5: Retriever — 검색기 (LLM이 검색할 대상)생성
# as_retriever()로 vectorstore를 검색기로 변환합니다.
# Django의 Model.objects.filter()[:3]과 비슷한 개념
# ─────────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("[OK] STEP 5: Retriever 구성 완료 (상위 3개 반환)\n")

# ─────────────────────────────────────────────────────────────
# STEP 6: LCEL RAG 체인 구성
# | 기호로 각 단계를 연결합니다 (왼쪽 → 오른쪽으로 데이터 흐름)
# ─────────────────────────────────────────────────────────────
prompt = PromptTemplate.from_template(
    """당신은 CCTV 보안 분석 전문가입니다.
아래 과거 탐지 로그를 참고하여 현재 상황에 대한 대응 방안을 제시하세요.
negative prompt : 주어지지 않은 정보에 대한 추론 금지.

[참고 과거 사례]
{context}

[현재 탐지 상황]
{question}

위험도(정상/주의/위험) 판단과 즉각적인 조치사항을 구체적으로 답변하세요."""
)

def format_docs(docs:List[Document]) ->str:
   """
    Retriever가 반환한 Document 리스트를 하나의 문자열로 합칩니다.
    이 문자열이 프롬프트의 {context} 자리에 들어갑니다.
   """
   
   return "\n----\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()  # qurey를 변경하지 않고 그대로 자리에 넣는다 
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("[OK] STEP 6: LCEL RAG 체인 구성 완료\n")


# ─────────────────────────────────────────────────────────────
# STEP 7: 실제 쿼리 실행
# ─────────────────────────────────────────────────────────────
query = "새벽 2시 창고 출입구에서 person:2 / car:1 탐지. 침입 의심. 어떻게 대응해야 하나요?"

print("=" * 60)
print(f"[SEARCH] 현재 상황: {query}")
print("=" * 60)

retrieved = retriever.invoke(query)

for i, doc in enumerate(retrieved, 1):
    lines = doc.page_content.split("\n")
    print(f"  [{i}] {' | '.join(l for l in lines[:3] if l.strip())[:80]}")

answer = rag_chain.invoke(query)

print(f"\n[ASSISTANT] GPT-4o-mini 대응 방안:\n{answer}")
