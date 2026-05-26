# ch03_02_multi_query_retriever.py
# LangChain MultiQueryRetriever 실습 코드
# 모호한 자연어 질문 -> LLM 다중 질문 생성 -> 벡터 검색 병렬 처리 -> RAG 분석 대응 방안 생성

import os
import sys
import logging
from dotenv import load_dotenv
from typing import List

# Windows 터미널 한글 깨짐 방지 및 UTF-8 출력 강제 설정
# Pylance 등 IDE 정적 분석기의 타입 경고 밑줄 방지를 위해 type: ignore 주석을 추가합니다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 특수 래퍼 패키지 경로에 대한 IDE의 미확인 참조(Import Resolution) 밑줄 경고를 예방하기 위해 type: ignore를 선언합니다.
from langchain_classic.retrievers.multi_query import MultiQueryRetriever  # type: ignore

# -------------------------------------------------------------
# STEP 1: MultiQueryRetriever의 대체 질문 생성을 관찰하기 위한 로깅 설정
# -------------------------------------------------------------
logging.basicConfig()
logger = logging.getLogger("langchain_classic.retrievers.multi_query")
logger.setLevel(logging.INFO)

# [중요 오류 수정]: 부모(루트) 로거로의 전파를 명시적으로 차단(propagate = False)합니다.
# 이를 차단하지 않으면 우리가 추가한 sys.stdout 핸들러의 출력([LOGGER])과
# 루트 로거의 기본 sys.stderr 출력이 이중으로 터미널 콘솔에 찍히는 중복 출력 및 오버헤드 결함이 유발됩니다.
logger.propagate = False

# 명시적으로 StreamHandler를 달아서 터미널 출력을 보장합니다.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[LOGGER] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 환경 변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

CSV_PATH = os.path.join(current_dir, "detection_logs.csv")

print("[STEP 2] DocumentLoader - CSV 파일 로드 및 Document 객체 생성")
loader = CSVLoader(
    file_path=CSV_PATH,
    encoding="utf-8",
    source_column="timestamp",
)
raw_docs = loader.load()
print(f"로드된 전체 문서 개수: {len(raw_docs)}개")

print("\n[STEP 3] TextSplitter - 문서를 청크로 분할")
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
docs = splitter.split_documents(raw_docs)
print(f"분할 후 청크 개수: {len(docs)}개")

print("\n[STEP 4] OpenAI Embedding 및 ChromaDB 저장")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("ChromaDB에 데이터 적재 완료 및 기본 검색기 구성")

# -------------------------------------------------------------
# STEP 5: MultiQueryRetriever 구성
# -------------------------------------------------------------
print("\n[STEP 5] MultiQueryRetriever 구성")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 기본 검색기와 LLM을 전달하여 다중 질의 검색기(MultiQueryRetriever) 생성
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
print("다중 질의 검색기(MultiQueryRetriever) 생성 성공")

# -------------------------------------------------------------
# STEP 6: LCEL RAG 체인 구성
# -------------------------------------------------------------
prompt = PromptTemplate.from_template(
    """당신은 CCTV 보안 분석 전문가입니다.
아래 과거 탐지 로그를 참고하여 현재 상황에 대한 대응 방안을 제시하세요.
부정적 지시사항: 주어지지 않은 정보에 대한 상상이나 추론은 엄격히 금지합니다. 오직 참고용 사실에만 기반하여 답변하세요.

[참고 과거 사례]
{context}

[현재 탐지 상황]
{question}

위험도(정상/주의/위험) 판단과 즉각적인 조치사항을 구체적으로 한국어로만 답변하세요."""
)

def format_docs(docs: List[Document]) -> str:
    """
    검색된 고유 문서들을 포맷팅하여 컨텍스트 문자열로 합칩니다.
    """
    return "\n----\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": multi_query_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
print("LCEL Multi-Query RAG 체인 구성 완료")

# -------------------------------------------------------------
# STEP 7: 실제 모호한 자연어 쿼리 실행
# -------------------------------------------------------------
query = "늦은 밤 시간대에 누군가가 들어오려 하거나 순찰을 돌았던 긴급 대응 내역이 있나요?"

print("\n" + "=" * 60)
print(f"검색할 자연어 질의: {query}")
print("=" * 60)

# MultiQueryRetriever 내부 로깅을 통해 자동 생성된 질문들이 터미널에 출력됩니다.
print("\n--- [로깅 출력 시작: 대체 질문 생성 및 병렬 검색 진행] ---")
retrieved_docs = multi_query_retriever.invoke(query)
print("--- [로깅 출력 종료] ---\n")

print(f"조회된 고유 문서 수: {len(retrieved_docs)}개")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[유사 사례 문서 {i}]")
    print(doc.page_content)

# 최종 체인 실행
print("\n" + "=" * 60)
print("GPT-4o-mini 기반 최종 위험도 및 대응 분석 리포트")
print("=" * 60)
answer = rag_chain.invoke(query)
print(answer)
