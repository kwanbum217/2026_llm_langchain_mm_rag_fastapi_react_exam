# ChromaDB에 탐지 로그를 저장하고 유사 상황을 검색해 봅시다


# [SYSTEM_INIT] Windows 터미널 인코딩 충돌 방지 강제 설정
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core import documents
from annotated_types import doc
from chromadb import Collection
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# OpenAIEmbeddingFunction : chromadb가 문서 저장/검색 시 자동으로 
# OpenAI API를 호출하여 벡터로 변환해준다.
embedding_fn = OpenAIEmbeddingFunction(    
    model_name="text-embedding-3-small"
)

# 인메모리 클라이언트 (실습용 — 프로그램 종료 시 데이터 사라짐)
# 영구 저장하려면: client = chromadb.PersistentClient(path="./chroma_db")
# client = chromadb.Client()
client = chromadb.PersistentClient(path="./chroma_db")
# collection : Vector DB안의 데이터를 저장하는 단위 (RDB에서의 테이블)
collection = client.create_collection(
    name="cctv_detection_logs",
    embedding_function=embedding_fn
)

# ─── 탐지 로그 데이터 ─────────────────────────────────────────
detection_logs = [
    {
        "id": "log_001",
        "text": "새벽 2시 31분 창고 출입구 침입. person:2 car:1 탐지. 위험.",
        "metadata": {
            "timestamp": "2024-11-20 02:31",
            "location": "창고 출입구",
            "risk_level": "위험",
            "action_taken": "경보 후 도주",
        },
    },
    {
        "id": "log_002",
        "text": "새벽 3시 15분 공장 외곽 침입 시도. person:3 car:2 truck:1 탐지. 위험.",
        "metadata": {
            "timestamp": "2025-01-08 03:15",
            "location": "공장 외곽",
            "risk_level": "위험",
            "action_taken": "경찰 출동",
        },
    },
    {
        "id": "log_003",
        "text": "오후 2시 22분 주차장 A구역. person:1 탐지. 정상 이용객.",
        "metadata": {
            "timestamp": "2024-12-05 14:22",
            "location": "주차장 A",
            "risk_level": "정상",
            "action_taken": "없음",
        },
    },
    {
        "id": "log_004",
        "text": "심야 창고 주변 배회. person:1 반복 배회 감지. 주의.",
        "metadata": {
            "timestamp": "2024-10-15 23:45",
            "location": "창고 주변",
            "risk_level": "주의",
            "action_taken": "경비 순찰",
        },
    },
    {
        "id": "log_005",
        "text": "낮 12시 정문 앞 person:5 집단 탐지. 정상 회의 참석자.",
        "metadata": {
            "timestamp": "2024-09-03 12:00",
            "location": "정문",
            "risk_level": "정상",
            "action_taken": "없음",
        },
    },
    {
        "id": "log_006",
        "text": "새벽 1시 주차장 B구역 침입 시도. person:2 탐지. 차량 접근 반복. 위험.",
        "metadata": {
            "timestamp": "2025-02-11 01:05",
            "location": "주차장 B",
            "risk_level": "위험",
            "action_taken": "CCTV 확대 후 경보",
        },
    },
]

# ChromaDB에 저장 (RDB 에서의 insert into문)
# text 를 자동으로 벡터로 변환 후 저장한다
collection.add(
    documents=[log['text'] for log in detection_logs ],
    ids = [log['id'] for log in detection_logs],
    metadatas = [log['metadata'] for log in detection_logs]
)

print(f"로그가 {collection.count()}개 저장되었습니다.\n")

# ─── 유사 상황 검색 ──────────────────────────────────────────
query = "새벽 창고 출입구 침입 의심. person:2 탐지."

results = collection.query(
    query_texts=[query],
    n_results=3, # 상위 3개 반환
)

# results 딕셔너리 구조:
# {
#   "documents": [[doc1, doc2, doc3], [,,]],   ← 검색된 텍스트 (리스트의 리스트)
#   "metadatas": [[m1, m2, m3]],         ← 메타데이터
#   "distances": [[d1, d2, d3]],         ← 거리 (낮을수록 유사)
# }
# [0] 인덱스: 쿼리가 1개이므로 첫 번째(0번) 결과만 사용

print(f"[SEARCH] 현재 상황: {query}\n")
print("[ACTION] 유사 과거 사례 TOP 3:")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
), 1):
    print(f"\n  [{i}위] 거리: {dist:.3f} (낮을수록 유사)")
    print(f"  [DATE] {meta['timestamp']} | [LOCATION] {meta['location']}")
    print(f"  [FILE] {doc}")
    print(f"  [ACTION_TAKEN] 당시 조치: {meta['action_taken']}")

# 메타데이터 필터링 검색
print("\n\n[TIP] 추가: '위험' 등급 로그만 필터링 검색")

# where 조건 이용
filtered= collection.query(
    query_texts=[query],
    n_results=5,
    where={"risk_level":"위험"},  # 딕셔너리로 제공
)

print("'위험' 등급 중 유사 사례:")
for doc, meta in zip(filtered["documents"][0], filtered["metadatas"][0]):
    print(f"  - {meta['timestamp']} | {meta['location']} | {doc[:50]}")