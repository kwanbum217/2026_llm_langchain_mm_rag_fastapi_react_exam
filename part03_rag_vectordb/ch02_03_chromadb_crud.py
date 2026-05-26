# 영구히 저장(파일로 저장한) chroma db 운용법

import os
import sys
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 동일 디렉토리 내 모듈 임포트를 위해 sys.path 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ch02_01_cosine_similarity import get_embedding

DB_PATH = "./chroma_db" # DB 경로
COLLECTION_NAME = "cctv_detection_logs" # 컬렉션 이름

load_dotenv()

client_oai = OpenAI()

# collection 불러오기
# 1. create_collection("컬렉션 이름") - 없으면 생성, 있으면 에러
# 2. get_collection("컬렉션 이름") - 없으면 에러.
# 3. get_or_create_collection ("컬렉션 이름") - 없으면 생성, 있으면 가져옴

chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection(
  name=COLLECTION_NAME,
  metadata={
    "hnsw:space": "cosine" # 텍스트 임베딩에 가장 적합한 거리 측정방식(방향만 보고 크기무시)
  }
)

print(f"[OK] DB 경로: {os.path.abspath(DB_PATH)}")
print(f"[OK] 컬렉션: {COLLECTION_NAME} | 현재 저장된 개수: {collection.count()}\n")

if collection.count() == 0 :
  print("=== STEP 1: 초기 데이터 저장 (최초 실행) ===")
  initial_logs = [
        {
            "id": "log_001",
            "text": "새벽 2시 창고 출입구 침입. person:2 car:1. 위험.",
            "metadata": {"risk_level": "위험", "location": "창고 출입구", "resolved": False},
        },
        {
            "id": "log_002",
            "text": "오후 2시 주차장 A. person:1. 정상 이용객.",
            "metadata": {"risk_level": "정상", "location": "주차장 A", "resolved": False},
        },
        {
            "id": "log_003",
            "text": "심야 창고 주변 배회. person:1. 주의.",
            "metadata": {"risk_level": "주의", "location": "창고 주변", "resolved": False},
        },
        {
            "id": "log_004",
            "text": "새벽 공장 외곽 침입. person:3 car:2. 위험.",
            "metadata": {"risk_level": "위험", "location": "공장 외곽", "resolved": False},
        },
        {
            "id": "log_005",
            "text": "낮 정문 정상 출입. person:5. 정상.",
            "metadata": {"risk_level": "정상", "location": "정문", "resolved": False},
        },
    ]

  collection.add(
        embeddings=[get_embedding(log["text"]) for log in initial_logs],
        documents=[log["text"] for log in initial_logs],
        ids=[log["id"] for log in initial_logs],
        metadatas=[log["metadata"] for log in initial_logs],
    )
  print(f"저장 완료. 총 {collection.count()}개\n")
  print("-> 프로그램을 종료하고 다시 실행하면 데이터가 그대로 유지됩니다.\n")

# 모든 분기에서 데이터를 안전하게 조회할 수 있도록 if-else 바깥에서 공통으로 호출
all_data = collection.get()

# 데이터 출력 루프
for id_, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
    resolved_mark = "[V]" if meta.get("resolved", False) else "[ ]"
    print(f"  {resolved_mark} {id_} | {meta.get('risk_level', '정상')} | {doc[:40]}")
print()

# ─────────────────────────────────────────────────────────────
# STEP 2: UPDATE — 문서 내용 + 메타데이터 + 벡터 함께 변경
# 사건이 해결됐을 때 로그 전체를 업데이트하는 시나리오
#
# ⚠️ 핵심 주의사항:
# 문서 내용(documents)이 바뀌면 반드시 새 벡터(embeddings)도 함께 넘겨야 합니다.
# 벡터를 안 넘기면 내부 인덱스에는 예전 텍스트의 벡터가 그대로 남아서
# 새로 추가된 내용("경찰 출동 후 검거" 등)으로 유사도 검색이 안 됩니다.
# ─────────────────────────────────────────────────────────────
print("=== STEP 2: UPDATE (문서+벡터+메타 변경) ===")

before = collection.get(ids=["log_001"])
if not before["documents"]:
    # 이전 실행의 DELETE(where resolved=True)로 인해 지워진 경우 재삽입
    collection.add(
        embeddings=[get_embedding("새벽 2시 창고 출입구 침입. person:2 car:1. 위험.")],
        documents=["새벽 2시 창고 출입구 침입. person:2 car:1. 위험."],
        ids=["log_001"],
        metadatas=[{"risk_level": "위험", "location": "창고 출입구", "resolved": False}],
    )
    before = collection.get(ids=["log_001"])

print(f"변경 전: {before['documents'][0]}")
# print(f"         resolved={before['metadatas'][0]['resolved']}")

new_text = "새벽 2시 창고 출입구 침입. person:2 car:1. 위험. 경찰 출동 후 용의자 검거"
collection.update(
  ids=["log_001"],
  embeddings=[get_embedding(new_text)],  # 내용이 바뀌었으므로 벡터 재생성 필수
  documents=[new_text],
  metadatas=[{"risk_level": "위험", "location": "창고 출입구", "resolved": True}],
)

after = collection.get(ids=["log_001"])
print(f"변경 후: {after['documents'][0]}")
print(f"         resolved={after['metadatas'][0]['resolved']}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: UPDATE — 메타데이터만 변경 (벡터 생략 가능)
# 문서 내용은 그대로이고 처리 상태(resolved)만 바꾸는 시나리오
# 벡터를 생략해도 기존 벡터가 그대로 유지됩니다.
# ─────────────────────────────────────────────────────────────
print("=== STEP 3: UPDATE (메타데이터만 변경) ===")

result = collection.get(ids=["log_003"])
if not result["documents"]:
    # 이전 실행의 DELETE로 인해 지워진 경우 재삽입
    collection.add(
        embeddings=[get_embedding("심야 창고 주변 배회. person:1. 주의.")],
        documents=["심야 창고 주변 배회. person:1. 주의."],
        ids=["log_003"],
        metadatas=[{"risk_level": "주의", "location": "창고 주변", "resolved": False}],
    )
    result = collection.get(ids=["log_003"])

collection.update(
    ids=["log_003"],
    metadatas=[{"risk_level": "주의", "location": "창고 주변", "resolved": True}],
    # documents, embeddings 생략 → 기존 값 유지
)

result = collection.get(ids=["log_003"])
print(f"log_003: {result['documents'][0]}")
print(f"         resolved={result['metadatas'][0]['resolved']} (문서 내용은 그대로)\n")

upsert_logs = [
    {
        "id": "log_002",    # 기존 id → 수정 (update)
        "text": "오후 2시 주차장 A. person:1. 정상 이용객. 차량 확인 완료.",
        "metadata": {"risk_level": "정상", "location": "주차장 A", "resolved": True},
    },
    {
        "id": "log_999",    # 새 id → 추가 (add)
        "text": "새벽 후문 침입 시도. person:1. 위험.",
        "metadata": {"risk_level": "위험", "location": "후문", "resolved": False},
    },
]
print("=== STEP 4: UPSERT (ID 기준 생성/수정) ===")

collection.upsert(
  embeddings=[get_embedding(log["text"]) for log in upsert_logs],
  documents=[log["text"] for log in upsert_logs],
  ids=[log["id"] for log in upsert_logs],
  metadatas=[log["metadata"] for log in upsert_logs],
)

print(f"upsert 후 개수: {collection.count()}  (log_999 신규 추가됨)")
log999 = collection.get(ids=["log_999"])
print(f"log_999: {log999['documents'][0]}\n")

result2 = collection.get(ids=["log_002"])
print(f"log_002: {result2['documents'][0]}")
print(f"    resolved={result2['metadatas'][0]['resolved']} (문서 내용 및 메타데이터 업데이트됨)\n")  

result999 = collection.get(ids=["log_999"])
print(f"log_999: {result999['documents'][0]}")
print(f"    resolved={result999['metadatas'][0]['resolved']} (신규추가)\n")

# ─────────────────────────────────────────────────────────────
# STEP 5: DELETE — id로 삭제
# 오탐(false positive)으로 판명된 로그 제거
# ─────────────────────────────────────────────────────────────
print("=== STEP 5: DELETE (id 지정) ===")
print(f"삭제 전 개수: {collection.count()}")

collection.delete(ids=["log_005"])

print(f"log_005 삭제 후 개수: {collection.count()}\n")

# ─────────────────────────────────────────────────────────────
# STEP 6: DELETE — where 조건으로 삭제 (단일 조건)
# "처리 완료된(resolved=True) 로그 전체 정리"
# ─────────────────────────────────────────────────────────────
print("=== STEP 6: DELETE (where 단일 조건) ===")
print(f"삭제 전 개수: {collection.count()}")

collection.delete(where={"resolved": True})

print(f"resolved=True 전체 삭제 후 개수: {collection.count()}")

remaining = collection.get()
print("남은 로그 (미해결 건만):")
for id_, doc, meta in zip(
    remaining["ids"], remaining["documents"], remaining["metadatas"]
):
    print(f"  {id_} | {meta['risk_level']} | {doc[:40]}")
print()

# ─────────────────────────────────────────────────────────────
# STEP 7: DELETE — where 복합 조건 ($and)
# "위험 등급이면서 특정 구역 로그만 삭제"
# ─────────────────────────────────────────────────────────────
print("=== STEP 7: DELETE ($and 복합 조건) ===")
print(f"삭제 전 개수: {collection.count()}")

collection.delete(where={
    "$and": [
        {"risk_level": "위험"},
        {"location": "공장 외곽"},
    ]
})

print(f"위험+공장외곽 삭제 후 개수: {collection.count()}\n")

# ─────────────────────────────────────────────────────────────
# STEP 8: 컬렉션 전체 초기화 방법 두 가지
# ─────────────────────────────────────────────────────────────
print("=== STEP 8: 컬렉션 전체 초기화 ===")

# 방법 A: 남은 id 전부 조회 후 삭제
# → 컬렉션 구조(메타데이터, 설정)는 유지하면서 내용만 비울 때
all_ids = collection.get()["ids"]
if all_ids:
    collection.delete(ids=all_ids)
print(f"방법 A (id 전체 삭제) — 개수: {collection.count()}")

# 방법 B: 컬렉션 자체를 삭제 후 재생성
# → 설정까지 완전히 초기화할 때
chroma_client.delete_collection(COLLECTION_NAME)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
print(f"방법 B (컬렉션 재생성) — 개수: {collection.count()}")
print()

