# 탐지 로그 텍스트를 OpenAI Embedding으로 벡터화하고
# 코사인 유사도를 직접 계산해서 비슷한 상황을 찾습니다.

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env에서 OPENAI_API_KEY 읽기

client = OpenAI()  # API 키는 환경변수에서 자동으로 읽힘

def get_embedding(text : str) -> np.array :
  """
    OpenAI text-embedding-3-small 모델로 텍스트를 벡터로 변환합니다.
    반환값: 1536차원 숫자 배열 (numpy array)

    text-embedding-3-small: 빠르고 저렴한 임베딩 모델
    text-embedding-3-large: 더 정확하지만 느리고 비쌈 (3072차원)
  """
  response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = text.strip() # 혹시 모를 앞뒤 공백 제거
  )

  #response.data[0].embedding -> 숫자 리스트 -> numpy 배열로 변환
  return np.array(response.data[0].embedding)

def cosine_similarity(a:np.ndarray, b:np.ndarray) -> float :
  """
    코사인 유사도 계산.
    수식: cos(θ) = (A · B) / (|A| × |B|)

    내적(dot): 두 벡터가 같은 방향을 얼마나 가리키는지
    norm(크기): 벡터의 길이 (크기 차이를 제거하기 위해 나눔)
    결과: -1.0 ~ 1.0 (1에 가까울수록 의미가 유사)
  """
  dot = np.dot(a, b)
  norm = np.linalg.norm(a) * np.linalg.norm(b)
  if norm == 0 :
    return 0.0
  return dot / norm

# ─── 과거 탐지 로그 텍스트 ────────────────────────────────────
log_texts = [
    "새벽 2시 창고 출입구에서 person:2 / car:1 탐지. 침입 시도 의심.",
    "새벽 3시 공장 외곽에서 person:3 / car:2 탐지. 침입 후 도주.",
    "오후 2시 주차장에서 person:1 탐지. 정상 이용객으로 판단.",
    "심야 배회 감지. 창고 주변에서 person:1 반복 배회.",
    "낮 12시 정문 앞 person:5 집단 탐지. 정상 회의 참석자.",
]

# ─── 현재 상황 쿼리 ──────────────────────────────────────────
if __name__ == "__main__":
    query = "새벽 창고 침입 의심. person:2 탐지."

    print(f"[검색 상황] 현재 상황: {query}\n")
    print("[API] OpenAI Embedding API 호출 중...")

    # 쿼리와 모든 로그를 벡터로 변환
    query_vec = get_embedding(query)
    log_vecs = [get_embedding(log) for log in log_texts]

    # 유사도 계산 및 정렬
    results = []
    for log, log_vecs in zip(log_texts, log_vecs) :
      similarity = cosine_similarity(query_vec, log_vecs)
      results.append((similarity, log))

    # 유사도를 기준으로 내림차순 정렬
    results.sort(key= lambda x :x[0], reverse=True)

    print("\n[결과] 유사도 검색 결과 (높은 순):")
    for i, (sim, log) in enumerate(results, 1):
        bar = "o" * int(sim * 20)   # 유사도를 막대 그래프로 시각화 (이모지 및 특수문자 깨짐 방지)
        print(f"  {i}위 (유사도: {sim:.3f}) {bar}")
        print(f"     → {log}")
        print()
  

  