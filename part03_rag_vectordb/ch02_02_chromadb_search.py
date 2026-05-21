# ChromaDB에 탐지 로그를 저장하고 유사 상황을 검색해 봅시다

import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()