import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
VECTOR_DB_DIR = "data/vector_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

class Retriever:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "use_auth_token": HF_TOKEN
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embedding_model
        )

    # 기본 검색 (단순 유사도 기반)
    def query_similar_documents(self, query: str, k: int = 10) -> List[Document]:
        print(f"🔍 유사도 기반 검색 (Top-{k}) 실행 중...")
        results = self.vectorstore.similarity_search(query, k=k)
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.metadata.get('title', '')} / {doc.metadata.get('citation', '')}")
        return results

    # score 기반 필터링 (유사도 점수 기준 제거)
    def query_with_score_threshold(self, query: str, k: int = 20, threshold: float = 0.5) -> List[Document]:
        print(f"🔎 Score Threshold 검색 (Top-{k}, threshold ≥ {threshold})...")
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        filtered = []
        for doc, score in results_with_scores:
            if score >= threshold:
                filtered.append(doc)
                print(f"[✓] {doc.metadata.get('title', '')} / Score: {score:.4f}")
            else:
                print(f"[✗] Skipped low score: {score:.4f}")
        return filtered

    # MMR 기반 다양성 중심 검색
    def query_mmr_documents(self, query: str, k: int = 10, lambda_mult: float = 0.5) -> List[Document]:
        print(f"🧠 MMR 검색 실행 (Top-{k}, λ={lambda_mult})...")
        retriever: VectorStoreRetriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "lambda_mult": lambda_mult}
        )
        results = retriever.get_relevant_documents(query)
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.metadata.get('title', '')} / {doc.metadata.get('citation', '')}")
        return results

    # 타이틀 기반 필터
    def query_documents_by_title(self, title: str) -> List[Document]:
        print(f"📄 제목으로 문서 필터링: {title}")
        all_docs = self.vectorstore.get(include=['documents', 'metadatas'])
        filtered = []
        for i, metadata in enumerate(all_docs["metadatas"]):
            if metadata.get("title", "").strip().lower() == title.strip().lower():
                filtered.append(Document(page_content=all_docs["documents"][i], metadata=metadata))
        print(f"✅ 검색된 문서 수: {len(filtered)}")
        return filtered

    # 사용자가 입력한 query 리라이팅 (단순형)
    def rewrite_query(self, query: str) -> str:
        return f"이 질문은 다음과 같은 의미를 가집니다: {query}. 관련 연구 논문이나 사례 중심으로 검색해 주세요."

    # 전체 타이틀 목록
    @staticmethod
    def list_all_titles(vector_db_path: str) -> List[str]:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu", "use_auth_token": HF_TOKEN},
            encode_kwargs={"normalize_embeddings": True}
        )
        db = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)
        all_docs = db.get(include=["metadatas"])

        seen = set()
        unique_titles = []

        for metadata in all_docs["metadatas"]:
            title = metadata.get("title", "Unknown").strip()
            authors = metadata.get("authors", "").strip()
            year = metadata.get("year", "n.d.")
            key = f"{title}_{authors}"

            if key not in seen:
                seen.add(key)
                unique_titles.append(f'"{title}" ({authors}, {year})')

        return unique_titles
