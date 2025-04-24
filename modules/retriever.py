import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import torch

# 환경 변수 로드
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

    def query_similar_documents(self, query: str, k: int = 10) -> List[Document]:
        print(f"🔍 유사한 청크 {k}개 검색 중...")
        results = self.vectorstore.similarity_search(query, k=k)
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.metadata.get('title', '')} / {doc.metadata.get('citation', '')}")
        return results

    def query_documents_by_title(self, title: str) -> List[Document]:
        print(f"📄 제목으로 전체 문서 검색 중: {title}")
        all_docs = self.vectorstore.get(include=['documents', 'metadatas'])
        filtered = []
        for i, metadata in enumerate(all_docs["metadatas"]):
            if metadata.get("title", "").strip().lower() == title.strip().lower():
                filtered.append(Document(page_content=all_docs["documents"][i], metadata=metadata))
        print(f"✅ 검색된 문서 수: {len(filtered)}")
        return filtered

    @staticmethod
    def list_all_titles(vector_db_path: str) -> List[str]:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu", "use_auth_token": HF_TOKEN},
            encode_kwargs={"normalize_embeddings": True}
        )
        db = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)
        docs = db.similarity_search("dummy", k=1000)

        seen_titles = set()
        unique_titles = []

        for doc in docs:
            title = doc.metadata.get("title", "Unknown").strip()
            authors = doc.metadata.get("authors", "").strip()
            year = doc.metadata.get("year", "n.d.")
            key = f"{title}_{authors}"

            if key not in seen_titles:
                seen_titles.add(key)
                unique_titles.append(f'"{title}" ({authors}, {year})')

        return unique_titles