import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import torch

# 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ .env 파일에 HUGGINGFACEHUB_API_TOKEN이 설정되어야 합니다.")

# 경로 설정
VECTOR_DB_DIR = "data/vector_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

class VectorStoreBuilder:
    def __init__(self, persist_directory=VECTOR_DB_DIR):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "use_auth_token": HF_TOKEN
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

    def build_from_documents(self, documents: List[Document], overwrite=False):
        if overwrite:
            print("⚠️ 기존 벡터 저장소를 초기화합니다.")
            import shutil
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )

        print(f"💾 벡터 저장 중... 총 문서 수: {len(documents)}")
        batch_size = 500  # Chroma 제한을 고려한 배치 크기 설정
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.vectorstore.add_documents(batch)
        self.vectorstore.persist()
        print(f"✅ 저장 완료: {self.persist_directory}")

    def load_vectorstore(self):
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

def list_all_titles(persist_directory="data/vector_db") -> List[str]:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import os
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu", "use_auth_token": HF_TOKEN},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    all_metadata = vectorstore.get()["metadatas"]
    unique_titles = sorted(set(
        meta.get("title", "").strip()
        for meta in all_metadata
        if meta.get("title") and meta["title"] != "Unknown"
    ))

    return unique_titles
