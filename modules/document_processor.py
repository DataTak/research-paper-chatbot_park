import os
import uuid
import json
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai

# Load .env settings
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
if not API_KEY:
    raise ValueError("❌ .env 파일에 GEMINI_API_KEY가 설정되지 않았습니다.")


class MetadataExtractorLLM:
    """Gemini SDK 기반 논문 메타데이터 추출기 (최대 3페이지 시도, 점진적 정보 수집)"""
    def __init__(self, model_name=MODEL_NAME, api_key=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "string"},
                        "year": {"type": "string"},
                        "keywords": {"type": "string"},
                        "journal": {"type": "string"},
                        "abstract": {"type": "string"},
                        "doi": {"type": "string"},
                    },
                    "required": ["title", "authors", "year"]
                }
            )
        )

    def extract_metadata(self, pages: List[str]) -> dict:
        """1~3페이지 중에서 앞쪽 정보 우선, 점진적으로 보완"""
        priority_keys = ["title", "authors", "year", "abstract"]
        optional_keys = ["keywords", "journal", "doi"]
        collected = {key: "" for key in priority_keys + optional_keys}

        for i, page_text in enumerate(pages[:3]):
            print(f"🔍 메타데이터 추출 시도 (페이지 {i+1})...")
            try:
                response = self.model.generate_content(
                    f"""
    다음은 논문 일부입니다. 아래 정보를 가능한 한 정확히 추출해서 JSON 형식으로 응답해 주세요:

    - title
    - authors (쉼표로 구분)
    - year
    - keywords (쉼표로 구분)
    - journal
    - abstract
    - doi

    [논문 텍스트]
    {page_text}
    """, request_options={"timeout": 60})

                raw_json = response.text.strip()
                print("📩 Gemini 응답 (앞부분):", raw_json[:100])
                parsed = json.loads(raw_json)

                for key in collected:
                    val = parsed.get(key, "").strip()
                    if not collected[key] and val.lower() not in ["", "unknown", "n.d.", "n/a"]:
                        collected[key] = val

                if all(collected[k] for k in priority_keys):
                    break

            except Exception as e:
                print(f"⚠️ LLM 추출 실패 (페이지 {i+1}): {e}")

        # 필수 필드 보완
        for key in priority_keys:
            if not collected[key]:
                collected[key] = "Unknown" if key != "year" else "n.d."

        return collected



class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.extractor = MetadataExtractorLLM(api_key=API_KEY)

    def load_documents_from_folder(self, folder_path: str) -> List:
        all_chunks = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    full_path = os.path.join(root, file)
                    chunks = self._process_single_file(full_path)
                    all_chunks.extend(chunks)
        return all_chunks

    def _process_single_file(self, filepath: str) -> List:
        loader = PyPDFLoader(filepath)
        raw_docs = loader.load()

        full_text = "\n".join([doc.page_content for doc in raw_docs])
        pages = [doc.page_content for doc in raw_docs[:3]]

        metadata = self.extractor.extract_metadata(pages)
        doc_id = str(uuid.uuid4())

        # ❗ 메타데이터 이상 여부 로그 기록
        if metadata["title"].lower() in ["", "unknown", "n/a"] or metadata["authors"].lower() in ["", "unknown", "n/a"]:
            print(f"⚠️ 메타데이터 오류 - 파일: {os.path.basename(filepath)}")
            print(f"    ⮕ title: {metadata['title']}")
            print(f"    ⮕ authors: {metadata['authors']}")
            os.makedirs("logs", exist_ok=True)
            with open("logs/metadata_failures.log", "a", encoding="utf-8") as f:
                f.write(f"{os.path.basename(filepath)}\n")

        chunks = self.splitter.create_documents([full_text])
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "title": metadata.get("title"),
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "keywords": metadata.get("keywords"),
                "journal": metadata.get("journal"),
                "abstract": metadata.get("abstract"),
                "doi": metadata.get("doi"),
                "doc_id": doc_id,
                "citation": f'{metadata.get("authors").split(",")[0].strip()}_{metadata.get("year")}_Page_{i+1}',
                "pdf_path": filepath
            }

        return chunks