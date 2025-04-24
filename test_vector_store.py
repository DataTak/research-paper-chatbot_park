from modules.document_processor import DocumentProcessor
from modules.vector_store_builder import VectorStoreBuilder

processor = DocumentProcessor()
chunks = processor.load_documents_from_folder("data/papers")

builder = VectorStoreBuilder()
builder.build_from_documents(chunks, overwrite=True)  # 새로 저장