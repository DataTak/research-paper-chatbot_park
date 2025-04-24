import os
import shutil
from modules.document_processor import DocumentProcessor
from modules.vector_store_builder import VectorStoreBuilder

SOURCE_DIR = "./newpapers"
TARGET_DIR = "./data/papers"

def move_new_papers():
    if not os.path.exists(SOURCE_DIR):
        print("❌ 'newpapers' 폴더가 존재하지 않습니다.")
        return []

    moved = []
    for file in os.listdir(SOURCE_DIR):
        if file.endswith(".pdf"):
            src_path = os.path.join(SOURCE_DIR, file)
            dst_path = os.path.join(TARGET_DIR, file)
            if not os.path.exists(dst_path):  # 중복 방지
                shutil.move(src_path, dst_path)
                moved.append(file)

    return moved

def rebuild_vectorstore():
    processor = DocumentProcessor()
    documents = processor.load_documents_from_folder(TARGET_DIR)
    
    builder = VectorStoreBuilder()
    builder.build_from_documents(documents, overwrite=True)
    return len(documents)

if __name__ == "__main__":
    print("📂 새 논문 이동 중...")
    moved_files = move_new_papers()

    if not moved_files:
        print("📭 이동할 새 논문이 없습니다.")
    else:
        print(f"✅ 총 {len(moved_files)}개 논문 이동 완료: {moved_files}")
        print("🧠 Chroma 벡터스토어 재구성 시작...")
        total_docs = rebuild_vectorstore()
        print(f"✅ 벡터 저장소 재구성 완료! 총 문서 수: {total_docs}")