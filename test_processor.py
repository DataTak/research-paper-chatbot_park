from modules.document_processor import DocumentProcessor

processor = DocumentProcessor()
chunks = processor.load_documents_from_folder("data/papers")

print(f"총 청크 수: {len(chunks)}")

# 논문별로 정리된 메타데이터 추출 (doc_id 기준 그룹핑)
from collections import defaultdict
metadata_by_doc = defaultdict(list)

for chunk in chunks:
    metadata_by_doc[chunk.metadata["doc_id"]].append(chunk.metadata)

# 중복 제거한 문서 단위 메타데이터만 출력
print("\n📄 [논문별 메타데이터 추출 결과 요약]\n")
for i, (doc_id, metas) in enumerate(metadata_by_doc.items(), start=1):
    meta = metas[0]  # 같은 논문은 메타데이터가 동일함
    print(f"[{i}] 제목: {meta['title']}")
    print(f"    저자: {meta['authors']}")
    print(f"    연도: {meta['year']}")
    print(f"    키워드: {meta['keywords']}")
    print(f"    초록: {meta['abstract'][:100]}...")  # 앞 100자만 요약
    print(f"    DOI: {meta['doi']}")
    print("-" * 80)
