from modules.retriever import Retriever

retriever = Retriever(k=3)
query = "청년의 부채가 자아존중감에 미치는 영향은?"
docs = retriever.query_similar_documents(query)

print(f"✅ 검색된 청크 수: {len(docs)}")
print(docs[0].page_content[:300])  # 첫 청크 일부 출력
