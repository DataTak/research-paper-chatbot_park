from modules.retriever import Retriever
from modules.llm_chain import LLMAnswerGenerator

retriever = Retriever()
docs = retriever.query_similar_documents("청년 부채가 자아존중감에 미치는 영향은?")
llm = LLMAnswerGenerator()
answer = llm.generate_answer("청년 부채가 자아존중감에 미치는 영향은?", docs)

print("\n📚 최종 응답:")
print(answer)