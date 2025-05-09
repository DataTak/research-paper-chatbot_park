import os
import streamlit as st
from modules.retriever import Retriever
from modules.llm_chain import LLMAnswerGenerator, is_meta_request
from modules.vector_store_builder import list_all_titles
from utils.s3_utils import S3Manager
from dotenv import load_dotenv

load_dotenv()

# 페이지 설정
st.set_page_config(page_title="📚 학술 논문 기반 챗봇", page_icon="📘")

# S3 매니저 초기화 및 DB 다운로드
def initialize_s3_and_db():
    if 's3_manager' not in st.session_state:
        st.session_state.s3_manager = S3Manager()
    
    db_path = "data/temp_vector_db/chroma.sqlite3"
    st.session_state.s3_manager.download_db_if_needed(db_path)
    return st.session_state.s3_manager, db_path

s3_manager, db_path = initialize_s3_and_db()

st.title("📘 박박사님 경제논문 저장소 챗봇")
st.markdown("영어 논문을 기반으로 한국어로 정확하고 출처가 달린 답변을 제공합니다.\n등록된 논문들이 궁금하면 \"등록된 논문 제목 알려줘\" 라고 물어보세요.")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Retriever 및 LLM 초기화
retriever = Retriever()
llm = LLMAnswerGenerator()

# 사이드바
with st.sidebar:
    st.header("🔧 시스템 정보")
    st.markdown("- 검색 모델: `BAAI/bge-m3`")
    st.markdown("- 답변 모델: `Gemini-2.0-flash`")
    st.markdown("- 저장소: `AWS S3 + Chroma`")
    
    if st.button("💾 대화 초기화"):
        st.session_state.chat_history = []
        
    if st.button("�� DB 새로고침"):
        st.session_state.s3_manager.download_db_if_needed(db_path, force_download=True)
        st.success("DB가 새로고침되었습니다!")

# 기존 대화 표시
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# 사용자 질문 입력
query = st.chat_input("논문에서 어떤 부분이 궁금한가요?")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 관련 논문을 찾고, 학술적으로 답변 중입니다..."):
            if is_meta_request(query):
                titles = list_all_titles("data/vector_db/")
                answer = "### 📚 현재 등록된 논문 목록:\n" + "\n".join([f"- {title}" for title in titles])
            else:
                # 대화 맥락 수집
                context_history = [
                    {"user": h["content"], "assistant": st.session_state.chat_history[i+1]["content"]}
                    for i, h in enumerate(st.session_state.chat_history[:-1]) if h["role"] == "user"
                ]

                # 등록된 제목과 정확히 일치하는 경우 전체 논문으로 추출
                all_titles = list_all_titles("data/vector_db/")
                matched_title = next((t for t in all_titles if t.strip('"').lower() in query.lower()), None)

                if matched_title:
                    st.info(f"📄 논문 전체로 분석 중: {matched_title}")
                    docs = retriever.query_documents_by_title(matched_title.strip('"'))
                else:
                    # 일반 검색 + k 추정
                    k = llm.estimate_k(query)
                    docs = retriever.query_similar_documents(query, k=k)

                result = llm.generate_answer(query, docs, context_history=context_history)
                answer = result["answer"]

        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})