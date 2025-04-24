import os
import re
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

# 답변 프롬프트
ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
당신은 학술 논문 기반 지식 도우미입니다. 아래의 영어 논문 내용을 참고하여, 한국어로 정확하고 학술적인 답변을 작성하세요.

[규칙]
1. 제공된 논문 내용에서만 정보를 사용하세요.
2. 답을 찾을 수 없다면 모른다고 답하세요.
3. 주장 또는 주요 정보에는 반드시 인용 `({{citation}})` 형태로 표시하세요.
4. 응답은 반드시 한국어로 작성하세요.

[논문 내용]
{context}

[질문]
{question}
"""
)

# citation id 리스트 추출
def extract_citation_ids(docs: List[Document]) -> List[str]:
    return list({doc.metadata.get("citation") for doc in docs if "citation" in doc.metadata})

# 로컬 파일 경로 추출
def extract_pdf_paths(docs: List[Document]) -> List[str]:
    return list({doc.metadata.get("pdf_path") for doc in docs if "pdf_path" in doc.metadata})

# citation 문자열 치환
def process_citations(text: str, citation_ids: List[str]) -> str:
    for citation_id in citation_ids:
        parts = citation_id.split("_")
        if len(parts) >= 2:
            author, year = parts[0], parts[1]
            formatted = f"{author} et al., {year}"
            text = text.replace(f"({{{{citation_id}}}})", f"({formatted})")
    return text

# 질문에 따라 Gemini 온도 설정
def detect_temperature(question: str) -> float:
    if re.search(r"(정의|뜻|무엇|어떤 의미)", question):
        return 0.0
    elif re.search(r"(비교|차이|좋은|효과|장점|단점|의견|생각)", question):
        return 0.5
    return 0.3

# 메인 클래스
class LLMAnswerGenerator:
    def __init__(self, api_key=API_KEY, model_name=MODEL_NAME):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0.3,
            max_output_tokens=1024
        )

    def estimate_k(self, question: str) -> int:
        prompt = f"""
질문: "{question}"
이 질문을 바탕으로 얼마나 많은 관련 논문 조각(context)을 활용해야 가장 좋은 답을 만들 수 있을지 판단해줘.
가능한 값은 20, 30, 50 중 하나입니다. 단순히 많이 주는 게 아니라 질문이 구체적인지, 복잡한 비교나 논쟁을 요구하는지를 고려해서 결정해줘.
답변 형식은 숫자 하나만 출력해줘. 예: 20
"""
        try:
            response = self.llm.invoke(prompt)
            parsed = int(re.findall(r"\\d+", response.content.strip())[0])
            return parsed if parsed in [20, 30, 50] else 20
        except:
            return 15

    def generate_answer(self, question: str, docs: List[Document], context_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, any]:
        context_text = "\n\n".join([
            f"[출처: {doc.metadata.get('citation', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

        citation_ids = extract_citation_ids(docs)
        pdf_paths = extract_pdf_paths(docs)
        temperature = detect_temperature(question)
        self.llm.temperature = temperature

        # 대화 맥락 반영 프롬프트 구성
        chat_history_text = ""
        if context_history:
            for turn in context_history:
                chat_history_text += f"\n[이전 질문]\n{turn['user']}\n[이전 답변]\n{turn['assistant']}"

        prompt_text = f"""
{chat_history_text}

[논문 내용]
{context_text}

[질문]
{question}
"""

        prompt = ANSWER_PROMPT.format(question=question, context=context_text)

        print(f"🧠 Gemini 호출 중 (temperature={temperature})...")
        response = self.llm.invoke(prompt)
        answer = process_citations(response.content.strip(), citation_ids)

        return {
            "answer": answer,
            "citations": citation_ids,
            "pdf_paths": pdf_paths
        }

def is_meta_request(query: str) -> bool:
    return "제목" in query and ("질문할 수 있는" in query or "가능한 논문" in query or "등록된 논문" in query)