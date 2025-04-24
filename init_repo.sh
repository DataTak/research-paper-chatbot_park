#!/bin/bash
git init
git remote add origin https://github.com/DataTak/research-paper-chatbot_park
touch .gitignore requirements.txt README.md
mkdir -p modules data logs .streamlit
echo '__pycache__/
.env
data/vector_db/
logs/' > .gitignore
echo '# Research Paper Chatbot

A Korean-English academic chatbot for research paper Q&A using RAG with LangChain + Gemini + ChromaDB.' > README.md
echo 'streamlit
langchain
langchain_community
langchain_google_genai
python-dotenv
chromadb
torch
sentence-transformers' > requirements.txt
