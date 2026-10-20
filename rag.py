"""RAG chain — vectorstore loading and QA chain construction."""

import os

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA

from config import DB_FAISS_PATH, EMBEDDING_MODEL, DEFAULT_MODEL, PROMPT_TEMPLATE, RETRIEVER_K


@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


def build_qa_chain(model: str = DEFAULT_MODEL, api_key: str | None = None):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name=model,
            temperature=0.0,
            groq_api_key=api_key or os.environ["GROQ_API_KEY"],
        ),
        chain_type="stuff",
        retriever=get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
