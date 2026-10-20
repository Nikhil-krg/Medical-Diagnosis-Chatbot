"""Central config — paths, models, prompts, and constants."""

DB_FAISS_PATH   = "vectorstore/db_faiss"
EVAL_DATASET    = "eval_dataset.json"
RESULTS_PATH    = "eval_results.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL   = "openai/gpt-oss-120b"
GROQ_MODELS     = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]
TARGET_ACCURACY = 0.85
REQUEST_DELAY   = 3.0
RETRIEVER_K     = 3

PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Score column names produced by ragas 0.4.3
SCORE_COLS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]
