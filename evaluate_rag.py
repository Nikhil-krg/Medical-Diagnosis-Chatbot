"""
evaluate_rag.py

Automated LLM evaluation pipeline for the Medical Diagnosis Chatbot.
Measures faithfulness, answer relevancy, and context precision using RAGAS
over a 50-question test set (eval_dataset.json).

Usage:
    python evaluate_rag.py

Requirements:
    GROQ_API_KEY must be set as an environment variable.

Output:
    - Per-question scores printed to console
    - Aggregate scores saved to eval_results.json
"""

import os
import json
import time
import logging
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_FAISS_PATH      = "vectorstore/db_faiss"
EVAL_DATASET_PATH  = "eval_dataset.json"
RESULTS_PATH       = "eval_results.json"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL         = "meta-llama/llama-4-maverick-17b-128e-instruct"
RETRIEVER_K        = 3          # chunks retrieved per question
REQUEST_DELAY      = 1.5        # seconds between Groq API calls (rate-limit safety)
TARGET_ACCURACY    = 0.85       # threshold reported in project description

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_eval_dataset(path: str) -> list[dict]:
    """Load the 50-question evaluation dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d questions from %s", len(data), path)
    return data


def build_qa_chain(groq_api_key: str) -> RetrievalQA:
    """Initialise the RAG chain (same configuration as chatbot.py)."""
    log.info("Loading FAISS vector store from %s …", DB_FAISS_PATH)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    llm = ChatGroq(
        model_name=GROQ_MODEL,
        temperature=0.0,
        groq_api_key=groq_api_key,
    )

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    log.info("RAG chain ready.")
    return chain


def run_inference(chain: RetrievalQA, questions: list[str]) -> tuple[list[str], list[list[str]]]:
    """
    Run the RAG chain over all questions.
    Returns:
        answers  – LLM-generated answer strings
        contexts – list of retrieved chunk texts per question
    """
    answers: list[str] = []
    contexts: list[list[str]] = []

    total = len(questions)
    for idx, question in enumerate(questions, start=1):
        log.info("[%d/%d] Querying: %s", idx, total, question[:80])
        try:
            response = chain.invoke({"query": question})
            answers.append(response["result"])
            contexts.append([doc.page_content for doc in response["source_documents"]])
        except Exception as exc:
            log.warning("  ⚠  Error on question %d: %s — using empty fallback.", idx, exc)
            answers.append("")
            contexts.append([""])

        # Respect Groq rate limits between requests
        if idx < total:
            time.sleep(REQUEST_DELAY)

    return answers, contexts


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    groq_api_key: str,
) -> dict:
    """
    Run RAGAS evaluation on the collected Q/A/context triples.
    Metrics:
        - faithfulness       (answer grounded in retrieved context)
        - answer_relevancy   (answer relevant to the question)
        - context_precision  (retrieved context relevant to the question + ground truth)
    """
    log.info("Building RAGAS dataset …")
    dataset = Dataset.from_dict(
        {
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        }
    )

    # Use the same Groq LLM as the judge model to avoid needing OpenAI keys
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model_name=GROQ_MODEL,
            temperature=0.0,
            groq_api_key=groq_api_key,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    log.info("Running RAGAS evaluation (this may take a few minutes) …")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )

    return result


def compute_per_question_scores(ragas_result) -> list[dict]:
    """Extract per-question scores from the RAGAS result dataframe."""
    df = ragas_result.to_pandas()
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "question":          row.get("question", ""),
                "faithfulness":      round(float(row.get("faithfulness",      0) or 0), 4),
                "answer_relevancy":  round(float(row.get("answer_relevancy",  0) or 0), 4),
                "context_precision": round(float(row.get("context_precision", 0) or 0), 4),
            }
        )
    return records


def print_summary(aggregate: dict, per_question: list[dict]) -> None:
    """Pretty-print the evaluation summary to the console."""
    print("\n" + "=" * 65)
    print("  RAGAS EVALUATION RESULTS — Medical Diagnosis Chatbot")
    print("=" * 65)
    print(f"  Questions evaluated  : {len(per_question)}")
    print(f"  Faithfulness         : {aggregate['faithfulness']:.4f}  (target ≥ {TARGET_ACCURACY})")
    print(f"  Answer Relevancy     : {aggregate['answer_relevancy']:.4f}  (target ≥ {TARGET_ACCURACY})")
    print(f"  Context Precision    : {aggregate['context_precision']:.4f}  (target ≥ {TARGET_ACCURACY})")
    print(f"  Overall Accuracy     : {aggregate['overall_accuracy']:.4f}")

    passed = aggregate["overall_accuracy"] >= TARGET_ACCURACY
    status = "✅  TARGET MET" if passed else "❌  BELOW TARGET"
    print(f"\n  {status} (threshold = {TARGET_ACCURACY})")
    print("=" * 65)

    print("\n  Per-question breakdown (first 10):")
    print(f"  {'#':<4} {'Faith':>8} {'AnsRel':>8} {'CtxPrec':>9}  Question")
    print("  " + "-" * 61)
    for i, row in enumerate(per_question[:10], start=1):
        q_short = row["question"][:42] + ("…" if len(row["question"]) > 42 else "")
        print(
            f"  {i:<4} {row['faithfulness']:>8.4f} "
            f"{row['answer_relevancy']:>8.4f} "
            f"{row['context_precision']:>9.4f}  {q_short}"
        )
    if len(per_question) > 10:
        print(f"  … and {len(per_question) - 10} more (see {RESULTS_PATH})")
    print()


def save_results(aggregate: dict, per_question: list[dict]) -> None:
    """Persist full results to eval_results.json."""
    output = {
        "metadata": {
            "timestamp":          datetime.utcnow().isoformat() + "Z",
            "model":              GROQ_MODEL,
            "embedding_model":    EMBEDDING_MODEL,
            "retriever_k":        RETRIEVER_K,
            "num_questions":      len(per_question),
            "target_accuracy":    TARGET_ACCURACY,
        },
        "aggregate_scores": aggregate,
        "per_question_scores": per_question,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", RESULTS_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Export it as an environment variable before running this script."
        )

    # 1. Load evaluation dataset
    eval_data     = load_eval_dataset(EVAL_DATASET_PATH)
    questions     = [item["question"]     for item in eval_data]
    ground_truths = [item["ground_truth"] for item in eval_data]

    # 2. Build RAG chain and run inference
    chain = build_qa_chain(groq_api_key)
    log.info("Running inference over %d questions …", len(questions))
    answers, contexts = run_inference(chain, questions)

    # 3. RAGAS evaluation
    ragas_result = run_ragas_evaluation(
        questions, answers, contexts, ground_truths, groq_api_key
    )

    # 4. Aggregate scores
    per_question = compute_per_question_scores(ragas_result)

    faithfulness_score      = sum(r["faithfulness"]      for r in per_question) / len(per_question)
    answer_relevancy_score  = sum(r["answer_relevancy"]  for r in per_question) / len(per_question)
    context_precision_score = sum(r["context_precision"] for r in per_question) / len(per_question)
    overall_accuracy        = (faithfulness_score + answer_relevancy_score + context_precision_score) / 3

    aggregate = {
        "faithfulness":      round(faithfulness_score,      4),
        "answer_relevancy":  round(answer_relevancy_score,  4),
        "context_precision": round(context_precision_score, 4),
        "overall_accuracy":  round(overall_accuracy,        4),
    }

    # 5. Display and persist
    print_summary(aggregate, per_question)
    save_results(aggregate, per_question)


if __name__ == "__main__":
    main()
