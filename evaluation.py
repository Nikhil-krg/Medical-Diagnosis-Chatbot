"""RAGAS evaluation pipeline — inference pass + metric scoring."""

import json
import time
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from config import (
    EVAL_DATASET, RESULTS_PATH, EMBEDDING_MODEL, DEFAULT_MODEL,
    REQUEST_DELAY, TARGET_ACCURACY, SCORE_COLS,
)
from rag import build_qa_chain


def run_evaluation(api_key: str, model: str, num_samples: int,
                   progress_cb, status_cb) -> dict:
    """Run RAGAS evaluation: inference pass + metric scoring. Returns results dict."""

    with open(EVAL_DATASET, encoding="utf-8") as f:
        eval_data = json.load(f)[:num_samples]
    questions = [d["question"] for d in eval_data]
    ground_truths = [d["ground_truth"] for d in eval_data]
    total = len(questions)

    # Inference pass
    status_cb("Building RAG chain...")
    qa = build_qa_chain(model=model, api_key=api_key)

    answers, contexts = [], []
    for i, q in enumerate(questions, 1):
        status_cb(f"Inference [{i}/{total}]: {q[:60]}...")
        try:
            r = qa.invoke({"query": q})
            answers.append(r["result"])
            contexts.append([d.page_content for d in r["source_documents"]])
        except Exception:
            answers.append("")
            contexts.append([])
        progress_cb(i / (total * 2))
        if i < total:
            time.sleep(REQUEST_DELAY)

    # RAGAS scoring
    status_cb("Running RAGAS metrics...")
    samples = [
        SingleTurnSample(user_input=q, response=a, retrieved_contexts=ctx, reference=gt)
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
    ]
    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model_name=model, temperature=0.0, groq_api_key=api_key)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1),
               LLMContextPrecisionWithReference()]

    result = evaluate(
        dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_emb,
        run_config=RunConfig(timeout=120, max_retries=2, max_wait=30),
        batch_size=1, raise_exceptions=False,
    )
    progress_cb(1.0)

    # Extract scores
    df = result.to_pandas()
    per_q = []
    for _, row in df.iterrows():
        per_q.append({
            "question": str(row.get("user_input", "")),
            "answer": str(row.get("response", "")),
            "faithfulness": round(float(row.get(SCORE_COLS[0], 0) or 0), 4),
            "answer_relevancy": round(float(row.get(SCORE_COLS[1], 0) or 0), 4),
            "context_precision": round(float(row.get(SCORE_COLS[2], 0) or 0), 4),
        })

    if not per_q:
        raise RuntimeError("RAGAS returned no rows.")

    avg = {m: round(sum(r[m] for r in per_q) / len(per_q), 4) for m in
           ["faithfulness", "answer_relevancy", "context_precision"]}
    avg["overall_accuracy"] = round(sum(avg.values()) / 3, 4)

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": model,
            "num_questions": len(per_q),
            "target_accuracy": TARGET_ACCURACY,
        },
        "aggregate_scores": avg,
        "per_question_scores": per_q,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    status_cb("Done")
    return output
