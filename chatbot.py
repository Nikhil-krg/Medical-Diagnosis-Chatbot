"""Medical Diagnosis Chatbot — Streamlit UI."""

import json
import os

import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from config import (
    DEFAULT_MODEL, GROQ_MODELS, TARGET_ACCURACY,
    EVAL_DATASET, RESULTS_PATH,
)
from rag import build_qa_chain
from evaluation import run_evaluation


# ── Chatbot tab ───────────────────────────────────────────────────────────────

def render_chatbot_tab():
    st.header("Medical Diagnosis Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Ask a medical question...")
    if not prompt:
        return

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        model = st.session_state.get("selected_model", DEFAULT_MODEL)
        qa = build_qa_chain(model=model)
        resp = qa.invoke({"query": prompt})
        answer = resp["result"] + "\n\n**Source Docs:**\n" + str(resp["source_documents"])
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.error(f"Error: {exc}")


# ── Evaluation tab ────────────────────────────────────────────────────────────

def render_evaluation_tab():
    st.header("RAG Evaluation Pipeline")
    st.markdown(
        "RAGAS evaluation over the medical test set. Measures **faithfulness**, "
        f"**answer relevancy**, and **context precision** (target: **{TARGET_ACCURACY}**). "
        "All LLM calls use `temperature=0` for deterministic scores."
    )

    # Show previous results
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        agg, meta = cached["aggregate_scores"], cached["metadata"]

        st.subheader("Last Evaluation Results")
        st.caption(
            f"{meta.get('timestamp', '?')} · Model: {meta.get('model', '?')} · "
            f"{meta.get('num_questions', '?')} questions"
        )

        c1, c2, c3, c4 = st.columns(4)
        for col, key, label in [
            (c1, "faithfulness", "Faithfulness"),
            (c2, "answer_relevancy", "Answer Relevancy"),
            (c3, "context_precision", "Context Precision"),
            (c4, "overall_accuracy", "Overall Accuracy"),
        ]:
            col.metric(label, f"{agg[key]:.4f}",
                       delta="Target met" if agg[key] >= TARGET_ACCURACY else "Below target")

        with st.expander("Per-question breakdown"):
            per_q = cached.get("per_question_scores", [])
            if per_q:
                df = pd.DataFrame(per_q)[["question", "faithfulness", "answer_relevancy", "context_precision"]]
                df.index += 1
                st.dataframe(df, use_container_width=True)

        with open(RESULTS_PATH, "rb") as f:
            st.download_button("Download eval_results.json", f, "eval_results.json", "application/json")
        st.divider()

    # Run new evaluation
    st.subheader("Run New Evaluation")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.warning("GROQ_API_KEY not set. Add it to .env and restart.")
        return

    with open(EVAL_DATASET, encoding="utf-8") as f:
        n = len(json.load(f))

    st.info(f"Dataset: **{n} questions** available.")
    num_samples = st.slider("Test cases to evaluate", 1, n, min(10, n),
                            help="Fewer = faster and fewer API tokens.")
    st.caption(f"Selected **{num_samples}** of {n} · ~{num_samples * 2} Groq calls · "
               f"~{num_samples * 2}-{num_samples * 4} min")

    if st.button("Run Evaluation", type="primary"):
        model = st.session_state.get("selected_model", DEFAULT_MODEL)
        bar = st.progress(0.0, text="Starting...")
        status = st.empty()
        try:
            run_evaluation(
                api_key, model, num_samples,
                progress_cb=lambda v: bar.progress(min(v, 1.0)),
                status_cb=lambda m: status.text(m),
            )
            st.success("Evaluation complete! Results saved to eval_results.json.")
            st.rerun()
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Medical Diagnosis Chatbot", page_icon="\U0001fa7a", layout="wide")

    with st.sidebar:
        st.title("Settings")
        current = st.session_state.get("selected_model", DEFAULT_MODEL)
        selected = st.selectbox("Groq model", GROQ_MODELS, index=GROQ_MODELS.index(current))
        if selected != current:
            st.session_state["selected_model"] = selected
            st.session_state["messages"] = []
        st.caption(f"Active: `{selected}`")
        st.divider()
        st.markdown("**Tip:** smaller models (20b, 7b) are faster and use fewer tokens.")

    chat_tab, eval_tab = st.tabs(["\U0001f4ac Chatbot", "\U0001f4ca Evaluation"])
    with chat_tab:
        render_chatbot_tab()
    with eval_tab:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
