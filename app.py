from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_assistant import (
    CHUNK_CONFIGS,
    answer_with_rag,
    answer_without_rag,
    check_ollama_model,
    create_embeddings,
    create_llm,
    create_vector_store,
    format_sources,
    load_notes,
    split_notes,
)
from evaluation import validate_evaluation_rows


st.set_page_config(page_title="RAG-Powered Knowledge Assistant", page_icon="R", layout="wide")


CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #081827 30%, #0d1b2a 100%);
        color: #ecf6ff;
    }

    .stApp > header {
        display: none;
    }

    .main {
        padding-top: 0.5rem;
    }

    .page-shell {
        max-width: 1200px;
        margin: 0 auto;
        padding-bottom: 2rem;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0 1.5rem;
        color: #d8e9ff;
    }

    .brand-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .brand-mark {
        width: 2.25rem;
        height: 2.25rem;
        border-radius: 0.8rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #67d8ff 0%, #6c7bff 100%);
        color: #07111f;
        font-size: 0.9rem;
        font-weight: 800;
    }

    .nav-links {
        display: flex;
        gap: 1.5rem;
        font-size: 0.9rem;
        color: #b9cbdf;
    }

    .nav-links span {
        opacity: 0.8;
    }

    .nav-actions {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .nav-button {
        border: 1px solid rgba(126, 192, 255, 0.35);
        background: rgba(13, 27, 42, 0.7);
        color: #ddedff;
        border-radius: 999px;
        padding: 0.55rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
        gap: 2rem;
        align-items: center;
        padding: 1.25rem 0 2rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(103, 216, 255, 0.12);
        border: 1px solid rgba(103, 216, 255, 0.25);
        color: #67d8ff;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: clamp(2.6rem, 4vw, 4.4rem) !important;
        line-height: 1.02 !important;
        letter-spacing: -0.06em !important;
        margin: 0 0 1rem !important;
        color: #f3f8ff !important;
    }

    .gradient-text {
        background: linear-gradient(120deg, #7ad7ff 0%, #8bb5ff 30%, #d2c9ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-copy {
        color: #c7d8ee;
        font-size: 1.08rem;
        max-width: 560px;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }

    .cta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.85rem;
        margin-bottom: 1.6rem;
    }

    .primary-btn, .secondary-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.9rem 1.4rem;
        font-weight: 700;
        text-decoration: none;
        border: 1px solid transparent;
        transition: transform 0.2s ease;
    }

    .primary-btn {
        background: linear-gradient(135deg, #7ad7ff 0%, #6c7bff 100%);
        color: #06162a;
        box-shadow: 0 18px 38px rgba(104, 153, 255, 0.35);
    }

    .secondary-btn {
        background: rgba(12, 24, 39, 0.8);
        border-color: rgba(159, 196, 255, 0.28);
        color: #eaf5ff;
    }

    .feature-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        color: #dceafc;
    }

    .feature-chip {
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(126, 142, 255, 0.09);
        border: 1px solid rgba(126, 142, 255, 0.22);
        font-size: 0.8rem;
    }

    .phone-frame {
        width: min(100%, 420px);
        margin-left: auto;
        background: linear-gradient(180deg, rgba(20, 36, 51, 0.94), rgba(12, 20, 31, 0.96));
        border: 1px solid rgba(155, 195, 255, 0.2);
        border-radius: 2.1rem;
        padding: 1rem;
        box-shadow: 0 30px 60px rgba(4, 11, 21, 0.7);
    }

    .phone-top {
        height: 18px;
        width: 38%;
        background: rgba(9, 15, 22, 0.9);
        border-radius: 999px;
        margin: 0 auto 0.75rem;
    }

    .phone-shell {
        background: linear-gradient(180deg, #101c2b 0%, #0c1624 50%, #0b1420 100%);
        border-radius: 1.5rem;
        padding: 1rem 0.9rem 0.8rem;
        overflow: hidden;
    }

    .phone-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        color: #d9ebff;
    }

    .phone-title {
        font-weight: 700;
        font-size: 0.9rem;
    }

    .phone-badge {
        background: rgba(103, 216, 255, 0.12);
        border: 1px solid rgba(103, 216, 255, 0.2);
        border-radius: 999px;
        padding: 0.28rem 0.5rem;
        font-size: 0.66rem;
        color: #7ad7ff;
    }

    .user-tag {
        background: rgba(87, 135, 255, 0.12);
        color: #a8c8ff;
        padding: 0.32rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(87, 135, 255, 0.25);
        font-size: 0.68rem;
        margin-bottom: 0.8rem;
        display: inline-block;
    }

    .chat-window {
        display: flex;
        flex-direction: column;
        gap: 0.7rem;
        min-height: 300px;
        padding: 0.2rem 0;
    }

    .chat-message {
        max-width: 84%;
        padding: 0.72rem 0.8rem;
        border-radius: 1rem;
        line-height: 1.5;
        font-size: 0.76rem;
        letter-spacing: 0.01em;
    }

    .chat-message.user {
        background: linear-gradient(135deg, #7ad7ff 0%, #84bbff 100%);
        color: #061a2c;
        margin-left: auto;
        border-bottom-right-radius: 0.45rem;
    }

    .chat-message.assistant {
        background: rgba(148, 163, 184, 0.12);
        color: #e8f4ff;
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-bottom-left-radius: 0.45rem;
    }

    .quick-actions {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        margin-top: 0.9rem;
        padding-top: 0.5rem;
    }

    .quick-btn {
        flex: 1;
        border: 1px solid rgba(124, 128, 163, 0.26);
        background: rgba(17, 24, 39, 0.7);
        color: #dfeafd;
        border-radius: 0.9rem;
        padding: 0.7rem 0.5rem;
        font-size: 0.7rem;
        text-align: center;
    }

    .composer {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-top: 0.8rem;
        padding: 0.55rem 0.6rem 0.55rem 0.8rem;
        border-radius: 1rem;
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(128, 159, 211, 0.12);
    }

    .composer span {
        color: #9baec9;
        font-size: 0.76rem;
    }

    .composer .send {
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 0.8rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #76d8ff 0%, #6a7cff 100%);
        color: #061b30;
        font-weight: 900;
    }

    .workspace-card {
        margin-top: 2rem;
        background: rgba(12, 20, 31, 0.85);
        border: 1px solid rgba(133, 167, 215, 0.16);
        border-radius: 1.5rem;
        padding: 1.25rem;
        box-shadow: 0 20px 40px rgba(3, 8, 15, 0.24);
    }

    .workspace-grid {
        display: grid;
        grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.5fr);
        gap: 1.5rem;
    }

    .stack {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .metric-panel {
        background: rgba(8, 19, 28, 0.9);
        border: 1px solid rgba(153, 183, 240, 0.12);
        border-radius: 1rem;
        padding: 1rem;
    }

    .metric-panel h4 {
        margin-bottom: 0.5rem;
        color: #edf7ff;
    }

    .stButton > button {
        border-radius: 0.8rem !important;
        font-weight: 700 !important;
        border: 1px solid rgba(126, 192, 255, 0.3) !important;
        background: linear-gradient(135deg, rgba(103, 216, 255, 0.18), rgba(108, 123, 255, 0.18)) !important;
        color: #ebf6ff !important;
    }

    .stTextInput input, .stSelectbox div[role='button'], .stSlider div[data-baseweb='slider'], .stFileUploader > div {
        border-radius: 0.8rem !important;
        border: 1px solid rgba(132, 168, 212, 0.24) !important;
        background: rgba(11, 23, 35, 0.8) !important;
        color: #f0f8ff !important;
    }

    .stFileUploader label {
        color: #cfe6ff !important;
    }

    @media (max-width: 960px) {
        .hero {
            grid-template-columns: 1fr;
        }

        .phone-frame {
            margin: 0 auto;
        }

        .workspace-grid {
            grid-template-columns: 1fr;
        }

        .nav-links {
            display: none;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def load_uploaded_notes(file_bytes: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name)
    try:
        documents = load_notes(temp_path)
        for document in documents:
            document.metadata["source_file"] = Path(filename).name
        return documents
    finally:
        temp_path.unlink(missing_ok=True)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return create_embeddings()


st.markdown(
    """
    <div class="page-shell">
      <div class="topbar">
        <div class="brand-group">
          <span class="brand-mark">R</span>
          <span>RAG Assistant</span>
        </div>
        <div class="nav-links">
          <span>Home</span>
          <span>Features</span>
          <span>Docs</span>
          <span>Pricing</span>
        </div>
        <div class="nav-actions">
          <button class="nav-button">Sign in</button>
          <button class="nav-button">Book demo</button>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.15, 0.85])
with left_col:
    st.markdown('<div class="eyebrow">AI Knowledge Layer</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h1>RAG-powered <span class="gradient-text">Knowledge Assistant</span></h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-copy">
            Create a chatbot that retrieves information from documents and generates context-aware responses
            using RAG.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="cta-row">
            <a class="primary-btn" href="#workspace">Try the demo</a>
            <a class="secondary-btn" href="#workspace">View workflow</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-chips">
            <span class="feature-chip">Grounded answers</span>
            <span class="feature-chip">PDF / DOCX notes</span>
            <span class="feature-chip">Citations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        """
        <div class="phone-frame">
            <div class="phone-top"></div>
            <div class="phone-shell">
                <div class="phone-header">
                    <div class="phone-title">RAG-Powered</div>
                    <div class="phone-badge">Live</div>
                </div>
                <div class="user-tag">Recent Queries</div>
                <div class="chat-window">
                    <div class="chat-message assistant">Summarize the key decision criteria for this chapter.</div>
                    <div class="chat-message user">Here are the main factors and supporting references.</div>
                    <div class="chat-message assistant">I can cite the relevant notes and explain the evidence.</div>
                </div>
                <div class="quick-actions">
                    <div class="quick-btn">Not interested</div>
                    <div class="quick-btn">Interested</div>
                </div>
                <div class="composer">
                    <span>Ask a question</span>
                    <div style="margin-left: auto;" class="send">➤</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div id="workspace" class="workspace-card">', unsafe_allow_html=True)

with st.container():
    st.subheader("Knowledge workspace")
    left_setup, right_setup = st.columns([0.9, 1.5])
    with left_setup:
        st.markdown("<div class='stack'>", unsafe_allow_html=True)
        model_name = st.text_input("Ollama model", value="llama3.2:3b")
        chunk_choice = st.selectbox("Index chunk size", list(CHUNK_CONFIGS))
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
        notes_file = st.file_uploader("Upload notes (PDF or DOCX)", type=["pdf", "docx"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right_setup:
        if notes_file is None:
            st.info("Upload the Unit I-IV notes to build the two chunk indexes and ask grounded questions.")
            st.markdown(
                """
                <div class="metric-panel">
                    <h4>What the app does</h4>
                    <p>It loads course notes, splits them into the 500/50 and 1000/100 chunk configurations, builds a local FAISS index, and cites source passages alongside each answer.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            file_bytes = notes_file.getvalue()
            try:
                with st.spinner("Extracting text from your notes..."):
                    documents = load_uploaded_notes(file_bytes, notes_file.name)
                chunks_by_config = {
                    label: split_notes(documents, **settings)
                    for label, settings in CHUNK_CONFIGS.items()
                }
            except Exception as error:
                st.error(f"Could not read this document: {error}")
                chunks_by_config = {}

            if chunks_by_config:
                counts = pd.DataFrame(
                    [
                        {"Configuration": label, "Chunk count": len(chunks)}
                        for label, chunks in chunks_by_config.items()
                    ]
                )
                st.dataframe(counts, hide_index=True, width="stretch")
                st.caption(f"Loaded {len(documents)} document page(s) from {notes_file.name}.")
                ocr_page_count = sum(bool(document.metadata.get("ocr")) for document in documents)
                if ocr_page_count:
                    st.caption(f"OCR extracted text from {ocr_page_count} scanned page(s).")

                chosen_chunks = chunks_by_config[chunk_choice]
                content_digest = hashlib.sha256(file_bytes).hexdigest()
                note_key = f"{content_digest}:{chunk_choice}"
                if st.session_state.get("index_key") != note_key:
                    with st.spinner("Loading embeddings and building the FAISS index..."):
                        try:
                            embeddings = load_embedding_model()
                            st.session_state.vector_store = create_vector_store(chosen_chunks, embeddings)
                            st.session_state.index_key = note_key
                        except Exception as error:
                            st.error(f"Could not build the index: {error}")

                if st.session_state.get("index_key") == note_key:
                    st.subheader("Ask the notes")
                    question = st.text_input("Question", placeholder="Ask something covered in the notes")
                    if st.button("Get grounded answer", type="primary", disabled=not question.strip()):
                        try:
                            check_ollama_model(model_name)
                            llm = create_llm(model_name)
                            answer, source_documents = answer_with_rag(
                                question, st.session_state.vector_store, llm, k=top_k
                            )
                            st.markdown("**Answer**")
                            st.write(answer)
                            st.markdown("**Retrieved source chunks**")
                            st.code(format_sources(source_documents), language="text")
                        except Exception as error:
                            st.error(f"The local model request failed: {error}")

                    st.divider()
                    st.subheader("Evaluation")
                    st.write("Use the 15-row CSV template. Fill in questions and expected answers from your notes before running.")
                    evaluation_file = st.file_uploader("Evaluation CSV", type=["csv"], key="evaluation_csv")
                    if evaluation_file is not None:
                        try:
                            evaluation = pd.read_csv(evaluation_file, keep_default_na=False)
                            required = {"question", "expected_answer", "answerable"}
                            if not required.issubset(evaluation.columns):
                                st.error("CSV must include question, expected_answer, and answerable columns.")
                            else:
                                validation_errors, answerability = validate_evaluation_rows(
                                    evaluation.to_dict(orient="records")
                                )
                                if validation_errors:
                                    for validation_error in validation_errors:
                                        st.warning(validation_error)
                                if st.button(
                                    "Run baseline and RAG evaluation",
                                    disabled=bool(validation_errors),
                                ):
                                    check_ollama_model(model_name)
                                    llm = create_llm(model_name)
                                    results = []
                                    progress = st.progress(0)
                                    for index, row in evaluation.reset_index(drop=True).iterrows():
                                        prompt = str(row["question"])
                                        baseline = answer_without_rag(prompt, llm)
                                        rag_answer, sources = answer_with_rag(
                                            prompt, st.session_state.vector_store, llm, k=top_k
                                        )
                                        results.append(
                                            {
                                                "question": prompt,
                                                "expected_answer": str(row["expected_answer"]),
                                                "answerable": answerability[index],
                                                "baseline_answer": baseline,
                                                "rag_answer": rag_answer,
                                                "rag_sources": format_sources(sources),
                                                "baseline_outcome": "unscored",
                                                "rag_outcome": "unscored",
                                            }
                                        )
                                        progress.progress((index + 1) / len(evaluation))
                                    st.session_state.evaluation_results = pd.DataFrame(results)
                                if "evaluation_results" in st.session_state:
                                    st.caption("Score each response against your notes. Do not treat an automatic model judgment as ground truth.")
                                    outcomes = [
                                        "unscored",
                                        "correct answer",
                                        "hallucination",
                                        "correct refusal",
                                        "incorrect refusal",
                                        "incorrect answer",
                                    ]
                                    edited_results = st.data_editor(
                                        st.session_state.evaluation_results,
                                        column_config={
                                            "baseline_outcome": st.column_config.SelectboxColumn("Baseline outcome", options=outcomes),
                                            "rag_outcome": st.column_config.SelectboxColumn("RAG outcome", options=outcomes),
                                            "expected_answer": st.column_config.TextColumn("Expected answer", width="medium"),
                                            "rag_sources": st.column_config.TextColumn("RAG sources", width="large"),
                                        },
                                        disabled=["question", "expected_answer", "answerable", "baseline_answer", "rag_answer", "rag_sources"],
                                        hide_index=True,
                                        width="stretch",
                                    )
                                    st.session_state.evaluation_results = edited_results
                                    summary_rows = []
                                    for method, column in [("Plain LLM", "baseline_outcome"), ("RAG", "rag_outcome")]:
                                        counts_for_method = edited_results[column].value_counts()
                                        summary_rows.append({"Method": method, **{outcome: int(counts_for_method.get(outcome, 0)) for outcome in outcomes[1:]}})
                                    st.markdown("**Manually scored summary**")
                                    st.dataframe(pd.DataFrame(summary_rows).fillna(0), hide_index=True, width="stretch")
                                    st.download_button(
                                        "Download results CSV",
                                        edited_results.to_csv(index=False).encode("utf-8"),
                                        file_name="rag_evaluation_results.csv",
                                        mime="text/csv",
                                    )
                        except Exception as error:
                            st.error(f"Evaluation could not be completed: {error}")

st.markdown("</div>", unsafe_allow_html=True)
