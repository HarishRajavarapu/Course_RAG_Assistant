from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CONFIGS = {
    "500 / 50": {"chunk_size": 500, "chunk_overlap": 50},
    "1000 / 100": {"chunk_size": 1000, "chunk_overlap": 100},
}
GROUNDING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the supplied notes. If the notes do not "
            "contain the answer, reply exactly: not in the notes. Do not use outside "
            "knowledge. Cite supporting chunks using their bracketed source labels.",
        ),
        ("human", "Question: {question}\n\nRetrieved notes:\n{context}"),
    ]
)


def load_notes(file_path: str | Path) -> list[Document]:
    """Load one PDF or DOCX notes file with a LangChain document loader."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
        empty_pages = [
            index for index, document in enumerate(documents) if not document.page_content.strip()
        ]
        if empty_pages:
            try:
                import numpy as np
                import pymupdf
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as error:
                raise RuntimeError(
                    "This PDF appears to contain scanned pages. Install the OCR "
                    "dependencies with `pip install -r requirements.txt` and retry."
                ) from error

            ocr = RapidOCR()
            with pymupdf.open(path) as pdf:
                for index in empty_pages:
                    page = pdf[index]
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                        pixmap.height, pixmap.width, pixmap.n
                    )
                    result, _ = ocr(image[:, :, ::-1])
                    text = "\n".join(line[1] for line in (result or []))
                    documents[index].page_content = text
                    documents[index].metadata["ocr"] = True
    elif suffix == ".docx":
        documents = Docx2txtLoader(str(path)).load()
    else:
        raise ValueError("Upload a PDF or DOCX file.")

    for document in documents:
        document.metadata["source_file"] = path.name
    if not any(document.page_content.strip() for document in documents):
        raise ValueError(
            "No text could be extracted. For scanned PDFs, confirm the pages are "
            "readable and try a higher-quality scan."
        )
    return documents


def split_notes(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def create_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(
    chunks: list[Document], embeddings: HuggingFaceEmbeddings
) -> FAISS:
    if not chunks:
        raise ValueError("No text was extracted from the uploaded notes.")
    return FAISS.from_documents(chunks, embeddings)


def create_llm(model_name: str = "llama3.2:3b") -> ChatOllama:
    return ChatOllama(model=model_name, temperature=0)


def check_ollama_model(model_name: str, base_url: str = "http://localhost:11434") -> None:
    """Raise a setup-focused error when Ollama or the requested model is unavailable."""
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2) as response:
            available_models = json.loads(response.read()).get("models", [])
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. Install and start "
            "Ollama, then retry. On Windows, open Ollama from the Start menu; if it "
            "is not installed, install it from https://ollama.com/download. "
            "The API should listen on port 11434."
        ) from error

    available_names = {model.get("name", "") for model in available_models}
    if model_name not in available_names:
        raise RuntimeError(
            f"The Ollama model '{model_name}' is not downloaded. Run "
            f"`ollama pull {model_name}` in PowerShell, then retry."
        )


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def answer_with_rag(
    question: str, vector_store: FAISS, llm: ChatOllama, k: int = 4
) -> tuple[str, list[Document]]:
    documents = vector_store.similarity_search(question, k=k)
    context_parts = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_file", "notes")
        page = document.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        context_parts.append(
            f"[Source {index}: {source}{page_text}]\n{document.page_content}"
        )
    response = (GROUNDING_PROMPT | llm).invoke(
        {"question": question, "context": "\n\n".join(context_parts)}
    )
    return _message_text(response), documents


def answer_without_rag(question: str, llm: ChatOllama) -> str:
    response = llm.invoke(question)
    return _message_text(response)


def format_sources(documents: list[Document]) -> str:
    if not documents:
        return "No source chunks were retrieved."
    parts = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_file", "notes")
        page = document.metadata.get("page")
        location = f"page {page + 1}" if isinstance(page, int) else "document"
        parts.append(
            f"[Source {index}: {source}, {location}]\n{document.page_content}"
        )
    return "\n\n".join(parts)
