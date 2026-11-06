# Course Notes RAG Assistant

A Streamlit application for the LangChain RAG assignment. It loads course notes from PDF or DOCX, compares the required 500/50 and 1000/100 character chunking configurations, builds a FAISS index with `sentence-transformers/all-MiniLM-L6-v2`, and answers questions with a local Ollama model and visible source chunks.

> Evaluation scores and chunk counts depend on your actual Unit I-IV notes. They are not filled with invented results.

## Requirements

- Windows, macOS, or Linux
- Python 3.14 (validated in the project-local environment)
- Ollama installed locally with a model pulled, for example `llama3.2:3b`
- Internet access for the initial Python package and embedding-model downloads
- Scanned PDF OCR can require initial model downloads by RapidOCR

## Setup on Windows

From this directory, create and activate a virtual environment, then install packages:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2:3b
streamlit run app.py
```

Ollama must be running and listening on `http://localhost:11434` while using the app. On Windows, the Ollama desktop app normally starts the local service. If it is installed but the API is unavailable, open Ollama from the Start menu. If the model is missing, run `ollama pull llama3.2:3b` in PowerShell.

If PowerShell prevents activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. You may use another Ollama model by changing the model field in the app, provided it is already pulled locally.

## Use the app

1. Upload the Unit I-IV notes as one PDF or DOCX, up to 50 MB. The local app binds to `127.0.0.1` so uploaded course notes are not exposed to other devices on your network. The app reports chunk counts for both assignment configurations. Extracted note text is cached for one hour to avoid repeating OCR on reruns.
2. Select the chunk configuration to use for retrieval. The first index build downloads the embedding model and may take a few minutes.
3. Ask a question. The result includes the retrieved source chunks and PDF page metadata when available.
4. Fill in `evaluation_template.csv` with exactly 10 answerable and 5 unanswerable questions, their expected answers for answerable rows, and `true`/`false` answerability values. The evaluation action remains disabled until these requirements are met.
5. Upload that CSV and run the baseline and RAG evaluation. Review each response against the notes and assign an outcome: correct answer, hallucination, correct refusal, incorrect refusal, or incorrect answer. Download the scored table.
6. Use `report_template.md` to write up the measured results and the approximately 150-word chunk-size analysis.

The plain LLM baseline is intentionally given only the question. The RAG prompt is instructed to answer only from retrieved notes and to say `not in the notes` when evidence is absent. Human scoring is necessary: the app does not claim model-generated scores are ground truth.

## Project files

- `app.py`: Streamlit interface and evaluation workflow.
- `rag_assistant.py`: document loaders, chunking, embeddings, FAISS, Ollama, prompt, and citations.
- `evaluation_template.csv`: 15-row starter dataset with 10 answerable and 5 unanswerable slots.
- `report_template.md`: theory, measured result tables, analysis, and demo checklist.
- `tests/test_project.py`: unit tests for evaluation validation, chunk metadata, and citations.

## Notes and limitations

- Scanned/image-only PDFs are OCR-processed page by page with RapidOCR when standard PDF text extraction returns no text. OCR quality depends on scan resolution and clarity.
- The prompt encourages grounded answers but cannot guarantee that a model never hallucinates. Verify answers and citations against the original notes.
- The application currently takes one PDF or DOCX at a time. Combine notes into one document if needed.
- The FAISS index is built in memory for the current app session and is not persisted.

## Deliverables

Commit the code and completed README/report to your GitHub repository. Export the evaluation results CSV, render the completed report as PDF, and record a roughly two-minute walkthrough of the running app.