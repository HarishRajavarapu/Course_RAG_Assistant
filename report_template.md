# RAG Assistant Evaluation Report

## Project

- Course / subject:
- Student pair:
- Notes file(s):
- Ollama model and version:
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS

## Part A: Theory

### LangChain building blocks

- **Document loaders** read source formats such as PDF and DOCX and return documents with text and metadata.
- **Text splitters** divide long documents into smaller, overlapping chunks that fit model context windows and can be retrieved more precisely.
- **Embedding models** map text into numerical vectors so that semantically related questions and passages are near one another in vector space.
- **Vector stores** persist vectors and their source documents and support similarity search. This project uses FAISS.
- **Retrievers** use a query to select the most relevant stored chunks for a question.
- **Prompt templates** combine a stable instruction with changing inputs such as the question and retrieved context.
- **Chains** connect retrieval, prompt formatting, and model generation into one callable workflow.

### Why RAG can reduce hallucination

RAG supplies relevant source passages at answer time and instructs the model to ground its response in those passages. The answer can cite the retrieved chunks, and the prompt can require a refusal when the evidence is absent. This can reduce unsupported answers compared with generation from model memory alone, but does not guarantee correctness. RAG can fail when relevant text was not extracted, chunk boundaries separate important context, retrieval selects irrelevant chunks, or the model ignores or misinterprets the evidence.

## Part B: Build

### Corpus and chunking

| Configuration | Chunk size / overlap (characters) | Chunk count |
|---|---:|---:|
| A | 500 / 50 | _Fill after uploading the notes_ |
| B | 1000 / 100 | _Fill after uploading the notes_ |

### Implementation

Describe the loader, embedding model, FAISS index, Ollama model, grounding prompt, UI, and source citation behavior. Add a screenshot of the running app and note any extraction issues.

## Part C: Evaluation

Fill in 15 course-specific questions in `evaluation_template.csv`: 10 answerable from the notes and 5 not answerable. Run both methods in the app and manually score each response against the notes.

| Method | Correct answers | Hallucinations | Correct refusals | Incorrect answers/refusals |
|---|---:|---:|---:|---:|
| Plain LLM | _pending_ | _pending_ | _pending_ | _pending_ |
| RAG | _pending_ | _pending_ | _pending_ | _pending_ |

### Chunk-size analysis (~150 words)

_Complete this only after running both chunk configurations. Compare the actual answer and refusal outcomes, source relevance, and citation quality. Explain whether smaller chunks improved retrieval precision or whether larger chunks preserved context better. Include the measured counts and results; do not infer a winner from chunk counts alone._

## Reproducibility

- Python version:
- OS:
- Commands used:
- Known limitations:

## Demo recording checklist

Show note upload, both chunk counts, a question answered with cited chunks, a question that should be refused, and the evaluation summary. Keep the screen recording to about two minutes.