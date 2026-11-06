# Workspace Instructions

- This project is a Python Streamlit RAG assistant for course notes.
- Keep the LangChain pipeline in `rag_assistant.py` and the UI in `app.py`.
- Preserve both required chunk configurations: 500/50 and 1000/100 characters.
- Do not fabricate course-specific chunk counts, evaluation outcomes, or answers. Measure these using the uploaded notes.
- Keep citations visible and ground RAG responses in retrieved note chunks.
- Validate Python edits with `python -m py_compile app.py rag_assistant.py` and the Streamlit `AppTest` smoke test.