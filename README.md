# RealEstateTool

Small RAG demo that scrapes URLs, indexes content into a Chroma vector store and answers questions using a GROQ LLM.

Paste your GROQ API KEY in .env before live demo
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://realestatetool-using-groq.streamlit.app/)

**Features**

- **URL scraping**: loads and extracts page text from one or more URLs (uses `UnstructuredURLLoader` and a requests/HTML fallback when needed).
- **Text splitting**: splits long documents into chunks using a recursive character text splitter for better retrieval granularity.
- **Embeddings**: creates vector embeddings using Hugging Face sentence-transformer models (`HuggingFaceEmbeddings`).
- **Vector store**: stores embeddings in a persistent Chroma collection for fast similarity search and retrieval.
- **Retrieval-Augmented Generation (RAG)**: answers user questions by retrieving relevant chunks and using a GROQ LLM (`ChatGroq`) to generate answers.
- **Streamlit UI**: simple web UI (`app.py`) to provide URLs, run ingestion, and ask questions interactively.
- **Progress streaming**: ingestion yields progress messages so the UI can display status while processing.
- **Resilient fetching & fallbacks**: handles CDN/Access-Denied cases by providing alternative fetch strategies and sensible fallbacks so indexing still works when possible.
- **Version-tolerant LangChain usage**: the code includes tolerant imports and invocation patterns to work across LangChain package variants and versions.
- **Config via `.env`**: easy configuration of secrets (GROQ API key) via a `.env` file in the project root.


Setup

- Create a Python virtual environment and activate it

- Install dependencies:

```zsh
pip install -r requirements.txt
```

Environment variables

- Create a `.env` file in the project root (same directory as this README).
- Add your GROQ API key to `.env` using the key name below (replace the value):

```text
GROQ_API_KEY=your_groq_api_key_here
```

Running

```zsh
source .venv/bin/activate
streamlit run app.py
```

Notes

- If some imports fail due to LangChain package layout differences, install the packages listed in `requirements.txt` and re-run.
- If a target site blocks scraping (Access Denied), consider providing alternate URLs or using a different fetch strategy.

Files

- `rag.py` — main script that loads URLs, builds the vector store and answers queries.
- `requirements.txt` — Python dependencies for the project.

Link:- https://realestatetool-using-groq.streamlit.app

License

This project is released under the MIT License. See `LICENSE` for details.
