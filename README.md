# RealEstateTool

Small RAG demo that scrapes URLs, indexes content into a Chroma vector store and answers questions using a GROQ LLM.

# Problem Statement

You are a Real Estate Analyst, In a Real Estate Investment Company. Your Job is to do research about the Real Estate news, market, study the financial data and make the report that will help the portfolio manager decide if they should invest in some property or not. So you have some trusted websites like CNBC, Bloomgerg, Ecom=nomic Times etc., to get the news for research, the mannual reading of all these articles are time consuming

# Solution

To solve the above problem we are creating th ereal estate tool that takes the news article website URLs, Process it and Store it in a Vector Database (ChromaDB) and use the LLM to Query/ ask Question about the article and get the relevant insights for the research report and answer that question based on those articles only. We are not using the LLMs knowlegde to answer the question. We will be answering the question along with the source link.

# ISSUE

## What are we solving here ?, as we can paste the news article in Chat GPT and get the insights!!!

- Copy Pasting the large and many articles are tedious and time consuming
- We need an aggregate database, some process kicks off at 5 AM or 6 PM goes through all the news article and aggregate and store it in Centralized database. We can have a Chatbot like chatgpt and ask the question
- Context window limit, if we copy paste multiple news articles it will not support context window after certain point
- LLM api and inference cost, if we paste the multiple news articles lot of Tokens will be wasted, as the most of the LLM providers charge you based on the number of tokes used, So this will reduce the number of tokens used while doing this report

Paste your GROQ API KEY in .env before live demo
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://realestatetool-using-groq.streamlit.app/)

**Features**

- **URL scraping**: loads and extracts page text from one or more URLs (uses `UnstructuredURLLoader` and a requests/HTML fallback when needed).
- **Text splitting / chunking**: splits long documents into chunks using a recursive character text splitter for better retrieval granularity.
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
