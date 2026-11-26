# Lead Generation & Outreach Hybrid RAG (Streamlit)

AI assistant built with **ChromaDB** (semantic vector search), **BM25** (lexical keyword search), **Reciprocal Rank Fusion (RRF)** hybrid retrieval, and **Groq (`openai/gpt-oss-120b`)**, deployed on **Streamlit**.

## Quick Start on Streamlit Community Cloud

1. Extract this zip archive and upload the contents to your GitHub repository.
2. In Streamlit Community Cloud, set `app.py` as your **Main file path**.
3. Add `GROQ_API_KEY = "your_groq_api_key_here"` under **App settings -> Secrets** (or paste it in the sidebar).
4. Upload `Lead Generation & Outreach SOP + CV.pdf` in the sidebar and click **Index Uploaded PDF**.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
