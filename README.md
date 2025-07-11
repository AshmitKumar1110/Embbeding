readme_content = """
# 📚 Embedding Search Pipeline

This project showcases a complete pipeline for **creating, storing, and searching code embeddings** using pretrained models like **CodeBERT**, with an optional **Streamlit interface**.

---

## 🧠 Overview

- **Embed Code Snippets**: Use `microsoft/codebert-base` to generate vector embeddings from a codebase.
- **Store in FAISS**: Build a FAISS index for efficient similarity search.
- **Search Interface**:
  - CLI-based querying via natural language or code.
  - (Optional) Streamlit web UI for interactive code search and explanation.
- **Explain Code**: Incorporates **CodeT5** to generate human-readable summaries of code snippets.

---

## 🔧 Requirements

```bash
pip install -r requirements.txt
streamlit
torch
transformers
faiss-cpu
sentencepiece
