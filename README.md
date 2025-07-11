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
```

**requirements.txt** includes:
```
streamlit
torch
transformers
faiss-cpu
sentencepiece
```

---

## ⚙️ Usage

### 1. Generate Embeddings (if not provided)

Update or run a Colab/`*.py` script to:

- Load `codebert-base`
- Tokenize and encode your code snippets
- Save:
  - `codebert_embeddings.index` — FAISS index
  - `code_snippets.json` — list of original code snippets

### 2. CLI-Based Search

A simple Python function enables you to:

```python
results = search("def add")
for item in results:
    print(item["rank"], item["code"], item["distance"])
```

This returns the top-k code snippets matching your query.

### 3. Streamlit App (Optional)

If `app.py` is present, run:

```bash
streamlit run app.py
```

Features:

- Query by **natural language** or **code snippet**
- Display **top-k results** with similarity distances
- (For `Code Snippet` queries) Click “Explain This Code” to get a summary from **CodeT5**

---

## 📂 Project Structure

```
├── app.py                     # Streamlit interface
├── code_snippets.json         # Source code dataset
├── codebert_embeddings.index  # FAISS index of embeddings
├── requirements.txt
└── README.md
```

---

## 🚀 Optional Enhancements

- Add **other embedder models** (e.g. CodeT5, Code2Vec)
- Deploy using **Streamlit Cloud** or **Hugging Face Spaces**
- Integrate into frameworks like **LangChain** or **Haystack**

---

## 🤝 Contributions

Feel free to open issues, fork, or submit pull requests! This repository is a solid foundation to build powerful **code search**, **summarization**, and **developer tooling** features.