
import streamlit as st
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
import json

st.set_page_config(page_title="Code Search & Explanation", layout="centered")

# Title
st.title("🔍 Code Search & Explanation with CodeBERT + CodeT5")
st.markdown("Search your codebase using natural language or code snippets, and get explanations with CodeT5.")

# Load CodeBERT model and tokenizer
@st.cache_resource
def load_codebert():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.eval()
    return tokenizer, model

# Load CodeT5 summarizer
@st.cache_resource
def load_codet5():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
    model.eval()
    return tokenizer, model

# Load FAISS index and code snippets
# Assuming 'codebert_embeddings.index' and 'code_snippets.json' are already created
try:
    index = faiss.read_index("codebert_embeddings.index")
    with open("code_snippets.json") as f:
        code_snippets = json.load(f)
except FileNotFoundError:
    st.error("Required data files (codebert_embeddings.index, code_snippets.json) not found. Please run the cell to create dummy files.")
    st.stop()


# Embedding function (CodeBERT)
def get_embedding(text):
    inputs = cb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = cb_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Search code
def search(query, top_k=3):
    # Ensure models and index are loaded before use
    global cb_tokenizer, cb_model, index, code_snippets
    try:
        cb_tokenizer, cb_model = load_codebert()
    except Exception as e:
        st.error(f"Error loading CodeBERT model: {e}")
        st.stop()

    try:
        query_emb = get_embedding(query).reshape(1, -1)
        distances, indices = index.search(query_emb, top_k)
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            results.append({
                "rank": rank,
                "code": code_snippets[idx],
                "distance": float(dist)
            })
        return results
    except Exception as e:
        st.error(f"Error during search: {e}")
        st.stop()


# Summarize code
def summarize_code(code):
    # Ensure CodeT5 model is loaded before use
    global t5_tokenizer, t5_model
    try:
        t5_tokenizer, t5_model = load_codet5()
    except Exception as e:
        st.error(f"Error loading CodeT5 model: {e}")
        st.stop()

    try:
        input_ids = t5_tokenizer("summarize: " + code, return_tensors="pt", truncation=True).input_ids
        with torch.no_grad():
            output_ids = t5_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        st.stop()


# UI: Query input
query_type = st.selectbox("Choose query type:", ["Natural Language", "Code Snippet"])
query = st.text_area("Enter your query:", height=120)

# Perform search
if query:
    st.info(f"Searching using {query_type.lower()}...")
    results = search(query, top_k=3)
    st.subheader("🔎 Top Matching Code Snippets:")
    for result in results:
        st.code(result["code"], language="python")
        st.caption(f"Rank #{result['rank']} • Distance: {result['distance']:.4f}")

    # Optional explanation
    if query_type == "Code Snippet":
        if st.button("🧠 Explain This Code with CodeT5"):
            explanation = summarize_code(query)
            st.subheader("🧠 Code Explanation:")
            st.success(explanation)

