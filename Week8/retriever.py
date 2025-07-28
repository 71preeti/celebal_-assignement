# retriever.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def prepare_corpus(csv_path="data/loan_data.csv"):
    df = pd.read_csv(csv_path)
    corpus = []
    for i, row in df.iterrows():
        row_text = f"Applicant ID: {i}, Gender: {row['Gender']}, Married: {row['Married']}, Education: {row['Education']}, Income: {row['ApplicantIncome']}, Loan Status: {row['Loan_Status']}"
        corpus.append(row_text)
    return corpus

def build_faiss_index(corpus, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save index and corpus
    faiss.write_index(index, "embeddings/faiss_index.bin")
    with open("embeddings/corpus.txt", "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")

def load_index():
    index = faiss.read_index("embeddings/faiss_index.bin")
    with open("embeddings/corpus.txt", "r", encoding="utf-8") as f:
        corpus = f.readlines()
    return index, corpus

def retrieve_relevant_docs(query, top_k=3):
    index, corpus = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query])
    _, I = index.search(np.array(query_vec), top_k)
    results = [corpus[i].strip() for i in I[0]]
    return results
