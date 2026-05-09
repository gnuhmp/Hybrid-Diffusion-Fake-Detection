"""
Text Preprocessing and Embedding Generation Pipeline.
Responsible for converting raw text into lightweight SBERT embeddings to save resources.
"""

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

def generate_text_embeddings(raw_parquet_path, output_npz_path, encoder_model='all-MiniLM-L6-v2'):
    """
    Reads raw textual data from parquet, encodes it using a SentenceTransformer, 
    and saves it as a sparse matrix (.npz) compatible with UPFD baselines.
    
    Args:
        raw_parquet_path (str): Path to the raw .parquet file (e.g., 'data/raw/politifact/raw_text/part-00000.parquet')
        output_npz_path (str): Path to save the output .npz file
        encoder_model (str): The HuggingFace model name for SentenceTransformers
    """
    print(f"[*] Loading SentenceTransformer model: {encoder_model}...")
    model = SentenceTransformer(encoder_model)
    
    print(f"[*] Reading raw text data from: {raw_parquet_path}...")
    try:
        df = pd.read_parquet(raw_parquet_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file. Ensure the file exists. Error: {e}")
        
    # Assume the text column is named 'text' or 'content'. Adjust if your parquet differs.
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    texts = df[text_col].fillna("").astype(str).tolist()
    
    print(f"[*] Encoding {len(texts)} documents. This may take a few minutes...")
    # SBERT encode process
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    print("[*] Converting embeddings to Sparse Matrix (CSR)...")
    sparse_embeddings = sp.csr_matrix(embeddings)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    
    print(f"[*] Saving feature matrix to: {output_npz_path}")
    sp.save_npz(output_npz_path, sparse_embeddings)
    
    print(f"Successfully generated embeddings! Shape: {sparse_embeddings.shape}")
    return sparse_embeddings