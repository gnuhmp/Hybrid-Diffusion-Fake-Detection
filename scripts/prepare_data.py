"""
Entry point script to prepare raw text data into lightweight embeddings.

Usage Example:
    python scripts/prepare_data.py --dataset politifact --encoder sbert
"""

import os
import sys
import argparse

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import generate_text_embeddings

def main():
    parser = argparse.ArgumentParser(description='Prepare Text Embeddings for GNN models')
    
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['politifact', 'gossipcop'],
                        help='Dataset to process')
    
    parser.add_argument('--encoder', type=str, default='sbert',
                        choices=['sbert', 'bert'],
                        help='Which encoder to use (sbert = 384d, bert = 768d)')
                        
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Root directory for raw data')
                        
    parser.add_argument('--interim_dir', type=str, default='data/interim',
                        help='Root directory for output embeddings')
                        
    args = parser.parse_args()
    
    print("="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    
    # 1. Define paths
    # Assuming your text data is stored in a file named 'news.parquet' inside the raw_text folder
    raw_parquet_path = os.path.join(args.raw_dir, args.dataset, 'raw_text', 'part-00000.parquet')
    
    # Define output path based on structure: data/interim/embeddings_{encoder}/{dataset}/new_{encoder}_feature.npz
    output_dir = os.path.join(args.interim_dir, f'embeddings_{args.encoder}', args.dataset)
    output_npz_path = os.path.join(output_dir, f'new_{args.encoder}_feature.npz')
    
    # 2. Select appropriate model
    if args.encoder == 'sbert':
        model_name = 'all-MiniLM-L6-v2'  # Lightweight 384D (Highly recommended)
    else:
        model_name = 'bert-base-uncased' # Heavy 768D (For baseline reproduction only)
        
    print(f"Dataset:      {args.dataset.upper()}")
    print(f"Encoder mode: {args.encoder.upper()} -> {model_name}")
    print(f"Input file:   {raw_parquet_path}")
    print(f"Output file:  {output_npz_path}")
    print("-" * 80)
    
    # 3. Execute Processing
    if not os.path.exists(raw_parquet_path):
        print(f"ERROR: Raw parquet file not found at {raw_parquet_path}")
        print("Please ensure your raw data is placed correctly according to the project structure.")
        return
        
    generate_text_embeddings(
        raw_parquet_path=raw_parquet_path,
        output_npz_path=output_npz_path,
        encoder_model=model_name
    )

if __name__ == "__main__":
    main()