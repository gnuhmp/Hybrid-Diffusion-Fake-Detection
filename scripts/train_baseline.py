"""
Training script for UPFD Baseline models (BiGCN, GCNFN, GAT, etc.).
Forces baselines to use the strict dataset splits to prevent Data Leakage 
and allows evaluating them using lightweight SBERT embeddings.

Usage Example:
    python scripts/train_baseline.py --dataset politifact --model bigcn --encoder sbert
"""

import os
import sys
import argparse
import json
import torch
from torch_geometric.loader import DataLoader

# Add parent to path for imports FIRST before any src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_builder import load_raw_parquet_data, create_pyg_graphs_from_raw, split_datasets_strict
from src.training.trainer import train_full_pipeline, build_optimizer
from src.training.losses import FocalLoss

# IMPORTANT: You need to import the baseline models you copied into src/models/baselines
try:
    from src.models.baselines.bigcn import BiGCN
    from src.models.baselines.gcnfn import GCNFN
    from src.models.baselines.gnn import Model as GNN
    from src.models.baselines.gnncl import GNNCLNet
except ImportError:
    print("Warning: Baseline models not fully imported. Ensure src/models/baselines contains the UPFD files.")

def main():
    parser = argparse.ArgumentParser(description='Train UPFD Baseline Models')
    
    # Dataset and baseline selection
    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--encoder', type=str, default='sbert', choices=['sbert', 'bert'])
    parser.add_argument('--model', type=str, required=True, choices=['bigcn', 'gcnfn', 'gnn', 'gnncl', 'gat', 'sage', 'gcn'], 
                        help='Name of the baseline model to run')
    
    # Directory paths
    parser.add_argument('--raw_dir', type=str, default='data/raw')
    parser.add_argument('--interim_dir', type=str, default='data/interim')
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    # Training hyperparameter
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30)
    
    args = parser.parse_args()

    # Route to appropriate baseline output folder
    # outputs/baselines_sbert/bigcn/politifact
    baseline_out_dir = os.path.normpath(os.path.join(
        args.output_dir, 
        f'baselines_{args.encoder}', 
        args.model, 
        args.dataset
    ))
    os.makedirs(baseline_out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print(f"UPFD BASELINE TRAINING - MODEL: {args.model.upper()}")
    print("="*80)
    
    # 1. Load Data (Using our strict leakage-free pipeline)
    print("Loading graph structures and text embeddings...")
    data_dict = load_raw_parquet_data(
        dataset_name=args.dataset, raw_dir=args.raw_dir,
        interim_dir=args.interim_dir, encoder=args.encoder
    )
    
    graphs = create_pyg_graphs_from_raw(data_dict)
    text_dim = data_dict['node_features'].shape[1] 
    
    # 2. Strict Dataset Splitting
    print("Splitting datasets strictly (Preventing Data Leakage)...")
    train_graphs, val_graphs, test_graphs = split_datasets_strict(
        graphs, dataset_name=args.dataset, raw_dir=args.raw_dir
    )
    
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # 3. Initialize chosen Baseline Model
    print(f"Initializing {args.model.upper()}...")
    if args.model == 'bigcn':
        model = BiGCN(num_features=text_dim, hidden_dim=args.hidden_dim, num_classes=2)
    elif args.model == 'gcnfn':
        model = GCNFN(num_features=text_dim, hidden_dim=args.hidden_dim, num_classes=2)
    elif args.model == 'gnn':
        model = GNN(in_channels=text_dim, num_classes=2, nhid=args.hidden_dim)
    elif args.model == 'gnncl':
        model = GNNCLNet(in_channels=text_dim, num_classes=2, nhid=args.hidden_dim)
    else:
        raise NotImplementedError(f"Model initialization for {args.model} is not yet hooked up.")
        
    model = model.to(device)
    
    # 4. Optimizer and Loss
    # Baselines usually use standard CrossEntropy, but we can stick to FocalLoss or CE
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=1e-4)
    
    # 5. Train using the unified pipeline
    print(f"Training for {args.epochs} epochs...")
    model, test_metrics = train_full_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=baseline_out_dir  
    )
    
    # 6. Save results
    metrics_path = os.path.join(baseline_out_dir, f'metrics_{args.model}_final.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
        
    print(f"\n Baseline {args.model.upper()} training complete! Results saved to {metrics_path}")
    print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"Test F1:       {test_metrics.get('f1', 0):.4f}")

if __name__ == '__main__':
    main()