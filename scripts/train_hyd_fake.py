"""
Training script for HyDFakeModel with exogenous and endogenous context encoding.

This script implements the advanced fake news detection architecture combining:
1. Exogenous Context: User engagement from graph structure
2. Endogenous Preferences: Learned text representations  
3. Multi-Modal Fusion: Attention-based combination
4. Deep Classification: Enhanced classifier head

Run from project root:
    python scripts/train_hyd_fake.py --dataset politifact --encoder sbert

Expected improvement over v1:
    - F1 Score: 85% → 90%+
    - Accuracy: 84% → 88%+
    - Better interpretability via attention weights
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hyd_fake import HyDFakeModel
from src.data.dataset_builder import load_raw_parquet_data, create_pyg_graphs_from_raw, split_datasets_strict
from src.data.augment_prune import augment_dataset, prune_dataset
from src.training.trainer import train_full_pipeline, build_optimizer, create_cosine_scheduler_with_warmup
from src.training.losses import FocalLoss


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train HyDFakeModel with exogenous/endogenous context')
    
    # Dataset and feature selection
    parser.add_argument('--dataset', type=str, default='politifact', 
                        choices=['politifact', 'gossipcop'],
                        help='Dataset name to train on (politifact or gossipcop)')
    parser.add_argument('--encoder', type=str, default='sbert',
                        choices=['sbert', 'bert'],
                        help='Text encoder used (determines which interim folder to load features from)')
    
    # Directory paths
    parser.add_argument('--raw_dir', type=str, default='data/raw', 
                        help='Path to raw data folder')
    parser.add_argument('--interim_dir', type=str, default='data/interim', 
                        help='Path to interim data folder containing embeddings')
    parser.add_argument('--output_dir', type=str, default='outputs/hyd_fake', 
                        help='Output directory for models and results (will auto-create dataset subfolders)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=72, 
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                        help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, 
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Model & Graph specific arguments
    parser.add_argument('--no-augment', action='store_true', default=False,
                        help='Disable feature augmentation (11D features)')
    parser.add_argument('--prune', action='store_true', default=False, 
                        help='Prune graphs (remove deep nodes/noisy edges)')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers for data loader')
    parser.add_argument('--num_gat_layers', type=int, default=4,
                        help='Number of GAT layers in exogenous encoder')
    parser.add_argument('--edge_prune_threshold', type=float, default=0.05,
                        help='Attention weight threshold for edge pruning (0.0-1.0)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Ratio of warmup epochs to total epochs (0.0-1.0)')

    args = parser.parse_args()

    raw_path = os.path.join(args.raw_dir, args.dataset, 'raw_text')
    
    interim_path = os.path.join(args.interim_dir, f'embeddings_{args.encoder}', args.dataset)
    
    out_dir = os.path.join(args.output_dir, f'hyd_fake', args.dataset)
    
    # Augmentation is enabled by default unless --no-augment is passed
    args.augment = not args.no_augment
    
    # Setup dataset-specific output directory to prevent overwriting
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("HYD-FAKE MODEL - EXOGENOUS/ENDOGENOUS CONTEXT ENCODING")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()} | Encoder: {args.encoder.upper()}")
    print(f"Raw data path: {args.raw_dir}/{args.dataset}")
    print(f"Embeddings path: {args.interim_dir}/embeddings_{args.encoder}/{args.dataset}")
    print(f"Output dir: {dataset_output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"GAT layers: {args.num_gat_layers}")
    print(f"Edge prune threshold: {args.edge_prune_threshold}")
    print(f"Warmup ratio: {args.warmup_ratio}")
    print(f"Augment: {args.augment}")
    print(f"Prune: {args.prune}")
    print("="*80 + "\n")
    
    # ============= LOAD DATA =============
    print("Loading raw data and embeddings...")
    data_dict = load_raw_parquet_data(
        dataset_name=args.dataset,
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        encoder=args.encoder
    )
    
    # ============= CREATE GRAPHS =============
    print("\nCreating PyG graphs...")
    graphs = create_pyg_graphs_from_raw(data_dict)
    print(f"  Created {len(graphs)} graphs")
    
    # Dynamically extract text feature dimensions
    text_dim = data_dict['node_features'].shape[1]
    
    # ============= AUGMENT FEATURES =============
    if args.augment:
        print(f"Augmenting features (3D → 11D)...")
        graphs = augment_dataset(graphs)
        
    # Dynamically extract graph feature dimensions (after potential augmentation)
    graph_in_dim = graphs[0].x.shape[1]
    
    # ============= PRUNE GRAPHS =============
    if args.prune:
        print(f"Pruning graphs...")
        graphs = prune_dataset(graphs)
    
    # ============= SPLIT DATA =============
    print("\nSplitting datasets strictly (Preventing Data Leakage)...")
    train_graphs, val_graphs, test_graphs = split_datasets_strict(
        graphs, dataset_name=args.dataset, raw_dir=args.raw_dir
    )
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val:   {len(val_graphs)} graphs")
    print(f"  Test:  {len(test_graphs)} graphs")
    
    # ============= CREATE DATA LOADERS =============
    print(f"\nCreating data loaders (batch_size={args.batch_size})...")
    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_graphs, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_graphs, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    
    # ============= BUILD MODEL =============
    print("\nBuilding enhanced model...")
    model = HyDFakeModel(
        text_frozen_dim=text_dim,
        graph_in_dim=graph_in_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.3,
        mode='graph_text',
        num_gat_layers=args.num_gat_layers,
        edge_prune_threshold=args.edge_prune_threshold,
        text_encoder_name=args.encoder,
    )
    
    model = model.to(device)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ============= BUILD LOSS & OPTIMIZER =============
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=1e-4)
    
    print(f"  Loss: FocalLoss (α=0.25, γ=2.0)")
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=1e-4)")
    
    # ============= SETUP LEARNING RATE SCHEDULER =============
    # Create scheduler: linear warmup + cosine annealing
    warmup_epochs = max(1, int(args.epochs * args.warmup_ratio))
    steps_per_epoch = len(train_loader)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    print(f"  Scheduler: Cosine Annealing with Linear Warmup")
    print(f"    - Warmup: {warmup_epochs} epochs ({warmup_epochs * steps_per_epoch} steps)")
    print(f"    - Cosine: {args.epochs - warmup_epochs} epochs ({(args.epochs - warmup_epochs) * steps_per_epoch} steps)")
    
    # ============= TRAIN =============
    print(f"\nTraining for up to {args.epochs} epochs (patience: {args.patience})...")
    print("-" * 80)
    
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
        scheduler=scheduler,
    )
    
    # ============= SAVE RESULTS =============
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save model in the dataset-specific output directory
    model_path = os.path.join(dataset_output_dir, 'model_hyd_fake_final.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics in the dataset-specific output directory
    metrics_path = os.path.join(dataset_output_dir, 'metrics_hyd_fake_final.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save config
    config = {
        'model': 'HyDFakeModel',
        'version': 'v2',
        'dataset': args.dataset,
        'encoder': args.encoder,
        'architecture': {
            'exogenous_encoder': f'{args.num_gat_layers}-layer GAT with social context',
            'endogenous_encoder': 'Deep text representation learning',
            'fusion': 'Multi-head attention with learnable weights',
            'classifier': 'Deep neural classifier',
        },
        'hyperparameters': {
            'text_frozen_dim': text_dim,
            'graph_in_dim': graph_in_dim,
            'hidden_dim': args.hidden_dim,
            'dropout': 0.3,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'patience': args.patience,
        },
        'optimizations': {
            'augment': args.augment,
            'prune': args.prune,
            'focal_loss': True,
            'batch_norm': True,
            'residual_connections': True,
        },
    }
    config_path = os.path.join(dataset_output_dir, 'config_hyd_fake_final.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # ============= FINAL RESULTS =============
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    if 'auc' in test_metrics:
        print(f"Test AUC:       {test_metrics['auc']:.4f}")
    print("="*80)
    print("Training complete!\n")


if __name__ == '__main__':
    main()