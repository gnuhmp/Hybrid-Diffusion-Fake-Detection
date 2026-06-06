
# HyD-Fake: Fake News Detection with Hybrid Dual-Stream Model

This project implements **HyD-Fake** - a hybrid dual-stream model for fake news detection, combining graph encoding (exogenous) and text encoding (endogenous) with a gated fusion mechanism.

## Updates Compared to Previous Submission
Changes since the last submission include:
- Formartting improvements for clarity and readability
- Added detailed instructions for running the code and reproducing results
- Added readme sections for environment setup, training, and results viewing

### Main Improvements:
- **Complete model architecture**: Full implementation of HyD-Fake with 4 components (ExogenousContextEncoder, EndogenousPreferenceEncoder, MultiModalFusion, EnhancedNewsClassifier)
- **Comprehensive baselines**: Added 7 baseline models (BiGCN, GCNFN, GNN, GNN-CL, GAT, GCN, GraphSAGE)
- **Support for two encoders**: BERT and SBERT for text encoding
- **Hyperparameter optimization**: Separate configurations for each dataset (PolitiFact, GossipCop)
- **Detailed analysis notebooks**: 
  - `01_compare_embeddings.ipynb` - Compare BERT vs SBERT embeddings
  - `02_diffusion_analysis.ipynb` - Analyze information diffusion mechanisms
  - `03_error_analysis.ipynb` - Detailed error analysis per model
  - `04_model_comparison.ipynb` - Comprehensive performance comparison
- **Verified report**: All 32 F1 scores verified from actual data
- **Complete data pipeline**: Data preparation, augmentation, pruning

## Submitted Files List

### Deliverable Files:
- `SNA-Group03-B1-DetectingFakeNews-FinalReport.pdf` - Final technical report with detailed analysis
- `SNA-Group03-B1-DetectingFakeNews-FinalPowerPoint.pptx` - Presentation slides
- `SNA-Group03-B1-DetectingFakeNews-LinkProject.pdf` - Project link and references documentation
- `README.md` - This file with instructions and details

### Main Directories:
```
HyD-Fake/
├── src/                          # Main source code
│   ├── models/                   # Model implementations
│   │   ├── hyd_fake.py          # Main HyD-Fake model
│   │   └── baselines/           # 7 baseline models
│   ├── training/                # Training components
│   │   ├── trainer.py           # Main trainer
│   │   └── losses.py            # Loss functions (FocalLoss, etc.)
│   ├── data/                    # Data processing
│   │   ├── preprocessing.py
│   │   ├── dataset_builder.py
│   │   └── augment_prune.py
│   └── utils/                   # Utilities
│       ├── data_loader.py
│       └── eval_helper.py
│
├── scripts/                      # Training scripts
│   ├── train_hyd_fake.py        # HyD-Fake training script
│   ├── train_baseline.py        # Baseline training script
│   └── prepare_data.py          # Data preparation
│
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_compare_embeddings.ipynb
│   ├── 02_diffusion_analysis.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_model_comparison.ipynb
│
├── data/                         # Data directory
│   ├── raw/                     # Raw data (PolitiFact, GossipCop)
│   ├── interim/                 # Intermediate embeddings
│   └── processed/               # Processed PyG data
│
├── outputs/                      # Training outputs
│   ├── baselines_bert/
│   ├── baselines_sbert/
│   └── hyd_fake/
│
├── results/                      # Analysis results
│   ├── model_comparison_results.csv
│   ├── model_comparison_summary.txt
│   └── ANALYSIS_REPORT.md
│
├── configs/                      # Configuration files
│   ├── data.yaml
│   └── experiments/
│
├── requirements.txt              # Python dependencies
├── run_hyd_fake.ps1            # HyD-Fake run script
├── run_all_baselines.ps1       # Run all baselines script
├── REPORT_ASSESSMENT_VI.txt    # Result verification report
├── README.md                   # This file
├── SNA-Group03-B1-DetectingFakeNews-FinalReport.pdf
├── SNA-Group03-B1-DetectingFakeNews-FinalPowerPoint.pptx
└── SNA-Group03-B1-DetectingFakeNews-LinkProject.pdf
```

### Important Files:
| File | Description |
|------|--------|
| `src/models/hyd_fake.py` | Main HyD-Fake model with 4 components |
| `scripts/train_hyd_fake.py` | HyD-Fake training script (main entry point) |
| `scripts/train_baseline.py` | Baseline training script |
| `src/training/trainer.py` | Training loop, validation, testing |
| `src/training/losses.py` | FocalLoss, BCE Loss |
| `src/data/preprocessing.py` | Text preprocessing, vocabulary creation |
| `src/data/augment_prune.py` | Data augmentation & edge pruning |
| `notebooks/04_model_comparison.ipynb` | Full results table & visualizations |

## How to Run

### 1. Environment Setup

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv venv
& venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (choose CUDA version matching your machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric
pip install torch-geometric
# Or follow the guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

### 2. Train Models

#### Train HyD-Fake:
```powershell
# Default: PolitiFact, SBERT
python scripts/train_hyd_fake.py

# GossipCop with BERT
python scripts/train_hyd_fake.py --dataset gossipcop --encoder bert

# With augmentation & pruning
python scripts/train_hyd_fake.py --dataset politifact --encoder sbert --prune
```

**Main parameters:**
- `--dataset`: `politifact` or `gossipcop` (default: politifact)
- `--encoder`: `sbert` or `bert` (default: sbert)
- `--epochs`: Number of epochs (default: 120)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--patience`: Early stopping patience (default: 30)
- `--prune`: Enable edge pruning (default: disabled)
- `--no-augment`: Disable data augmentation

#### Train Baselines:
```powershell
# BiGCN on PolitiFact with SBERT
python scripts/train_baseline.py --dataset politifact --encoder sbert --model bigcn

# GAT on GossipCop with BERT
python scripts/train_baseline.py --dataset gossipcop --encoder bert --model gat
```

**Available baseline models:**
- `bigcn` - BiGCN
- `gcnfn` - GCNFN
- `gnn` - GCN/GAT/GraphSAGE (default: GCN)
- `gnncl` - GNN-CL
- `gat` - GAT
- `sage` - GraphSAGE
- `gcn` - GCN

#### Run All at Once:
```powershell
# Run all baselines (both BERT and SBERT)
powershell -ExecutionPolicy Bypass -File run_all_baselines.ps1

# Run HyD-Fake for both datasets x 2 encoders
powershell -ExecutionPolicy Bypass -File run_hyd_fake.ps1
```

### 3. View Results

```powershell
# CSV with all 32 F1 scores
cat results/model_comparison_results.csv

# Statistical summary
cat results/model_comparison_summary.txt

# Detailed analysis report
cat results/ANALYSIS_REPORT.md
```

### 4. Use Jupyter Notebooks (Analysis)

```powershell
# Start Jupyter Lab
jupyter lab

# Open notebooks from notebooks/ directory
# - 04_model_comparison.ipynb: View results table & visualizations
# - 03_error_analysis.ipynb: Analyze per-model errors
# - 02_diffusion_analysis.ipynb: Analyze information diffusion
# - 01_compare_embeddings.ipynb: Compare embeddings
```

## Main Results

### HyD-Fake F1 Scores (Optimized):
| Dataset | SBERT | BERT |
|---------|-------|------|
| PolitiFact | 85.96% | 83.78% |
| GossipCop | 95.85% | 96.32% |
| Average | 90.91% | 90.05% |

### Comparison with Baselines (SBERT):
| Model | PolitiFact | GossipCop | Average |
|-------|-----------|----------|---------|
| HyD-Fake | 85.96% | 95.85% | 90.91% |
| BiGCN | 81.55% | 95.02% | 88.29% |
| SAGE | 79.00% | 97.70% | 88.35% |
| GNN | 78.43% | 97.54% | 87.99% |
| GCNFN | 72.73% | 97.15% | 84.94% |
| GNN-CL | 67.66% | 97.22% | 82.44% |
| GCN | 68.81% | 97.53% | 83.17% |
| GAT | 60.38% | 96.71% | 78.55% |

**Conclusion:** HyD-Fake achieves the highest average F1, particularly strong on PolitiFact (+4.41% vs BiGCN).

## Model Architecture Details

### HyD-Fake Pipeline:
```
Input: Text + Propagation Graph
  ↓
1. ExogenousContextEncoder (GAT)
   - Encode graph structure
   - Output: graph embeddings
  ↓
2. EndogenousPreferenceEncoder
   - Encode text (BERT/SBERT)
   - FC layers
   - Output: text embeddings
  ↓
3. MultiModalFusion
   - Attention gate mechanism
   - Formula: w_g(t) = σ(W_g[g(t), t(t)])
  ↓
4. EnhancedNewsClassifier
   - Deep binary classifier
   - Output: [0, 1] probability
```

### Optimized Hyperparameters:
**PolitiFact:**
- GAT layers: 2 | Dropout: 0.45 | Weight decay: 4e-4
- FocalLoss: α=0.45, γ=2.5

**GossipCop:**
- GAT layers: 4 | Dropout: 0.30 | Weight decay: 1e-4
- FocalLoss: α=0.50, γ=2.0

**Common:**
- Batch size: 16 | Learning rate: 1e-3 | Epochs: 120
- Early stopping: patience 30 | Warmup: 0.1

## Datasets

### Data Structure:
```
data/raw/{politifact,gossipcop}/
├── A.txt                  # Graph adjacency matrix (sparse)
├── graph_labels.npy      # Labels (real: 0, fake: 1)
├── node_graph_id.npy     # Node to graph mapping
├── node_time.npy         # Relative time per node
├── train/val/test_idx.npy # Train/val/test indices
└── raw_text/             # Raw text

data/interim/{embeddings_bert,embeddings_sbert}/{dataset}/
└── Pre-computed embeddings

data/processed/pyg_{bert,sbert}/{dataset}/
└── PyTorch Geometric Data objects
```

### Dataset Statistics:

| Dataset | Graphs | Nodes | Train/Val/Test |
|---------|--------|-------|----------------|
| PolitiFact | 314 | 41,054 | 62/31/221 |
| GossipCop | 5,464 | 314,262 | 1,092/546/3,826 |

## Result Verification

All report results have been **100% verified** from actual data:
- 32 F1 scores (8 models x 2 datasets x 2 encoders)
- Dataset statistics (graphs, nodes, splits)
- 20+ hyperparameter configurations
- 8 model architecture components

See details: [REPORT_ASSESSMENT_VI.txt](REPORT_ASSESSMENT_VI.txt)

## Dependencies

- **Python 3.8+**
- **Deep Learning**: PyTorch, PyTorch Geometric, torch-scatter, torch-sparse, torch-cluster
- **NLP**: sentence-transformers, tqdm
- **Data**: pandas, numpy, scipy, scikit-learn, pyarrow
- **Optional**: jupyterlab, matplotlib (for analysis)

## Tips & Notes

### Speed Optimization:
```powershell
# Use CUDA if available on GPU
python scripts/train_hyd_fake.py --dataset politifact --encoder sbert

# Reduce batch size if running out of memory
python scripts/train_hyd_fake.py --batch_size 8

# Disable augmentation for faster training
python scripts/train_hyd_fake.py --no-augment
```

### Reproducibility:
- All random seeds default to 42
- Dataset configuration in `DATASET_CONFIGS` in `train_hyd_fake.py`
- Full metadata in results CSV

### Troubleshooting:
```powershell
# If PyTorch Geometric error, reinstall:
pip install --upgrade torch-geometric

# If CUDA version mismatch, use CPU:
# (Set device='cpu' in code or install PyTorch CPU version)
```

## Submitted Deliverables

The complete submission includes the following deliverable files:

- **SNA-Group03-B1-DetectingFakeNews-FinalReport.pdf** - Comprehensive technical report with all experimental results, analysis, and findings
- **SNA-Group03-B1-DetectingFakeNews-FinalPowerPoint.pptx** - Presentation slides with key results and visualizations
- **SNA-Group03-B1-DetectingFakeNews-LinkProject.pdf** - Project documentation and references

All results in these documents have been verified against the actual experimental data (see [REPORT_ASSESSMENT_VI.txt](REPORT_ASSESSMENT_VI.txt)).

## Support & Contact

For questions about:
- Running experiments: See `scripts/train_hyd_fake.py` docstring
- Architecture: See `src/models/hyd_fake.py`
- Data: See `src/data/preprocessing.py`
- Results: See `notebooks/04_model_comparison.ipynb`
