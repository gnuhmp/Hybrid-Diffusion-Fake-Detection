
# HyD-Fake — Submission & Run Instructions

This repository contains the implementation, data pipeline, and evaluation for the HyD-Fake hybrid dual-stream fake-news detection project.

Files to submit


Quick setup
1. Create and activate a Python virtual environment (Windows PowerShell example):
```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Install PyTorch and PyTorch Geometric following their official instructions (choose correct CUDA/CPU tag). Example:
```powershell
# Example only — pick the right CUDA wheel for your machine
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
# Then follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

How to run
- Train HyD-Fake (SBERT on PolitiFact):
```powershell
python scripts/train_hyd_fake.py --dataset politifact --encoder sbert
```
- Train a baseline (example):
```powershell
python scripts/train_baseline.py --dataset politifact --model bigcn --encoder sbert
```
- Train all baselines and HyD-Fake for both datasets:
```powershell 
python run_all_baselines.ps1
python run_hyd_fake.ps1
```
Notes
- Reproducibility: All hyperparameters and dataset splits used for the report can be found in `scripts/train_hyd_fake.py` and `data/raw/{dataset}`. Results used in the report are in `results/model_comparison_results.csv`.
- If you need to inspect the DOCX programmatically, the Python package `python-docx` is available (listed in `requirements.txt`).

Contact
- For questions about running experiments or the report generator, open an issue or contact the project owner.
