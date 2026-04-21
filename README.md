# Fairness in Public Screening Models for Hiring Tasks
### Benchmarking and Stress Testing under Distribution Shifts

Diploma Thesis | Data Science | TU Wien  
**Annika Katharina Loos** (12229942)  
Supervisor: Univ.Prof. Dr. Allan Hanbury  
May 2026

---

## Overview

This repository contains the implementation for the master thesis 
*"Fairness in Public Screening Models for Hiring Tasks: Benchmarking 
and Stress Testing under Distribution Shifts"*. The study investigates 
group-level fairness in automated resume screening systems by 
benchmarking classical machine learning and transformer-based NLP 
models on a synthetic CV dataset with explicit demographic annotations.

The experiments cover:
- Baseline fairness and performance evaluation across three models
- Bias mitigation using sample reweighing, label massaging, and 
  group-specific threshold adjustment
- Distribution shift analysis under controlled demographic composition changes

---

## Models Evaluated

| Model | Type | Description |
|-------|------|-------------|
| XGBoost | Classical ML | TF-IDF + structured features |
| BERT | Transformer | General-purpose pretrained LM |
| JobBERT | Transformer | Domain-adapted recruitment LM |

---

## Project Structure
```
thesis_project/
├── src/
│   ├── make_dataset.py          # Dataset construction and label generation
│   ├── run_model.py             # Model training and evaluation pipeline
│   ├── postprocess_thresholds.py# Post-processing threshold optimization
│   ├── distribution_shift.py   # Distribution shift experiments
│   ├── mitigation.py           # Reweighing and massaging implementations
│   ├── evaluation.py           # Performance and fairness metrics
│   ├── labeling.py             # Ground truth label construction
│   ├── splitting.py            # Stratified dataset splitting
│   ├── make_plots.py           # Visualization and reporting
│   └── init.py
├── notebooks/
│   └── 01_data_exploration_labeling.ipynb
├── LICENSE
└── README.md
```
---

## Dataset

The experiments use the **FINDHR Synthetic CV Dataset** 
([Saldívar et al., 2025](https://arxiv.org/abs/2508.21179)), containing 
1,730 semi-synthetic candidate profiles annotated with sensitive 
demographic attributes. The dataset is not included in this repository 
and must be requested separately.

Expected location:
```
data/findhr_synthetic_cv_dataset-ver20251203/
├── semisynthetic_cv/
│   ├── semisynthetic_cv_list.csv
│   └── json_format/
```


---

## Setup

**Requirements:** Python 3.11+

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd thesis_project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing the Experiments

### 1. Build the dataset

```bash
python -m src.make_dataset
```

### 2. Train and evaluate models

```bash
# Baseline (no mitigation)
python -m src.run_model --model xgb --label-col label_unbiased --mitigation none
python -m src.run_model --model bert --label-col label_unbiased --mitigation none
python -m src.run_model --model jobbert --label-col label_unbiased --mitigation none

# Pre-processing: reweighing
python -m src.run_model --model xgb --label-col label_unbiased --mitigation reweight

# Pre-processing: label massaging
python -m src.run_model --model xgb --label-col label_unbiased --mitigation massaging
```

### 3. Post-processing threshold adjustment

```bash
python -m src.postprocess_thresholds --model xgb --label-col label_unbiased \
    --objective eo --attribute gender
python -m src.postprocess_thresholds --model xgb --label-col label_unbiased \
    --objective dp --attribute gender
```

### 4. Distribution shift experiments

```bash
python -m src.distribution_shift --model xgb --label-col label_unbiased \
    --variant none --shift-attr gender --target-value Woman --target-share 0.5
python -m src.distribution_shift --model xgb --label-col label_unbiased \
    --variant none --shift-attr gender --target-value Woman --target-share 0.7
```

### 5. Generate plots and tables

```bash
python -m src.make_plots
```

---

## Fairness Metrics

Group-level fairness is evaluated using:
- **Demographic Parity Difference (DPD)**: disparity in selection rates
- **Equal Opportunity Difference (EOD)**: disparity in true positive rates
- **Group-wise TPR**: per-group true positive rates

All metrics are computed on the held-out test set with respect to 
predefined reference groups (e.g., Men for gender, No for minority status).

---

## License

This project is licensed under the MIT License — see the 
[LICENSE](LICENSE) file for details.

---

## Citation

If you use this code, please cite:

```
Loos, A. K. (2026). Fairness in Public Screening Models for Hiring Tasks:
Benchmarking and Stress Testing under Distribution Shifts.
Diploma Thesis, TU Wien.
```


