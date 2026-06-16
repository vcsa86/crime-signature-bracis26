# Crime Signature: Gang Graffiti for Predicting Criminal Risk in Street Networks

Project focused on criminal risk prediction (General Crimes) using a Hierarchical Modeling approach (Incremental Ablation). The project compares the efficiency of Gradient Boosting models (LightGBM) and Deep Learning on tabular data (TabNet).

## Project Overview
The objective is to validate the hypothesis that the progressive addition of territorial intelligence and network metrics increases predictive accuracy and the operational efficiency of public security forces.

## 🔒 Data Availability and Confidentiality
Due to strict confidentiality terms with the public security department and the sensitive nature of criminal intelligence data (crimes in general and faction territories), the raw and processed datasets cannot be made publicly available.

The source code in this repository is provided solely to demonstrate the transparency of the methodological flow, the spatial resource engineering process, and the experimental setup for the hierarchical ablation study.

### Hierarchical Modeling (Ablation Study)
**1. Model A (Baseline):** Only road infrastructure and basic urban network metrics (Highway types, Betweenness, Closeness, Degree).

**2.Model B (+ POIs):** Adds the spatial distribution of Points of Interest (Banks, Schools, Bars, etc.).

**3.Model C (+ Territorial Dynamics):** Adds territorial intelligence variables (Distance and spatial influence of criminal organizations/gangs).


## 🚀 How to Run

### 1. Data Pipeline (ETL)
Extraction, graph processing, georeferencing, and creation of the Parquet dataset.
```bash
python src/ETL/01_process_graph.py
python src/ETL/02_link_crimes.py
python src/ETL/03_add_faccoes.py
python src/ETL/04_build_dataset.py
```

### 2. Optimization & Training
```bash
# Optimization
python src/Optimization/optimize_lgbm.py
python src/Optimization/optimize_tabnet.py

# Training
python src/Training/train_lgbm_ablation.py
python src/Training/train_tabnet_ablation.py
```
### 3. Analysis
```bash
python src/Analysis/_final_explainability.py
```

## 📂 Project Structure

```text
PROJECT/
├── data/                                   
│   ├── raw/ (City 1 & City 2)
│   └── processed/                          # Final Parquet Datasets
├── src/                                    # Source Code (Anonymized)
│   ├── ETL/                                # Data Engineering
│   ├── Optimization/                       # Hyperparameter Tuning
│   ├── Training/                           # Ablation Experiments
│   └── Analysis/                           # XAI (SHAP, Bootstrap, Top-K)
├── results/                                # Output Artifacts
│   ├── models/                             # Persisted .txt and .zip models
│   ├── rankings/                           # Probability CSVs
│   └── analysis/                           # Statistical tables
└── requirements.txt

