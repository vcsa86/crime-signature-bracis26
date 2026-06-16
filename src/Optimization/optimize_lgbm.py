"""
Script: optimize_lgbm.py
Module: Hyperparameter Optimization (LightGBM)
Description:
    Executes hyperparameter tuning (Grid Search) for the LightGBM model 
    using temporal cross-validation (TimeSeriesSplit).
    Saves the best parameters and the results history in the models folder 
    of the corresponding city.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR, 
    MODELS_DIR, 
    TRAIN_MONTHS, 
    ABLATION_SCENARIOS,
    CITY_PLACE
)

TARGET_COL = "label"
warnings.filterwarnings("ignore")

def load_train_data(data_path: Path):
    """Carrega o dataset e aplica transformações básicas para o treino."""
    print(f"\n[1/2] Carregando dataset de {CITY_PLACE}: {data_path.name}...")
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Erro: Dataset não encontrado em {data_path}. Execute o pipeline ETL primeiro.")
        
    df_dataset = pd.read_parquet(data_path)

    for col in df_dataset.columns:
        if col.startswith("dist_") and pd.api.types.is_numeric_dtype(df_dataset[col]):
            df_dataset[col] = np.log1p(df_dataset[col].clip(lower=0))

    df_dataset = df_dataset[df_dataset["month_idx"].isin(TRAIN_MONTHS)].sort_values("month_idx")
    selected_features = ABLATION_SCENARIOS["Modelo_C"]

    print(f"  ✓ Shape de treino: {df_dataset[selected_features].shape}")
    return df_dataset[selected_features], df_dataset[TARGET_COL]

def run_grid_search(process_id: str):
    """Configura e executa o Grid Search com TimeSeriesSplit."""
    process_dir = PROCESSED_DIR / process_id
    data_path = process_dir / "dataset_node_month.parquet"
    
    best_params_path = MODELS_DIR / "best_params_lgbm.json"
    cv_results_path = MODELS_DIR / "lgbm_gridsearch_cv_results.csv"

    if best_params_path.exists():
        print(f"  ⚠️ Os melhores parâmetros já existem para {CITY_PLACE}.")
        return

    x_train, y_train = load_train_data(data_path)

    param_grid = {
        "num_leaves": [63, 127],
        "learning_rate": [0.03, 0.05],
        "reg_lambda": [0.1, 1.0, 5.0],
        "min_child_samples": [20, 50, 100],
        "colsample_bytree": [0.8, 0.9],
        "n_estimators": [500],
        "max_depth": [-1]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    model = lgb.LGBMClassifier(
        objective="binary", 
        class_weight="balanced", 
        random_state=42, 
        n_jobs=-1, 
        verbose=-1
    )

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring="average_precision", 
        verbose=1
    )

    print(f"\n[2/2] Iniciando Grid Search para {CITY_PLACE}...")
    grid_search.fit(x_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path, index=False)

    with open(best_params_path, "w", encoding="utf-8") as file:
        json.dump(grid_search.best_params_, file, indent=4)

    print(f"\n  Melhor AUPRC alcançada: {grid_search.best_score_:.5f}")
    print(f"  ✅ Parâmetros ótimos salvos em: {best_params_path.name}\n")

def main():
    parser = argparse.ArgumentParser(description="Otimização Grid Search LGBM")
    parser.add_argument("--process_id", type=str, default="base", help="ID do processamento do ETL")
    args = parser.parse_args()
    
    print(f"\nIniciando Otimização de Hiperparâmetros (LightGBM) - {CITY_PLACE}")
    run_grid_search(args.process_id)

if __name__ == "__main__":
    main()