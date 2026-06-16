"""
Description:
    Predictive model training script using the LightGBM architecture.
    Performs a hierarchical ablation study (Models A, B, and C) 
    to evaluate the impact of different feature sets.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR, RANKINGS_DIR, MODELS_DIR, TRAIN_MONTHS,
    EVAL_MONTHS, TEST_MONTHS, LGBM_PARAMS, ABLATION_SCENARIOS, CITY_PLACE
)

warnings.filterwarnings("ignore")

TARGET_COL = "label"
BEST_PARAMS_PATH = MODELS_DIR / "best_params_lgbm.json"
OUTPUT_RANKINGS_DIR = RANKINGS_DIR / "ablation_lgbm"
OUTPUT_MODELS_DIR = MODELS_DIR / "ablation_lgbm"

def load_optimized_params() -> dict:
    print(f"\nConfigurando hiperparâmetros do LightGBM para {CITY_PLACE}...")
    
    print(f"-> {BEST_PARAMS_PATH.resolve()}")
    
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as file:
            params = json.load(file)
        params.update({"objective": "binary", "metric": "average_precision", "verbosity": -1})
        print(f"  ✓ Parâmetros otimizados carregados")
        return params
        
    print(f"  ⚠️ Arquivo de otimização não encontrado. Utilizando padrão.")
    params = LGBM_PARAMS.copy()
    params.update({"objective": "binary", "metric": "average_precision"})
    return params

def load_and_split_data(data_path: Path):
    print(f"\n Preparando recortes temporais do dataset...")
    if not data_path.exists():
        raise FileNotFoundError(f"  ❌ Arquivo não encontrado: {data_path}")

    df_dataset = pd.read_parquet(data_path)
    for col in df_dataset.columns:
        if col.startswith("dist_") and pd.api.types.is_numeric_dtype(df_dataset[col]):
            df_dataset[col] = np.log1p(df_dataset[col].clip(lower=0))

    df_dataset.fillna(0, inplace=True)
    train_df = df_dataset[df_dataset["month_idx"].isin(TRAIN_MONTHS)].copy()
    val_df = df_dataset[df_dataset["month_idx"].isin(EVAL_MONTHS)].copy()
    test_df = df_dataset[df_dataset["month_idx"].isin(TEST_MONTHS)].copy()

    return train_df, val_df, test_df

def train_ablation_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    print(f"\n Iniciando Estudo de Ablação (Crimes Gerais)...")
    
    params = load_optimized_params()
    predictions_map = {}
    summary_metrics = []  
    
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    for scenario_name, feature_list in ABLATION_SCENARIOS.items():
        print(f"\n  Treinando {scenario_name} ({len(feature_list)} features)...")

        x_train, y_train = train_df[feature_list], train_df[TARGET_COL]
        x_val, y_val = val_df[feature_list], val_df[TARGET_COL]
        x_test, y_test = test_df[feature_list], test_df[TARGET_COL]

        lgb_train = lgb.Dataset(x_train, label=y_train)
        lgb_val = lgb.Dataset(x_val, label=y_val, reference=lgb_train)

        model = lgb.train(
            params, lgb_train, valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False), lgb.log_evaluation(period=0)]
        )

        model_path = OUTPUT_MODELS_DIR / f"{scenario_name.lower()}.txt"
        model.save_model(str(model_path))
        
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        predictions_map[scenario_name] = y_pred

        auprc_score = average_precision_score(y_test, y_pred)
        auroc_score = roc_auc_score(y_test, y_pred)
        
      
        summary_metrics.append({
            "scenario": scenario_name, "n_features": len(feature_list),
            "auprc": auprc_score, "auroc": auroc_score
        })
        print(f"    ✓ AUPRC: {auprc_score:.4f} | AUROC: {auroc_score:.4f}")

    
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(OUTPUT_MODELS_DIR / "ablation_summary.csv", index=False)

    ranking_df = test_df[["segment_id", "month_idx", TARGET_COL]].copy()
    ranking_df.rename(columns={TARGET_COL: "label_real"}, inplace=True)

    for scenario_name, predictions in predictions_map.items():
        ranking_df[f"score_{scenario_name.lower()}"] = predictions

    out_csv_path = OUTPUT_RANKINGS_DIR / "test_ranking_lgbm.csv"
    ranking_df.to_csv(out_csv_path, index=False)
    print(f"\n  ✓ Rankings e Resumos exportados com sucesso.")

def main():
    parser = argparse.ArgumentParser(description="Treinamento e Ablação LightGBM")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()

    data_path = PROCESSED_DIR / args.process_id / "dataset_node_month.parquet"
    print(f"\nIniciando Pipeline de Treinamento - {CITY_PLACE}")
    
    train_df, val_df, test_df = load_and_split_data(data_path)
    train_ablation_models(train_df, val_df, test_df)

if __name__ == "__main__":
    main()