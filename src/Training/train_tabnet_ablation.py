"""
Description:
    Training and Ablation Study script for the TabNet architecture.
    Trains the hierarchical scenarios (Models A, B, and C) focused on general crimes.
    Performs numerical feature scaling.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    RANKINGS_DIR,
    TRAIN_MONTHS,
    EVAL_MONTHS,
    TEST_MONTHS,
    ABLATION_SCENARIOS,
    CITY_PLACE
)

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TARGET_COL = "label"

BEST_PARAMS_PATH = MODELS_DIR / "best_params_tabnet_gpu.json"
OUTPUT_MODELS_DIR = MODELS_DIR / "ablation_tabnet"
OUTPUT_RANKINGS_DIR = RANKINGS_DIR / "ablation_tabnet"

def load_optimized_params() -> dict:
    params = {
        "n_d": 64, "n_a": 64, "n_steps": 7, "gamma": 1.0,
        "lambda_sparse": 0.0001, "learning_rate": 0.005,  
        "batch_size": 8192, "virtual_batch_size": 256
    }
    
    print(f"\n[1/3] Configurando hiperparâmetros do TabNet para {CITY_PLACE}...")
    
    if BEST_PARAMS_PATH.exists():
        try:
            with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as file:
                best_params = json.load(file)
                
            param_mapping = {
                "n_steps": int, "batch_size": int, "gamma": float, 
                "lambda_sparse": float, "learning_rate": float
            }
            for key, cast_func in param_mapping.items():
                if key in best_params: 
                    params[key] = cast_func(best_params[key])
            if "n_da" in best_params: 
                params["n_d"] = params["n_a"] = int(best_params["n_da"])
            print(f"  ✓ Parâmetros otimizados carregados ({BEST_PARAMS_PATH.name})")
        except Exception as e:
            print(f"  ❌ Erro ao processar arquivo de otimização: {e}. Usando fallback.")
    else:
        print(f"  ⚠️ Arquivo de otimização não encontrado. Utilizando parâmetros base.")
        
    return params

def load_and_split_data(data_path: Path):
    print(f"\n[2/3] Carregando e processando dataset...")
    if not data_path.exists():
        raise FileNotFoundError(f"  ❌ Dataset não encontrado em {data_path}")
        
    df_dataset = pd.read_parquet(data_path)
    
    for col in df_dataset.columns:
        if col.startswith("dist_") and pd.api.types.is_numeric_dtype(df_dataset[col]):
            df_dataset[col] = np.log1p(df_dataset[col].clip(lower=0))
            
    df_dataset.fillna(0, inplace=True)
    
    train_df = df_dataset[df_dataset["month_idx"].isin(TRAIN_MONTHS)].copy()
    val_df = df_dataset[df_dataset["month_idx"].isin(EVAL_MONTHS)].copy()
    test_df = df_dataset[df_dataset["month_idx"].isin(TEST_MONTHS)].copy()
    
    return train_df, val_df, test_df

def run_ablation_tabnet(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    print(f"\n[3/3] Iniciando Estudo de Ablação TabNet (Crimes Gerais)...")
    
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_map = {}
    summary_metrics = [] 
    params = load_optimized_params()

    for scenario_name, feature_list in ABLATION_SCENARIOS.items():
        print(f"\n  Treinando {scenario_name} ({len(feature_list)} features)...")

        x_train, y_train = train_df[feature_list].values, train_df[TARGET_COL].values
        x_val, y_val = val_df[feature_list].values, val_df[TARGET_COL].values
        x_test, y_test = test_df[feature_list].values, test_df[TARGET_COL].values

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        pos_weight = (y_train == 0).sum() / max(1, y_train.sum())

        clf = TabNetClassifier(
            n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"], 
            gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=params["learning_rate"]),
            mask_type="entmax", device_name=DEVICE, verbose=0, seed=SEED
        )

        clf.fit(
            X_train=x_train_scaled, y_train=y_train, eval_set=[(x_val_scaled, y_val)],
            max_epochs=40, patience=10, batch_size=params["batch_size"],
            virtual_batch_size=256, weights={0: 1.0, 1: pos_weight}
        )

        
        if scenario_name.lower() == "modelo_c":
            try:
                hist_dict = clf.history.history if hasattr(clf.history, 'history') else dict(clf.history)
                pd.DataFrame(hist_dict).to_csv(OUTPUT_MODELS_DIR / "modelo_c_history.csv", index=False)
            except Exception as e:
                print(f"    ⚠️ Aviso: Falha não-obstrutiva ao exportar histórico ({e})")

        clf.save_model(str(OUTPUT_MODELS_DIR / scenario_name.lower()))
        joblib.dump(scaler, OUTPUT_MODELS_DIR / f"{scenario_name.lower()}_scaler.pkl")
        with open(OUTPUT_MODELS_DIR / f"{scenario_name.lower()}_meta.json", "w", encoding="utf-8") as file:
            json.dump({"city": CITY_PLACE, "features": list(feature_list)}, file, indent=4)
        
        y_pred = clf.predict_proba(x_test_scaled)[:, 1]
        predictions_map[scenario_name] = y_pred
        
        auprc_score = average_precision_score(y_test, y_pred)
        auroc_score = roc_auc_score(y_test, y_pred)
        
        summary_metrics.append({
            "scenario": scenario_name, "n_features": len(feature_list),
            "auprc": auprc_score, "auroc": auroc_score
        })
        print(f"    ✓ AUPRC: {auprc_score:.4f} | AUROC: {auroc_score:.4f}")

    summary_df = pd.DataFrame(summary_metrics)
    summary_path = OUTPUT_MODELS_DIR / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    ranking_df = test_df[["segment_id", "month_idx", TARGET_COL]].copy()
    ranking_df.rename(columns={TARGET_COL: "label_real"}, inplace=True)
    
    for s_name, preds in predictions_map.items():
        ranking_df[f"score_{s_name.lower()}"] = preds
    
    out_rank_path = OUTPUT_RANKINGS_DIR / "test_ranking_tabnet.csv"
    ranking_df.to_csv(out_rank_path, index=False)
    print(f"\n  ✓ Rankings e Resumos exportados com sucesso.")

def main():
    parser = argparse.ArgumentParser(description="Treinamento e Ablação TabNet")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()

    data_path = PROCESSED_DIR / args.process_id / "dataset_node_month.parquet"
    print(f"\nIniciando Pipeline de Treinamento TabNet - {CITY_PLACE}")
    
    train_df, val_df, test_df = load_and_split_data(data_path)
    run_ablation_tabnet(train_df, val_df, test_df)

if __name__ == "__main__":
    main()