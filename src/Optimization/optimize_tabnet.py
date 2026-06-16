"""
Description:
    Executes hyperparameter optimization for the TabNet model using 
    the Optuna framework (Bayesian Optimization). Implements Pruning 
    strategies (MedianPruner) to halt suboptimal trials early.
    GPU acceleration enabled if available.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    TRAIN_MONTHS,
    EVAL_MONTHS,
    ABLATION_SCENARIOS,
    CITY_PLACE
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_TRIALS = 50

torch.manual_seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

class OptunaPruningCallback(Callback):
    """Callback para interromper trials ruins precocemente utilizando MedianPruner."""

    def __init__(self, trial, metric_name: str = "valid_auc"):
        self.trial = trial
        self.metric_name = metric_name

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        score = logs.get(self.metric_name)
        if score is None:
            return
            
        self.trial.report(score, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def load_and_prepare_data(data_path: Path):
    """Carrega, limpa e padroniza os dados para consumo no TabNet."""
    print(f"\n[1/2] Preparando dados TabNet para {CITY_PLACE}: {data_path.name}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Dataset não encontrado em {data_path}")

    df_dataset = pd.read_parquet(data_path)
    feature_cols = ABLATION_SCENARIOS["Modelo_C"]

    for col in feature_cols:
        if col.startswith("dist_"):
            df_dataset[col] = np.log1p(df_dataset[col].clip(lower=0))

    df_dataset[feature_cols] = df_dataset[feature_cols].fillna(0)

    train_mask = df_dataset["month_idx"].isin(TRAIN_MONTHS)
    val_mask = df_dataset["month_idx"].isin(EVAL_MONTHS)

    x_train_raw = df_dataset.loc[train_mask, feature_cols].values
    y_train = df_dataset.loc[train_mask, "label"].values
    
    x_val_raw = df_dataset.loc[val_mask, feature_cols].values
    y_val = df_dataset.loc[val_mask, "label"].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)

    pos_weight = (y_train == 0).sum() / max(1, y_train.sum())

    print(f"  ✓ Shape de Treino: {x_train_scaled.shape} | Shape de Validação: {x_val_scaled.shape}")
    return x_train_scaled, y_train, x_val_scaled, y_val, pos_weight

def build_objective(x_train, y_train, x_val, y_val, pos_weight):
    """Constrói a função objetivo para o solver do Optuna."""
    def objective(trial):
        n_dimension = trial.suggest_categorical("n_da", [8, 16, 24, 32, 64])

        clf = TabNetClassifier(
            n_d=n_dimension,
            n_a=n_dimension,
            n_steps=trial.suggest_int("n_steps", 3, 7),
            gamma=trial.suggest_float("gamma", 1.0, 1.7),
            lambda_sparse=trial.suggest_float("lambda_sparse", 1e-4, 1e-1, log=True),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={
                "lr": trial.suggest_float("learning_rate", 1e-3, 0.02, log=True)
            },
            mask_type="entmax",
            seed=SEED,
            verbose=0,
            device_name=DEVICE,
        )

        try:
            clf.fit(
                X_train=x_train,
                y_train=y_train,
                eval_set=[(x_val, y_val)],
                eval_name=["valid"],
                eval_metric=["auc"],
                max_epochs=30,
                patience=7,
                batch_size=trial.suggest_categorical("batch_size", [4096, 8192, 16384]),
                virtual_batch_size=256,
                weights={0: 1.0, 1: pos_weight},
                callbacks=[OptunaPruningCallback(trial)],
            )

            predictions = clf.predict_proba(x_val)[:, 1]
            auprc = average_precision_score(y_val, predictions)

        finally:
            del clf
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        return auprc

    return objective

def run_optimization(process_id: str):
    data_path = PROCESSED_DIR / process_id / "dataset_node_month.parquet"
    best_params_path = MODELS_DIR / "best_params_tabnet_gpu.json"
    trials_path = MODELS_DIR / "optuna_results_tabnet.csv"

    if best_params_path.exists():
        print(f"\n  ⚠️ Otimização prévia encontrada em: {best_params_path.name}")
        resposta = input("  Deseja sobrescrever e re-otimizar? (s/n): ").strip().lower()
        if resposta != "s":
            print("  ❌ Operação cancelada pelo usuário.\n")
            return
        print("  ♻️ Sobrescrevendo otimização anterior...")

    x_train, y_train, x_val, y_val, pos_weight = load_and_prepare_data(data_path)

    print(f"\n[2/2] Iniciando Otimização Bayesiana (Trials: {N_TRIALS})")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    
    study.optimize(
        build_objective(x_train, y_train, x_val, y_val, pos_weight), 
        n_trials=N_TRIALS
    )

    print(f"\n Melhor AUPRC: {study.best_value:.5f}")
    print(" Melhores Hiperparâmetros:")
    for key, value in study.best_params.items():
        print(f"      - {key}: {value}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    study.trials_dataframe().to_csv(trials_path, index=False)
    
    with open(best_params_path, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=4)

    print(f"\n  ✅ Histórico de trials salvo em: {trials_path.name}")
    print(f"  ✅ Melhores parâmetros salvos em: {best_params_path.name}\n")

def main():
    parser = argparse.ArgumentParser(description="Otimização Bayesiana TabNet (Optuna)")
    parser.add_argument(
        "--process_id",
        type=str,
        default="base",
        help="ID do processo para localizar o dataset em PROCESSED_DIR/<process_id>/",
    )
    args = parser.parse_args()
    
    print(f"\nIniciando Otimização de Hiperparâmetros (TabNet) - {CITY_PLACE}")
    print(f"Hardware detectado: {DEVICE.upper()}")
    
    run_optimization(args.process_id)

if __name__ == "__main__":
    main()