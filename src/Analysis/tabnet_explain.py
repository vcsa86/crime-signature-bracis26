"""
Description:
    Script to extract and visualize feature importance and attention masks 
    from the TabNet model focused on general crimes.
    Generates bar plots for global attention, heatmaps for local attention 
    (top-risk segments), and exports tabular data (CSV) with the raw 
    attention weights for further analysis.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    PROCESSED_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    TEST_MONTHS,
    CITY_PLACE,
    DICIONARIO_VARIAVEIS
)

warnings.filterwarnings("ignore")

IMG_DIR = FIGURES_DIR / "tabnet_explain"

def load_test_dataframe(data_path: Path, feature_cols: list) -> pd.DataFrame:
    df_dataset = pd.read_parquet(data_path)

    for col in feature_cols:
        if col.startswith("dist_") and pd.api.types.is_numeric_dtype(df_dataset[col]):
            df_dataset[col] = np.log1p(df_dataset[col].clip(lower=0))

    df_dataset[feature_cols] = df_dataset[feature_cols].fillna(0)
    test_df = df_dataset[df_dataset["month_idx"].isin(TEST_MONTHS)].copy()

    return test_df

def run_tabnet_explainability(process_id: str):
    print(f"Extração de Atenção do TabNet - {CITY_PLACE}")
    
    scenario_name = "modelo_c"
    model_folder = MODELS_DIR / "ablation_tabnet"
    data_path = PROCESSED_DIR / process_id / "dataset_node_month.parquet"
    
    model_path = model_folder / f"{scenario_name}.zip"
    scaler_path = model_folder / f"{scenario_name}_scaler.pkl"
    meta_path = model_folder / f"{scenario_name}_meta.json"

    if not model_path.exists():
        print(f"  ❌ Modelo não encontrado em: {model_path}")
        return
    
    if not data_path.exists():
        print(f"  ❌ Dataset não encontrado em: {data_path}")
        return

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("  [1/3] Carregando metadados, dados de teste e pesos do modelo...")
    with open(meta_path, "r", encoding="utf-8") as file:
        meta_info = json.load(file)
        
    feature_cols = meta_info["features"]

    test_df = load_test_dataframe(data_path, feature_cols)
    x_test_raw = test_df[feature_cols].values

    scaler = joblib.load(scaler_path)
    x_test_scaled = scaler.transform(x_test_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabNetClassifier(device_name=device)
    clf.load_model(str(model_path))
    
    print("   Extraindo matriz de atenção global e dados tabulares...")
    explain_matrix, _ = clf.explain(x_test_scaled)
    global_importance = explain_matrix.mean(axis=0)

    df_global = pd.DataFrame({
        'feature': feature_cols,
        'attention_weight': global_importance
    })
    df_global = df_global.sort_values(by='attention_weight', ascending=False)
    csv_global_path = IMG_DIR / "tabnet_global_importance.csv"
    df_global.to_csv(csv_global_path, index=False)
    print(f"    ✓ Dados tabulares de Atenção Global salvos em: {csv_global_path.name}")

    top_indices = np.argsort(global_importance)[-10:]
    top_feature_names_translated = [DICIONARIO_VARIAVEIS.get(feature_cols[i], feature_cols[i]) for i in top_indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_indices)), global_importance[top_indices], color='darkcyan')
    plt.yticks(range(len(top_indices)), top_feature_names_translated)
    plt.title(f"TabNet - Atenção Global (Modelo Completo)\n{CITY_PLACE} | Target: Crimes Gerais")
    plt.xlabel("Peso de Atenção (Importância Média da Máscara)")
    plt.tight_layout()
    
    global_plot_path = IMG_DIR / "tabnet_importance_geral.png"
    plt.savefig(global_plot_path, dpi=300)
    plt.close()
    
    print("   Gerando heatmap de atenção local e extração tabular...")
    predictions = clf.predict_proba(x_test_scaled)[:, 1]
    top_risk_indices = np.argsort(predictions)[-10:]

    df_local = pd.DataFrame(explain_matrix[top_risk_indices], columns=feature_cols)
    df_local.insert(0, 'risk_score', predictions[top_risk_indices])
    df_local = df_local.sort_values(by='risk_score', ascending=False)
    
    csv_local_path = IMG_DIR / "tabnet_local_attention_top_risk.csv"
    df_local.to_csv(csv_local_path, index=False)
    print(f"    ✓ Dados tabulares de Atenção Local (Top Risco) salvos em: {csv_local_path.name}")

    plt.figure(figsize=(14, 7))
    sns.heatmap(
        explain_matrix[top_risk_indices][:, top_indices[::-1]], 
        xticklabels=[DICIONARIO_VARIAVEIS.get(feature_cols[i], feature_cols[i]) for i in top_indices[::-1]],
        yticklabels=[f"Risco: {predictions[i]:.3f}" for i in top_risk_indices],
        cmap="YlGnBu",
        cbar_kws={'label': 'Intensidade da Atenção'}
    )
    plt.title(f"Heatmap de Atenção Local (Top 10 Segmentos de Maior Risco) - {CITY_PLACE}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    heatmap_plot_path = IMG_DIR / "tabnet_heatmap_geral.png"
    plt.savefig(heatmap_plot_path, dpi=300)
    plt.close()

    print(f"\n Interpretability Analysis Complete.")
    print(f" Files saved to: {IMG_DIR}\n")

def main():
    parser = argparse.ArgumentParser(description="Interpretabilidade TabNet (Attention Masks)")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()
    
    run_tabnet_explainability(args.process_id)

if __name__ == "__main__":
    main()