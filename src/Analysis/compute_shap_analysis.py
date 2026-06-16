"""
Description:
    Executes SHAP (Explainable AI) analysis for the LightGBM models focused 
    on general crime prediction.
    Generates Summary Plots (global feature impact), Dependence Plots 
    (non-linear relationships) segmented by ablation scenario, and exports
    the global importance in tabular format (CSV). Also includes monotonicity 
    analysis to validate criminal risk behavior.
"""

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    TEST_MONTHS,
    ABLATION_SCENARIOS,
    CITY_PLACE
)

warnings.filterwarnings("ignore")

LGBM_MODELS_PATH = MODELS_DIR / "ablation_lgbm"
OUTDIR = RESULTS_DIR / "analysis" / "shap_analysis"
TARGET_COL = "label"

def load_test_data(data_path: Path) -> pd.DataFrame:
    print(f"\nCarregando dados de teste para SHAP: {data_path.name}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"  ❌ Ficheiro não encontrado. Execute o ETL primeiro: {data_path}")
        
    df_dataset = pd.read_parquet(data_path)

    for col in df_dataset.columns:
        if col.startswith("dist_") and pd.api.types.is_numeric_dtype(df_dataset[col]):
            df_dataset[col] = np.log1p(df_dataset[col].clip(lower=0))

    df_dataset.fillna(0, inplace=True)
    
    test_df = df_dataset[df_dataset["month_idx"].isin(TEST_MONTHS)].copy()
    print(f"  ✓ Shape de Teste: {test_df.shape}")
    
    return test_df

def run_shap_analysis(test_df: pd.DataFrame, scenario_name: str, sample_size: int = 3000):
    print(f"\n  Iniciando extração SHAP para: {scenario_name}")

    model_file = LGBM_MODELS_PATH / f"{scenario_name.lower()}.txt"
    if not model_file.exists():
        print(f"  ⚠️ Modelo não encontrado ({model_file.name}). Saltando este cenário.")
        return

    model = lgb.Booster(model_file=model_file.as_posix())
    feature_list = ABLATION_SCENARIOS[scenario_name]

    x_test = test_df[feature_list]
    
    x_sample = (
        x_test.sample(n=sample_size, random_state=42)
        if len(x_test) > sample_size
        else x_test
    )

    print(f"    Calculando TreeExplainer (Amostra: {len(x_sample)})...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    scenario_out = OUTDIR / scenario_name.lower()
    scenario_out.mkdir(parents=True, exist_ok=True)

    print("    Exportando Importância Global (Tabular CSV)...")
    mean_abs_impact = np.abs(shap_values).mean(axis=0)
    
    df_shap_tabular = pd.DataFrame({
        'feature': feature_list,
        'shap_importance': mean_abs_impact
    })
    
    df_shap_tabular = df_shap_tabular.sort_values(by='shap_importance', ascending=False)
    
    tabular_path = scenario_out / "shap_feature_importance.csv"
    df_shap_tabular.to_csv(tabular_path, index=False)
    print(f"    ✓ CSV guardado em: {tabular_path.name}")

    print("    Gerando Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, x_sample, show=False, max_display=20)
    plt.title(f"SHAP Summary - {CITY_PLACE} ({scenario_name})")
    plt.tight_layout()
    plt.savefig(scenario_out / "shap_summary_plot.png", dpi=300)
    plt.close()

    print("    Gerando Dependence Plots (Top 3 Features)...")
    top_indices = np.argsort(mean_abs_impact)[-3:][::-1]

    for idx in top_indices:
        feature_name = feature_list[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values, x_sample, show=False)
        plt.title(f"SHAP Dependence: {feature_name} ({CITY_PLACE})")
        plt.tight_layout()
        safe_feat_name = feature_name.replace('(', '').replace(')', '')
        plt.savefig(scenario_out / f"dependence_{safe_feat_name}.png", dpi=200)
        plt.close()

    print("    Executando Teste de Monotonicidade (Correlação de Spearman)...")
    dist_features = [f for f in feature_list if f.startswith("dist_")]
    monotonicity_results = []

    for feature_name in dist_features:
        feat_idx = feature_list.index(feature_name)
        rho, p_value = spearmanr(x_sample[feature_name], shap_values[:, feat_idx])
        
        interpretation = "Esperado (Risco cai com distância)" if rho < -0.1 else "Inconclusivo"
        
        monotonicity_results.append({
            "city": CITY_PLACE,
            "feature": feature_name,
            "spearman_rho": round(rho, 4),
            "p_value": round(p_value, 6),
            "interpretation": interpretation
        })

    if monotonicity_results:
        mono_df = pd.DataFrame(monotonicity_results)
        mono_path = scenario_out / "monotonicity_dist_analysis.csv"
        mono_df.to_csv(mono_path, index=False)
        print(f"    ✓ Análise de monotonicidade guardada.")

    print(f"  ✅ Artefatos guardados em: {scenario_out.name}")

def main():
    parser = argparse.ArgumentParser(description="Análise de Explicabilidade SHAP")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()
    
    data_path = PROCESSED_DIR / args.process_id / "dataset_node_month.parquet"
    
    print(f"\nPipeline SHAP (Explicabilidade AI) - {CITY_PLACE}")
    
    try:
        test_df = load_test_data(data_path)
    except Exception as e:
        print(f"\n  ❌ Erro ao carregar dados: {e}")
        return

    for scenario_name in ABLATION_SCENARIOS.keys():
        run_shap_analysis(test_df, scenario_name)

    print(f"\n✅ Pipeline SHAP concluído com sucesso.")
    print(f"📁 Resultados exportados para: {OUTDIR}\n")

if __name__ == "__main__":
    main()