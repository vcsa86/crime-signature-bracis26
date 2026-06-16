"""
Description:
    Executes the statistical test via Global Bootstrap for evaluating the 
    incremental performance gain (Delta AUPRC) between the hierarchical ablation models (A, B and C) focused on general crimes.
    Generates histograms of the distribution and calculates empirical P-values and
    95% confidence intervals.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RANKINGS_DIR, RESULTS_DIR, CITY_PLACE

warnings.filterwarnings("ignore")

MODEL_PAIRS = {
    "B_vs_A": ("score_modelo_a", "score_modelo_b"),
    "C_vs_B": ("score_modelo_b", "score_modelo_c"),
    "C_vs_A": ("score_modelo_a", "score_modelo_c"),
}

N_BOOTSTRAP = 1000
RANDOM_SEED = 42

OUTDIR = RESULTS_DIR / "analysis" / "stat_tests"
OUTDIR.mkdir(parents=True, exist_ok=True)

def compute_auprc(y_true, y_score):
    """Calcula Area Under Precision-Recall Curve (AUPRC)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def bootstrap_delta(df_rankings: pd.DataFrame, col_ref: str, col_cmp: str) -> np.ndarray:
    """Realiza reamostragem para calcular a distribuição empírica do delta AUPRC."""
    rng = np.random.default_rng(RANDOM_SEED)
    y_real = df_rankings["label_real"].values
    score_ref = df_rankings[col_ref].values
    score_cmp = df_rankings[col_cmp].values
    n_samples = len(df_rankings)

    deltas = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_samples, n_samples)
        if y_real[idx].sum() == 0:
            continue

        auprc_ref = compute_auprc(y_real[idx], score_ref[idx])
        auprc_cmp = compute_auprc(y_real[idx], score_cmp[idx])
        deltas.append(auprc_cmp - auprc_ref)

    return np.array(deltas)

def plot_delta_hist(deltas: np.ndarray, comp_name: str, model_name: str):
    """Gera o histograma da distribuição dos deltas de performance."""
    plt.figure(figsize=(7, 5))
    plt.hist(deltas, bins=40, edgecolor="black", color='skyblue', alpha=0.8)
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Δ = 0")

    delta_mean = deltas.mean()
    plt.axvline(delta_mean, color="black", linestyle="-", linewidth=2, label=f"Média = {delta_mean:.4f}")

    plt.title(f"Bootstrap ΔAUPRC — {comp_name} ({model_name.upper()})\nTarget: Crimes Gerais - {CITY_PLACE}")
    plt.xlabel("Diferença de AUPRC (Incremental)")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    out_path = OUTDIR / f"hist_bootstrap_{comp_name}_{model_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def run_statistical_test(model_name: str):
    """Executa a análise de bootstrap para os rankings de um modelo específico."""
    print(f"\n  Modelo: {model_name.upper()} | Target: Crimes Gerais")

    ranking_path = RANKINGS_DIR / f"ablation_{model_name}" / f"test_ranking_{model_name}.csv"
    
    if not ranking_path.exists():
        print(f"  ❌ Erro: Arquivo de ranking não encontrado em {ranking_path}")
        return None

    df_rankings = pd.read_csv(ranking_path)
    metrics_rows = []

    for comp, (ref_col, cmp_col) in MODEL_PAIRS.items():
        print(f"    Analisando {comp}...", end=" ")
        
        deltas = bootstrap_delta(df_rankings, ref_col, cmp_col)

        mean_delta = deltas.mean()
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
        p_value = (deltas <= 0).mean()

        metrics_rows.append({
            "model": model_name,
            "comparison": comp,
            "delta_mean": mean_delta,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "p_value": p_value,
            "significant": p_value < 0.05
        })

        plot_delta_hist(deltas, comp, model_name)
        print(f"Δ: {mean_delta:+.4f} | IC95%: [{ci_low:.4f}, {ci_high:.4f}] | p: {p_value:.4f}")

    res_df = pd.DataFrame(metrics_rows)
    out_csv = OUTDIR / f"bootstrap_summary_{model_name}.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"  ✅ Resultados salvos em: {out_csv.name}")
    
    return res_df

def consolidate_results():
    """Consolida os resultados estatísticos de todos os modelos avaliados."""
    all_results = []
    
    print(f"\n📊 Iniciando Testes Bootstrap ΔAUPRC — {CITY_PLACE}")
    
    for model in ["lgbm", "tabnet"]:
        df_result = run_statistical_test(model)
        if df_result is not None:
            all_results.append(df_result)
    
    if all_results:
        consolidated_df = pd.concat(all_results, ignore_index=True)
        out_path = OUTDIR / "bootstrap_summary_consolidated.csv"
        consolidated_df.to_csv(out_path, index=False)
        
        print(f"\n✅ Resumo consolidado exportado: {out_path.name}")
        print("\nResumo Consolidado dos Testes Estatísticos:")
        print(consolidated_df.to_string(index=False))
        
        return consolidated_df
    return None

if __name__ == "__main__":
    consolidate_results()