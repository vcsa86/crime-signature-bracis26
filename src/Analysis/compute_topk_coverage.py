"""
Description:
    Calculates the Top-K coverage operational metrics. Evaluates the percentage 
    of street segments and the proportion of actual crimes captured if only the 
    highest-risk fractions (e.g., Top 1%, 5%, 10%) identified by the models 
    are targeted for patrolling.
    Compares performance across all ablation scenarios (Models A, B, and C)
    for the LightGBM and TabNet architectures.
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    RANKINGS_DIR, 
    RESULTS_DIR, 
    PROCESSED_DIR, 
    CITY_PLACE
)

warnings.filterwarnings("ignore")

OUTDIR = RESULTS_DIR / "analysis" / "topk_analysis"

def enrich_with_crime_counts(df_ranking: pd.DataFrame, process_id: str) -> pd.DataFrame:
    """Cruza o ranking de probabilidades com a contagem real e absoluta de crimes."""
    crime_nodes_path = PROCESSED_DIR / process_id / "crime_nodes_dataframe.pickle"
    
    if not crime_nodes_path.exists():
        print(f"  ⚠️ Ficheiro de crimes originais não encontrado. Assumindo crime_count = label_real.")
        df_ranking["num_crimes"] = df_ranking["label_real"]
        return df_ranking
        
    crimes_raw = pd.read_pickle(crime_nodes_path)
    
    date_col = next(col for col in ['data_hora_fato', 'data_ocorrencia', 'date'] if col in crimes_raw.columns)
    crimes_raw[date_col] = pd.to_datetime(crimes_raw[date_col])
    min_year = crimes_raw[date_col].dt.year.min()
    crimes_raw['month_idx'] = (crimes_raw[date_col].dt.year - min_year) * 12 + (crimes_raw[date_col].dt.month - 1)
    
    counts = crimes_raw.groupby(["segment_id", "month_idx"]).size().reset_index(name="num_crimes")
    
    df_ranking["segment_id"] = df_ranking["segment_id"].astype(str)
    counts["segment_id"] = counts["segment_id"].astype(str)
    
    enriched_df = df_ranking.merge(counts, on=["segment_id", "month_idx"], how="left").fillna(0)
    return enriched_df

def aggregate_by_segment(df_enriched: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    
    agg_dict = {col: "first" for col in score_cols}
    agg_dict["label_real"] = "max"
    agg_dict["num_crimes"] = "sum"
    return df_enriched.groupby("segment_id").agg(agg_dict).reset_index()

def compute_topk_metrics(df_ranking: pd.DataFrame, score_col: str, model_name: str, scenario_name: str) -> pd.DataFrame:
    """Calcula as métricas de cobertura (Segs, Crimes e Lift) para os percentis de corte."""
    metrics_rows = []
    total_segs_with_crime = df_ranking["label_real"].sum()
    total_crimes_absolute = df_ranking["num_crimes"].sum()
    
    cutoffs = [1, 5, 10, 20, 50]
    
    for pct in cutoffs:
        n_samples = max(1, int(len(df_ranking) * pct / 100))
        topk_subset = df_ranking.sort_values(score_col, ascending=False).head(n_samples)
        
        coverage_segs = topk_subset["label_real"].sum() / total_segs_with_crime if total_segs_with_crime > 0 else 0
        coverage_crimes = topk_subset["num_crimes"].sum() / total_crimes_absolute if total_crimes_absolute > 0 else 0
        lift = coverage_crimes / (pct / 100) if pct > 0 else 0
        
        metrics_rows.append({
            "model_name": model_name,
            "scenario": scenario_name,
            "topk_pct": pct, 
            "coverage_segs": coverage_segs, 
            "coverage_crimes": coverage_crimes,
            "n_segs_covered": topk_subset["label_real"].sum(), 
            "n_crimes_covered": topk_subset["num_crimes"].sum(),
            "lift": lift
        })
        
    return pd.DataFrame(metrics_rows)

def print_coverage_report(df_metrics: pd.DataFrame):
    """Imprime o relatório formatado das métricas de cobertura no terminal."""
    model_name = df_metrics.iloc[0]["model_name"]
    scenario = df_metrics.iloc[0]["scenario"]
    
    print(f"\n  📊 {model_name.upper()} — {scenario.upper()}")
    print(f"  {'Corte %':<8} | {'CobSegs':>8} | {'CobCri':>8} | {'Segs':>7} | {'Crimes':>8} | {'Lift':>8}")
    
    for _, row in df_metrics.iterrows():
        print(f"  {int(row['topk_pct']):>5}%   | {row['coverage_segs']:>7.1%} | {row['coverage_crimes']:>7.1%} | "
              f"{int(row['n_segs_covered']):>7,} | {int(row['n_crimes_covered']):>8,} | {row['lift']:>7.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Análise Operacional de Top-K Coverage")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()
    
    print(f"\nIniciando Análise Top-K Consolidada - {CITY_PLACE}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []

    # 1. Avaliação LightGBM
    lgbm_ranking_path = RANKINGS_DIR / "ablation_lgbm" / "test_ranking_lgbm.csv"
    if lgbm_ranking_path.exists():
        print("  Processando rankings do LightGBM...")
        df_lgbm = enrich_with_crime_counts(pd.read_csv(lgbm_ranking_path), args.process_id)
        df_lgbm = aggregate_by_segment(df_lgbm, ["score_modelo_a", "score_modelo_c"])
        print(f"  Segmentos únicos LightGBM: {len(df_lgbm):,}")

        metrics_lgbm_a = compute_topk_metrics(df_lgbm, "score_modelo_a", "LightGBM", "Modelo A (Base)")
        metrics_lgbm_c = compute_topk_metrics(df_lgbm, "score_modelo_c", "LightGBM", "Modelo C (Facções)")
        
        print_coverage_report(metrics_lgbm_a)
        print_coverage_report(metrics_lgbm_c)
        
        all_metrics.extend([metrics_lgbm_a, metrics_lgbm_c])
    else:
        print(f"  ⚠️ Ficheiro de ranking LightGBM não encontrado em {lgbm_ranking_path.name}")

    # 2. Avaliação TabNet
    tabnet_ranking_path = RANKINGS_DIR / "ablation_tabnet" / "test_ranking_tabnet.csv"
    if tabnet_ranking_path.exists():
        print("\n  Processando rankings do TabNet...")
        df_tabnet = enrich_with_crime_counts(pd.read_csv(tabnet_ranking_path), args.process_id)
        df_tabnet = aggregate_by_segment(df_tabnet, ["score_modelo_a", "score_modelo_c"])
        print(f"  Segmentos únicos TabNet: {len(df_tabnet):,}")

        metrics_tabnet_a = compute_topk_metrics(df_tabnet, "score_modelo_a", "TabNet", "Modelo A (Base)")
        metrics_tabnet_c = compute_topk_metrics(df_tabnet, "score_modelo_c", "TabNet", "Modelo C (Facções)")
        
        print_coverage_report(metrics_tabnet_a)
        print_coverage_report(metrics_tabnet_c)
        
        all_metrics.extend([metrics_tabnet_a, metrics_tabnet_c])
    else:
        print(f"  ⚠️ Ficheiro de ranking TabNet não encontrado em {tabnet_ranking_path.name}")

    # 3. Consolidação
    if all_metrics:
        consolidated_df = pd.concat(all_metrics, ignore_index=True)
        out_csv_path = OUTDIR / "topk_coverage_summary.csv"
        consolidated_df.to_csv(out_csv_path, index=False)
        print(f"\n✅ Relatório consolidado guardado em: {out_csv_path.name}\n")
    else:
        print("\n❌ Não foi possível gerar o relatório pois nenhum ranking foi encontrado.\n")

if __name__ == "__main__":
    main()