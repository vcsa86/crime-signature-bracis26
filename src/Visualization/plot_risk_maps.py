"""
Description:
    Generates static maps for visualizing criminal risk.
    Compares real occurrences (Ground Truth) with predictions of higher 
    risk (Top 5%) from LightGBM and TabNet models for each test month.
"""

import argparse
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    PROCESSED_DIR, 
    RANKINGS_DIR, 
    FIGURES_DIR, 
    GEODETIC_EPSG, 
    TEST_MONTHS,
    CITY_PLACE
)

TOP_K_PERCENT = 0.05
OUTPUT_DIR = FIGURES_DIR / "firefly_maps"
FILE_LGBM = RANKINGS_DIR / "ablation_lgbm" / "test_ranking_lgbm.csv"
FILE_TABNET = RANKINGS_DIR / "ablation_tabnet" / "test_ranking_tabnet.csv"

def load_base_geodata(process_id: str) -> gpd.GeoDataFrame:
    """Carrega a malha viária completa da cidade ativa."""
    edges_geom_path = PROCESSED_DIR / process_id / "gdf_edges.pickle"
    
    print(f"\n[1/3] Carregando malha viária ({CITY_PLACE}): {edges_geom_path.name}")
    
    if not edges_geom_path.exists():
        raise FileNotFoundError(f"  ❌ Malha viária não encontrada em {edges_geom_path}")
        
    gdf_edges = pd.read_pickle(edges_geom_path)

    if not isinstance(gdf_edges, gpd.GeoDataFrame):
        gdf_edges = gpd.GeoDataFrame(gdf_edges)

    gdf_edges["segment_id"] = gdf_edges["segment_id"].astype(str)

    if gdf_edges.crs is None:
        gdf_edges.set_crs(epsg=GEODETIC_EPSG, inplace=True)

    print(f"  ✓ Malha viária carregada com sucesso.")
    return gdf_edges[["segment_id", "geometry"]]

def load_month_data(month: int) -> pd.DataFrame:
    """Carrega os rankings de risco predito e ground truth para o mês especificado."""
    if not FILE_LGBM.exists() or not FILE_TABNET.exists():
        raise FileNotFoundError("  ❌ Ficheiros de ranking não encontrados. Execute a ablação primeiro.")

    df_lgbm = pd.read_csv(FILE_LGBM)
    df_tabnet = pd.read_csv(FILE_TABNET)

    df_lgbm = df_lgbm[df_lgbm["month_idx"] == month].copy()
    df_tabnet = df_tabnet[df_tabnet["month_idx"] == month].copy()

    df_lgbm["segment_id"] = df_lgbm["segment_id"].astype(str)
    df_tabnet["segment_id"] = df_tabnet["segment_id"].astype(str)

    df_real = df_lgbm[["segment_id", "label_real"]].rename(columns={"label_real": "risco_real"})
    df_lgbm_scores = df_lgbm[["segment_id", "score_modelo_c"]].rename(columns={"score_modelo_c": "lgbm_c"})
    df_tabnet_scores = df_tabnet[["segment_id", "score_modelo_c"]].rename(columns={"score_modelo_c": "tabnet_c"})

    df_consolidated = df_real.merge(df_lgbm_scores, on="segment_id", how="inner")
    df_consolidated = df_consolidated.merge(df_tabnet_scores, on="segment_id", how="inner")

    return df_consolidated

def plot_firefly(gdf_full: gpd.GeoDataFrame, gdf_top_k: gpd.GeoDataFrame, title: str, output_path: Path, glow_color: str = "cyan"):
    """Gera o plot com efeito de néon sobre fundo negro profundo."""
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    gdf_full.plot(ax=ax, color="#1a1a1a", linewidth=0.6, zorder=0)

    # Efeito Neon (Camadas de opacidade)
    gdf_top_k.plot(ax=ax, color=glow_color, linewidth=4.5, alpha=0.15, zorder=1)
    gdf_top_k.plot(ax=ax, color=glow_color, linewidth=2.0, alpha=0.35, zorder=2)
    gdf_top_k.plot(ax=ax, color="white", linewidth=0.8, alpha=1.0, zorder=3)

    ax.set_title(title, fontsize=20, color="white", fontweight="bold", pad=30)
    ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"      ✓ {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Geração de Mapas Estáticos Firefly")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()

    print(f"\nIniciando Geração de Mapas Firefly - {CITY_PLACE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gdf_base = load_base_geodata(args.process_id)
        
        print(f"\n[2/3] Processando predições e gerando plots mensais...")

        for month in sorted(TEST_MONTHS):
            print(f"  Mês {month}:")
            
            df_preds = load_month_data(month)
            gdf_month = gdf_base.merge(df_preds, on="segment_id", how="inner")

            if gdf_month.empty:
                print(f"    ⚠️ Aviso: Sem dados espaciais para o mês {month}. A saltar.")
                continue

            month_dir = OUTPUT_DIR / f"mes_{month:02d}"
            month_dir.mkdir(parents=True, exist_ok=True)

            scenarios = [
                ("risco_real", f"{CITY_PLACE} | Ocorrências Reais (Ground Truth) - Mês {month}", "firefly_real.png", "lime"),
                ("lgbm_c", f"{CITY_PLACE} | LightGBM Modelo C (Top 5%) - Mês {month}", "firefly_lgbm.png", "orange"),
                ("tabnet_c", f"{CITY_PLACE} | TabNet Modelo C (Top 5%) - Mês {month}", "firefly_tabnet.png", "magenta"),
            ]

            for score_col, title, filename, color in scenarios:
                if score_col == "risco_real":
                    gdf_subset = gdf_month[gdf_month[score_col] == 1]
                else:
                    threshold = gdf_month[score_col].quantile(1 - TOP_K_PERCENT)
                    gdf_subset = gdf_month[gdf_month[score_col] >= threshold]

                if not gdf_subset.empty:
                    plot_firefly(gdf_base, gdf_subset, title, month_dir / filename, glow_color=color)
                else:
                    print(f"    ⚠️ Cenário {filename} vazio, a saltar.")

        print(f"\n[3/3] Processo concluído com sucesso.")
        print(f"📁 Mapas guardados em: {OUTPUT_DIR}\n")

    except Exception as e:
        print(f"\n❌ ERRO NA GERAÇÃO DOS MAPAS: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()