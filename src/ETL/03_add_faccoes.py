"""
Module: Territorial Enrichment with Factions
Description:
    Extraction and Transformation (ETL) pipeline for enriching 
    spatial data with information on criminal territorial dynamics.
    Calculates distances from street segments to organization 
    domain points and generates buffer count features.
    
    Stage 3/4 of the Data Pipeline.
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR,
    CITY_PLACE,
    CSV_FACCOES,
    PROJECTED_EPSG,
    GEODETIC_EPSG,
    FACCAO_BUFFER_M,
    FACCAO_MAX_DIST_M
)

warnings.filterwarnings('ignore')

ENABLE_NEIGHBOR_SMOOTHING = True

def sanitize_column_name(text: str) -> str:
    return (str(text).strip().lower()
            .replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            .replace("-", "_").replace("__", "_"))

def detect_faccoes_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Detecta e padroniza as colunas do CSV de facções."""
    cols = {col.lower(): col for col in df_raw.columns}
    
    if {"faccao", "latitude", "longitude"}.issubset(cols):
        return df_raw[[cols["faccao"], cols["latitude"], cols["longitude"]]].rename(
            columns={cols["faccao"]: "faccao", cols["latitude"]: "latitude", cols["longitude"]: "longitude"}
        )
    
    if {"nome_do_estabelecimento", "latitude", "longitude"}.issubset(cols):
        return df_raw[[cols["nome_do_estabelecimento"], cols["latitude"], cols["longitude"]]].rename(
            columns={cols["nome_do_estabelecimento"]: "faccao", cols["latitude"]: "latitude", cols["longitude"]: "longitude"}
        )
        
    raise ValueError("CSV de facções sem colunas esperadas. Garanta as colunas: 'faccao', 'latitude', 'longitude'.")

def add_faccoes_features(out_dir: Path):
    """Executa o processamento espacial para calcular features de domínio criminal."""
    print(f"\n[1/4] Carregando malha viária de {CITY_PLACE}...")
    edges_path = out_dir / "gdf_edges.pickle"
    
    if not edges_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {edges_path}")

    gdf_edges = pd.read_pickle(edges_path)
    gdf_edges = gpd.GeoDataFrame(gdf_edges, geometry="geometry", crs=f"EPSG:{PROJECTED_EPSG}")
    print(f"  ✓ {len(gdf_edges)} segmentos carregados")

    representative_points = gdf_edges.copy()
    representative_points["geometry"] = representative_points.geometry.representative_point()
    representative_points = representative_points.reset_index(drop=True)
    representative_points["_row"] = representative_points.index

    print(f"\n[2/4] Processando dados de facções: {CSV_FACCOES.name}...")
    if not CSV_FACCOES.exists():
        raise FileNotFoundError(f"CSV de Facções não encontrado em: {CSV_FACCOES}")
        
    raw_df = pd.read_csv(CSV_FACCOES)
    df_faccoes = detect_faccoes_schema(raw_df).dropna(subset=["longitude", "latitude", "faccao"])
    
    gdf_faccoes = gpd.GeoDataFrame(
        df_faccoes,
        geometry=gpd.points_from_xy(df_faccoes["longitude"], df_faccoes["latitude"]),
        crs=f"EPSG:{GEODETIC_EPSG}"
    ).to_crs(epsg=PROJECTED_EPSG)

    faccoes_list = sorted(gdf_faccoes["faccao"].astype(str).unique())
    print(f"  Facções encontradas em {CITY_PLACE}: {faccoes_list}")

    print(f"  Calculando distâncias (Max: {FACCAO_MAX_DIST_M}m)...")
    for faccao in faccoes_list:
        faccao_tag = sanitize_column_name(faccao)
        gdf_faccao_subset = gdf_faccoes[gdf_faccoes["faccao"] == faccao][["geometry"]]
        dist_col_name = f"dist_{faccao_tag}_m"
        
        if len(gdf_faccao_subset) == 0:
            gdf_edges[dist_col_name] = FACCAO_MAX_DIST_M
            continue
            
        joined_distances = gpd.sjoin_nearest(
            representative_points[["_row", "geometry"]], 
            gdf_faccao_subset, 
            how="left", 
            distance_col=dist_col_name
        )
        
        dist_series = (joined_distances.groupby("_row")[dist_col_name].min()
                       .reindex(representative_points["_row"])
                       .fillna(FACCAO_MAX_DIST_M)
                       .clip(upper=FACCAO_MAX_DIST_M))
        
        gdf_edges[dist_col_name] = dist_series.to_numpy()

    print(f"  Calculando presença em buffer de {FACCAO_BUFFER_M}m...")
    buffer_radius = FACCAO_BUFFER_M
    buffer_col = f"buf_{buffer_radius}m"
    
    representative_points[buffer_col] = representative_points.geometry.buffer(buffer_radius)
    rep_buffered = representative_points.set_geometry(buffer_col)
    
    for faccao in faccoes_list:
        faccao_tag = sanitize_column_name(faccao)
        gdf_faccao_subset = gdf_faccoes[gdf_faccoes["faccao"] == faccao][["geometry"]]
        count_col_name = f"cnt_{faccao_tag}_{buffer_radius}m"
        
        gdf_edges[count_col_name] = 0
        
        if len(gdf_faccao_subset) == 0:
            continue
            
        sjoin_buffer = gpd.sjoin(
            gdf_faccao_subset, 
            rep_buffered[["segment_id", buffer_col]], 
            predicate="within", 
            how="left"
        )
        
        if not sjoin_buffer.empty:
            counts = sjoin_buffer.groupby("segment_id").size().rename("count_temp")
            gdf_edges.loc[gdf_edges['segment_id'].isin(counts.index), count_col_name] = \
                gdf_edges['segment_id'].map(counts).fillna(0).astype(int)
    
    if ENABLE_NEIGHBOR_SMOOTHING:
        print(f"\n[3/4] Aplicando suavização de vizinhos em {CITY_PLACE} ...")
        graph_path = out_dir / "graph-streets_as_nodes.pickle"
        
        if graph_path.exists():
            with open(graph_path, "rb") as file:
                graph_line = pickle.load(file)
                
            neighbors_map = {}
            for node_a, node_b in graph_line.edges():
                str_a, str_b = str(node_a), str(node_b)
                neighbors_map.setdefault(str_a, set()).add(str_b)
                neighbors_map.setdefault(str_b, set()).add(str_a)
            
            faccao_features_cols = [col for col in gdf_edges.columns if col.startswith(("dist_", "cnt_"))]
            
            for col in faccao_features_cols:
                values_series = gdf_edges.set_index("segment_id")[col].astype(float)
                mean_values = []
                
                for segment_id in gdf_edges["segment_id"].astype(str):
                    neighbor_nodes = [n for n in neighbors_map.get(segment_id, []) if n in values_series.index]
                    if neighbor_nodes:
                        mean_values.append(float(values_series.reindex(neighbor_nodes).mean()))
                    else:
                        mean_values.append(np.nan) 
                
                smoothed_col_name = f"{col}_nbr_mean"
                gdf_edges[smoothed_col_name] = pd.Series(mean_values).fillna(gdf_edges[col]).fillna(0.0)
        else:
            print(f"  ⚠️ Grafo de vizinhança não encontrado ({graph_path.name}). Pulando smoothing.")

    print(f"\n[4/4] Salvando resultados na pasta de {CITY_PLACE}...")
    gdf_edges.to_pickle(edges_path)
    print(f"  ✓ Arquivo atualizado: {edges_path.name}")
    print(f"  ✓ Total de colunas geradas/atualizadas: {len(gdf_edges.columns)}")

def main():
    parser = argparse.ArgumentParser(description="ETL de Features de Facções")
    parser.add_argument("--process_id", type=str, default="base", help="ID do processamento")
    args = parser.parse_args()
    
    out_dir = PROCESSED_DIR / args.process_id
    
    print(f"\nIniciando ETL de Facções Criminais - {CITY_PLACE}")
    print(f"Input Facções: {CSV_FACCOES.name}")
    print(f"Output Dir: {out_dir}\n")
    
    add_faccoes_features(out_dir)
    print(f"\nETL de Inteligência Territorial para {CITY_PLACE} Concluído.\n")

if __name__ == "__main__":
    main()