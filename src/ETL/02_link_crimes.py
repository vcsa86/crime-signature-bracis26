"""
Module: Spatial Crime Linking
Description:
    Extraction and Transformation (ETL) pipeline for spatial joining 
    (Spatial Join) between general crime occurrences and the street segments 
    (Street Nodes) processed in the previous stage.
    
    Loads crime data, cleans coordinates, and filters by the stipulated maximum distance.
    
    Stage 2/4 of the Data Pipeline.
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR,
    CITY_PLACE,
    CSV_CRIMES,
    PROJECTED_EPSG,
    GEODETIC_EPSG,
    REQUIRED_CRIME_COLUMNS
)

warnings.filterwarnings('ignore')

def load_crime_data() -> pd.DataFrame:
    """Carrega e padroniza os dados brutos de crimes gerais."""
    print(f"\n[1/3] Carregando dados criminais: {CSV_CRIMES.name}...")

    if not CSV_CRIMES.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_CRIMES}")

    df_crimes = pd.read_csv(CSV_CRIMES)
    df_crimes.columns = [col.strip() for col in df_crimes.columns]

    for col in ["latitude", "longitude"]:
        df_crimes[col] = (
            df_crimes[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        df_crimes[col] = pd.to_numeric(df_crimes[col], errors="coerce")

    print(f"  ✓ Crimes carregados: {len(df_crimes)} registros")
    return df_crimes

def clean_and_prepare_crimes(df_crimes: pd.DataFrame) -> pd.DataFrame:
    """Valida colunas exigidas, extrai variáveis temporais e remove coordenadas nulas."""
    missing_cols = set(REQUIRED_CRIME_COLUMNS) - set(df_crimes.columns)
    if missing_cols:
        raise ValueError(f"Colunas faltando no CSV: {missing_cols}")
    
    df_crimes['data_hora_fato'] = pd.to_datetime(df_crimes['data_hora_fato'])
    df_crimes['data_ocorrencia'] = df_crimes['data_hora_fato'].dt.date
    df_crimes['hora_fato'] = df_crimes['data_hora_fato'].dt.hour
    
    initial_len = len(df_crimes)
    df_crimes = df_crimes.dropna(subset=["latitude", "longitude"])
    
    if len(df_crimes) < initial_len:
        print(f"  Removidos {initial_len - len(df_crimes)} registros sem coordenadas.")
        
    df_crimes['crime_id'] = range(len(df_crimes))
    
    print(f"  Total válido para processamento: {len(df_crimes)} crimes")
    return df_crimes

def link_crimes_to_segments(out_dir: Path, max_distance: float):
    """Executa o fluxo principal de vinculação utilizando os diretórios da cidade ativa."""
    segments_path = out_dir / "street_nodes_dataframe.pickle"
    if not segments_path.exists():
        raise FileNotFoundError(f"Arquivo de segmentos não encontrado: {segments_path}")
        
    gdf_segments = pd.read_pickle(segments_path)
    gdf_segments = gpd.GeoDataFrame(gdf_segments, geometry="geometry", crs=f"EPSG:{PROJECTED_EPSG}")
    
    if gdf_segments['segment_id'].duplicated().any():
        print("  Removendo duplicatas de segmentos...")
        gdf_segments = gdf_segments.drop_duplicates(subset=['segment_id'], keep='first')
    
    seg_pts = gdf_segments.copy()
    seg_pts["geometry"] = seg_pts.geometry.representative_point()
    
    df_crimes = load_crime_data()
    df_crimes = clean_and_prepare_crimes(df_crimes)
    
    gdf_crimes = gpd.GeoDataFrame(
        df_crimes, 
        geometry=gpd.points_from_xy(df_crimes["longitude"], df_crimes["latitude"]), 
        crs=f"EPSG:{GEODETIC_EPSG}"
    ).to_crs(epsg=PROJECTED_EPSG)
    
    print(f"\n[2/3] Executando Spatial Join({CITY_PLACE})...")
    joined_crimes = gpd.sjoin_nearest(
        gdf_crimes, 
        seg_pts[["segment_id", "geometry"]], 
        how="left", 
        distance_col="dist_m"
    )
    
    duplicated_crimes = joined_crimes[joined_crimes['crime_id'].duplicated(keep=False)]
    if len(duplicated_crimes) > 0:
        print(f"  Resolvendo {duplicated_crimes['crime_id'].nunique()} casos de equidistância...")
        joined_crimes = joined_crimes.sort_values('dist_m').drop_duplicates(subset='crime_id', keep='first')
    
    linked = joined_crimes[joined_crimes["dist_m"] <= max_distance].copy()
    ignored = joined_crimes[joined_crimes["dist_m"] > max_distance].copy()
    
    linked = linked.merge(gdf_segments[["segment_id"]], on="segment_id", how="left")
    
    print(f"\nEstatísticas de Vinculação: {CITY_PLACE}")
    print(f"  Linked (<= {max_distance}m): {len(linked)}")
    print(f"  Ignored (> {max_distance}m): {len(ignored)}")
    print(f"  Taxa de Sucesso: {len(linked)/(len(linked)+len(ignored))*100:.1f}%")
    
    if len(ignored) > 0:
        print(f"  Distância Mediana dos Ignorados: {ignored['dist_m'].median():.0f} m")
    
    print(f"\n[3/3] Salvando resultados na pasta da cidade: {out_dir.name}...")
    linked.to_pickle(out_dir / "crime_nodes_dataframe.pickle")
    
    if len(ignored) > 0:
        ignored.to_pickle(out_dir / "crime_nodes_ignored_dataframe.pickle")
    
    print("✓ Vinculação concluída.")

def main():
    parser = argparse.ArgumentParser(description="ETL de Vinculação Espacial de Crimes")
    parser.add_argument("--process_id", type=str, default="base", help="ID do processamento")
    parser.add_argument("--max_distance", type=float, default=200.0, help="Distância máxima de snap")
    args = parser.parse_args()
    
    out_dir = PROCESSED_DIR / args.process_id
    
    print(f"\nIniciando Vinculação de Crimes - {CITY_PLACE}")
    print(f"Input Crimes: {CSV_CRIMES.name}")
    print(f"Output Dir: {out_dir}\n")
    
    link_crimes_to_segments(out_dir, args.max_distance)

if __name__ == "__main__":
    main()