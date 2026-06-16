"""
Module: Spatiotemporal Dataset Construction
Description:
    Final ETL script for consolidation.
    Joins the enriched road network with general crime data to 
    create monthly snapshots. Applies feature equalization logic 
    to ensure dimensional consistency across different cities.
    
    Stage 4/4 of the Data Pipeline.
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR,
    CITY_PLACE,
    CURRENT_CITY,
    FEATURES_MODELO_C 
)

warnings.filterwarnings('ignore')

def load_processed_edges(process_dir: Path) -> pd.DataFrame:
    """Carrega o GeoDataFrame de segmentos previamente processado."""
    file_path = process_dir / "gdf_edges.pickle"
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    return pd.read_pickle(file_path)

def build_spatiotemporal_dataset(process_dir: Path):
    """Constrói o dataset final tabular em Parquet."""
    print(f"\nIniciando construção de Snapshots Mensais - {CITY_PLACE}")
    
    print(f"  Carregando malha viária de {CURRENT_CITY}...")
    gdf_edges = load_processed_edges(process_dir)

    crimes_file = process_dir / "crime_nodes_dataframe.pickle"
    if not crimes_file.exists():
        raise FileNotFoundError(f"Crimes não encontrados em: {crimes_file}")
    
    df_crimes = pd.read_pickle(crimes_file)
    
    date_col = next((col for col in ['data_hora_fato', 'data_ocorrencia', 'date'] if col in df_crimes.columns), None)
    if not date_col: 
        raise ValueError("Coluna de data não encontrada no dataframe de crimes.")

      
    for col in gdf_edges.columns:
        if col.startswith('hw_'):
            gdf_edges[col] = pd.to_numeric(gdf_edges[col], errors='coerce').fillna(0).astype(int)
    
    cols_to_drop = [
        'geometry', 'highway', 'name', 'ref', 'service', 'access', 'bridge', 
        'tunnel', 'junction', 'lanes', 'maxspeed', 'oneway', 'width', 'osmid', 
        'reversed', 'from', 'to'
    ]
    
    cols_final = [
        col for col in gdf_edges.columns 
        if col == 'segment_id' or (
            col not in cols_to_drop and pd.api.types.is_numeric_dtype(gdf_edges[col])
        )
    ]
    
    clean_edges_df = gdf_edges[cols_final].copy()

    df_crimes[date_col] = pd.to_datetime(df_crimes[date_col])
    min_year = df_crimes[date_col].dt.year.min()
    df_crimes['month_idx'] = ((df_crimes[date_col].dt.year - min_year) * 12 + (df_crimes[date_col].dt.month - 1))
    
    months_list = sorted(df_crimes['month_idx'].unique())
    
    grid_data = [{'segment_id': seg, 'month_idx': month} 
                 for seg in gdf_edges['segment_id'].unique() for month in months_list]
    grid_df = pd.DataFrame(grid_data)
    
    print("  Agregando ocorrências criminais gerais ao grid espaço-temporal...")
    crimes_agg = df_crimes.groupby(['segment_id', 'month_idx']).size().reset_index(name='crime_count')
    grid_df = grid_df.merge(crimes_agg, on=['segment_id', 'month_idx'], how='left')
    
    grid_df['label'] = (grid_df['crime_count'].fillna(0) > 0).astype(int)
    
    dataset_df = grid_df.merge(clean_edges_df, on='segment_id', how='left').fillna(0)

    print("  Aplicando equalização de features entre cidades...")
    dataset_df = dataset_df.rename(columns={col: col.replace('poi_poi_', 'poi_') for col in dataset_df.columns if 'poi_poi_' in col})
    
    for feature in FEATURES_MODELO_C:
        if feature not in dataset_df.columns:
            dataset_df[feature] = 0
            
    redundant_cols = ['closeness_u', 'closeness_v', 'degree_u', 'degree_v', 'length', 'crime_count']
    dataset_df.drop(columns=[col for col in redundant_cols if col in dataset_df.columns], inplace=True)
    
    dataset_df['segment_id'] = dataset_df['segment_id'].astype(str)
    
    output_parquet = process_dir / "dataset_node_month.parquet"
    dataset_df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"  ✓ Dataset tabular salvo: {output_parquet.name} (Shape final: {dataset_df.shape})")

def main():
    parser = argparse.ArgumentParser(description="ETL de Consolidação do Dataset Espaço-Temporal Tabular")
    parser.add_argument("--process_id", type=str, default="base", help="ID do processamento")
    args = parser.parse_args()
    
    process_dir = PROCESSED_DIR / args.process_id
    build_spatiotemporal_dataset(process_dir)
    print("\nETL de Dataset Concluído com Sucesso.\n")

if __name__ == "__main__":
    main()