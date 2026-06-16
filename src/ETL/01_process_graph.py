"""
Script: 01_process_graph.py
Module: Road Network ETL
Description:
    Extraction and Transformation (ETL) pipeline for urban infrastructure.
    Downloads the road network via OpenStreetMap (OSMnx),
    calculates topological metrics on the line-graph (Betweenness, Closeness) 
    and performs the spatial aggregation of Points of Interest (POIs).
    
    Stage 1/4 of the Data Pipeline.
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DIR,
    RAW_DIR,
    CSV_POIS,
    CITY_PLACE,
    PROJECTED_EPSG,
    GEODETIC_EPSG,
    POI_BUFFER_METERS
)

warnings.filterwarnings('ignore')

ox.settings.use_cache = True
ox.settings.log_console = False

# Tipos de via considerados para features do modelo
HIGHWAY_TYPES = [
    'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
    'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'residential',
    'living_street', 'service', 'unclassified'
]

def _consolidate_roundabouts(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Consolidates roundabout nodes (junction=roundabout) via NetworkX.

    For each connected component of roundabout nodes, replaces all nodes
    with a single representative node at the component centroid.
    Internal edges (self-loops after relabeling) are removed.
    Nodes outside the roundabout graph are unchanged.
    """
    roundabout_edge_pairs = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get('junction') == 'roundabout'
    ]
    if not roundabout_edge_pairs:
        print("  (no roundabouts found)")
        return graph

    G_rb = nx.Graph()
    G_rb.add_edges_from(roundabout_edge_pairs)
    components = [list(c) for c in nx.connected_components(G_rb) if len(c) > 1]

    node_attrs = dict(graph.nodes(data=True))
    mapping = {}

    for comp in components:
        xs = [node_attrs[n]['x'] for n in comp if n in node_attrs]
        ys = [node_attrs[n]['y'] for n in comp if n in node_attrs]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        rep = comp[0]
        graph.nodes[rep]['x'] = cx
        graph.nodes[rep]['y'] = cy
        for n in comp[1:]:
            mapping[n] = rep

    graph_data = graph.graph.copy()
    G_new = nx.relabel_nodes(graph, mapping, copy=True)
    G_new.graph.update(graph_data)

    self_loops = list(nx.selfloop_edges(G_new, keys=True))
    G_new.remove_edges_from(self_loops)

    n_removed = len(graph.nodes) - len(G_new.nodes)
    print(f"  {len(components)} roundabouts consolidated: "
          f"{len(graph.nodes)} -> {len(G_new.nodes)} nodes (-{n_removed}), "
          f"{len(graph.edges)} -> {len(G_new.edges)} edges (-{len(self_loops)})")
    return G_new


def download_osm_graph() -> nx.MultiDiGraph:
    print(f"\n[1/6] Baixando grafo do OpenStreetMap para {CITY_PLACE}...")
    try:
        graph = ox.graph_from_place(
            CITY_PLACE,
            network_type='drive',
            simplify=True
        )
        print(f"  Raw graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        graph = ox.project_graph(graph, to_crs=f"EPSG:{PROJECTED_EPSG}")
        graph = _consolidate_roundabouts(graph)

        print(f"  Final: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    except Exception as e:
        print(f"  Error downloading graph: {e}")
        raise

def process_highway_value(highway_val):
    if isinstance(highway_val, list):
        links = [v for v in highway_val if '_link' in str(v)]
        if links:
            return links[0]
        return highway_val[0]
    return highway_val

def process_edges(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    print("\n[2/6] Processando segmentos viários...")
    
    gdf_edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    gdf_edges = gdf_edges.to_crs(epsg=PROJECTED_EPSG)
    gdf_edges = gdf_edges.reset_index(drop=False)
    
    gdf_edges['segment_id'] = [f"seg_{i:06d}" for i in range(len(gdf_edges))]
    gdf_edges['length_m'] = gdf_edges.geometry.length
    
    gdf_edges['highway'] = gdf_edges['highway'].apply(process_highway_value)
    
    gdf_edges['lanes'] = gdf_edges.get('lanes', 1).apply(
        lambda x: float(x[0]) if isinstance(x, list) else float(x) if pd.notna(x) else 1.0
    )
    gdf_edges['maxspeed'] = gdf_edges.get('maxspeed', 50).apply(
        lambda x: float(x[0]) if isinstance(x, list) else float(x) if pd.notna(x) else 50.0
    )
    gdf_edges['oneway'] = gdf_edges.get('oneway', False).apply(
        lambda x: x if isinstance(x, bool) else (x == 'yes' if isinstance(x, str) else False)
    )
    
    print(f"  Criando features one-hot para {len(HIGHWAY_TYPES)} tipos de via...")
    for hw_type in HIGHWAY_TYPES:
        col_name = f'hw_{hw_type}'
        gdf_edges[col_name] = (gdf_edges['highway'] == hw_type).astype(int)
    
    hw_cols = [f'hw_{t}' for t in HIGHWAY_TYPES]
    hw_sum = gdf_edges[hw_cols].sum(axis=1)
    if (hw_sum == 0).sum() > 0:
        gdf_edges.loc[hw_sum == 0, 'hw_unclassified'] = 1
    
    print(f"  ✓ {len(gdf_edges)} segmentos processados")
    return gdf_edges

def load_existing_pois() -> Optional[gpd.GeoDataFrame]:
    """Carrega POIs do arquivo CSV definido dinamicamente na config."""
    print(f"\n[3/6] Carregando POIs do CSV: {CSV_POIS.name}...")
    
    if not CSV_POIS.exists():
        print(f"  ⚠️  Arquivo não encontrado: {CSV_POIS}")
        return None
    
    df_pois = pd.read_csv(CSV_POIS, sep=';', decimal=',')
    
    if 'tipo_do_estabelecimento' in df_pois.columns:
        df_pois['tipo'] = df_pois['tipo_do_estabelecimento']
    
    gdf_pois = gpd.GeoDataFrame(
        df_pois,
        geometry=gpd.points_from_xy(df_pois.longitude, df_pois.latitude),
        crs=f"EPSG:{GEODETIC_EPSG}"
    )
    gdf_pois = gdf_pois.to_crs(epsg=PROJECTED_EPSG)
    
    print(f"  ✓ {len(gdf_pois)} POIs carregados")
    return gdf_pois

def aggregate_pois_to_edges(gdf_edges: gpd.GeoDataFrame, gdf_pois: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Realiza união espacial (Spatial Join) entre Buffer dos segmentos e POIs."""
    print("\n[4/6] Agregando POIs aos segmentos.")
    
    if gdf_pois is None or len(gdf_pois) == 0:
        return gdf_edges
    
    gdf_edges_buffered = gdf_edges.copy()
    gdf_edges_buffered['geometry'] = gdf_edges_buffered.geometry.buffer(POI_BUFFER_METERS)
    
    poi_types = gdf_pois['tipo'].unique()
    
    for poi_type in poi_types:
        col_name = f'poi_{poi_type}'.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        pois_subset = gdf_pois[gdf_pois['tipo'] == poi_type]
        
        joined = gpd.sjoin(
            gdf_edges_buffered[['segment_id', 'geometry']],
            pois_subset,
            how='left',
            predicate='intersects'
        )
        
        counts = joined[joined['index_right'].notna()].groupby('segment_id').size().to_dict()
        gdf_edges[col_name] = gdf_edges['segment_id'].map(counts).fillna(0).astype(int)
    
    print(f"  ✓ POIs agregados com buffer de {POI_BUFFER_METERS}m")
    return gdf_edges

def compute_network_metrics(gdf_edges: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, nx.DiGraph]:
    """Calcula métricas topológicas no Line Graph (abordagem dual)."""
    print("\n[5/6] Calculando métricas topológicas no Line Graph")
    
    t0 = time.time()
    print("  [1/5] Construindo grafo primal simplificado", end=' ')
    
    edges_df = (
        gdf_edges[['u', 'v', 'length_m']]
        .groupby(['u', 'v'], as_index=False)
        .agg({'length_m': 'mean'})
    )
    
    graph_primal = nx.DiGraph()
    for _, row in edges_df.iterrows():
        graph_primal.add_edge(row.u, row.v, weight=row.length_m)
    
    print(f"✓ ({time.time()-t0:.1f}s)")
    
    t0 = time.time()
    print("  [2/5] Criando Line Graph...", end=' ')
    graph_line = nx.line_graph(graph_primal)
    print(f"✓ ({time.time()-t0:.1f}s)")
    
    for edge in graph_line.edges():
        w1 = graph_primal[edge[0][0]][edge[0][1]].get('weight', 1)
        w2 = graph_primal[edge[1][0]][edge[1][1]].get('weight', 1)
        graph_line.edges[edge]['weight'] = (w1 + w2) / 2

    t0 = time.time()
    print("  [3/5] Mapeando segment_id...", end=' ')
    
    edge_to_segment = {}
    segment_to_edge = {}
    
    for _, row in gdf_edges.iterrows():
        edge_key = (row['u'], row['v'])
        edge_to_segment[edge_key] = row['segment_id']
        edge_to_segment[(row['v'], row['u'])] = row['segment_id']
        segment_to_edge[row['segment_id']] = edge_key
    
    print(f"✓ ({time.time()-t0:.1f}s)")

    n_nodes = graph_line.number_of_nodes()
    k_samples = int(0.02 * n_nodes)
    k_samples = max(500, min(k_samples, 2000))
    
    print(f"  Line Graph: {n_nodes} nós, {graph_line.number_of_edges()} arestas")
    print(f"  Amostragem: k = {k_samples} (~{100*k_samples/n_nodes:.1f}%)")

    t0 = time.time()
    print("  [4/5] Betweenness (aproximado)...", end=' ', flush=True)
    betweenness_line = nx.betweenness_centrality(
        graph_line,
        k=k_samples,
        weight='weight',
        normalized=True
    )
    print(f"✓ ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  [5/5] Closeness (topológica)...", end=' ', flush=True)
    closeness_line = nx.closeness_centrality(graph_line)
    print(f"✓ ({time.time()-t0:.1f}s)")

    degree_line = dict(graph_line.degree())

    t0 = time.time()
    print("  Projetando para segmentos...", end=' ')
    
    def get_metric_for_segment(segment_id, metric_dict):
        edge_key = segment_to_edge.get(segment_id)
        if edge_key is None:
            return 0.0
        return metric_dict.get(edge_key, 0.0)
    
    gdf_edges['betweenness'] = gdf_edges['segment_id'].apply(
        lambda sid: get_metric_for_segment(sid, betweenness_line)
    )
    
    gdf_edges['closeness_mean'] = gdf_edges['segment_id'].apply(
        lambda sid: get_metric_for_segment(sid, closeness_line)
    )
    
    gdf_edges['degree_mean'] = gdf_edges['segment_id'].apply(
        lambda sid: get_metric_for_segment(sid, degree_line)
    )

    print(f"✓ ({time.time()-t0:.1f}s)")
    return gdf_edges, graph_primal


def create_street_nodes_dataframe(gdf_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cria representação de nós baseada nos centróides das arestas."""
    print("\n[6/6] Criando DataFrame de Street Nodes (representação dual)...")
    street_nodes = gdf_edges.copy()
    street_nodes['geometry'] = street_nodes.geometry.representative_point()
    print(f"  ✓ {len(street_nodes)} street nodes criados")
    return street_nodes


def save_outputs(gdf_edges: gpd.GeoDataFrame, street_nodes: gpd.GeoDataFrame, 
                 out_dir: Path, graph_original: nx.MultiDiGraph, graph_primal: nx.DiGraph):
    print(f"\nSalvando resultados em: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gdf_edges.to_pickle(out_dir / "gdf_edges.pickle")
    street_nodes.to_pickle(out_dir / "street_nodes_dataframe.pickle")
    
    with open(out_dir / "graph-crossing_as_nodes.pickle", "wb") as f:
        pickle.dump(graph_original, f)
    
    print("  Gerando Line Graph para serialização...", end=' ')
    graph_line = nx.line_graph(graph_primal)
    with open(out_dir / "graph-streets_as_nodes.pickle", "wb") as f:
        pickle.dump(graph_line, f)
    print("✓")
    
    cols = ['segment_id', 'length_m', 'highway', 'betweenness', 'degree_mean', 'closeness_mean']
    gdf_edges[cols].to_csv(out_dir / "gdf_edges_summary.csv", index=False)
    
    print(f"  ✓ Arquivos salvos: {out_dir.name}.")


def main():
    parser = argparse.ArgumentParser(description="ETL do Grafo Viário")
    parser.add_argument("--process_id", type=str, default="base", help="Identificador do processamento")
    args = parser.parse_args()
    
    out_dir = PROCESSED_DIR / args.process_id
    
    print(f"\nIniciando ETL do Grafo Viário - {CITY_PLACE}")
    print(f"Inputs: {RAW_DIR}")
    print(f"Outputs: {out_dir}\n")
    
    graph_original = download_osm_graph()
    gdf_edges = process_edges(graph_original)
    gdf_pois = load_existing_pois()
    
    gdf_edges = aggregate_pois_to_edges(gdf_edges, gdf_pois)
    gdf_edges, graph_primal = compute_network_metrics(gdf_edges)  
    
    street_nodes = create_street_nodes_dataframe(gdf_edges)
    save_outputs(gdf_edges, street_nodes, out_dir, graph_original, graph_primal)  
    
    print(f"\nETL {CITY_PLACE} Concluído.\n")

if __name__ == "__main__":
    main()