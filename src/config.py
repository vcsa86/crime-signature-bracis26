"""
Description:
Central configuration file for the pipeline.
Defines spatial and temporal parameters, data directories, and model hyperparameters
for predicting general crime risk in urban street networks.
Manages ablation scenarios (Models A, B, and C) using road network data and territorial dynamics.
"""

import os
from pathlib import Path

CURRENT_CITY = "city" # Replace with the target city key (e.g., "city" or "city2")


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

CITY_CONFIGS = {
    "city": {
        "place_name": "City, State, Country",
        "raw_subfolder": "city",
        "files": {
            "crimes": "crimes.csv",
            "pois": "pois_city.csv",
            "faccoes": "faccoes_city.csv"
        },
        "faccao_prefixes": ["faccao_a", "faccao_b", "faccao_c"]
    },
    "city2": {
        "place_name": "City2, State2, Country2",
        "raw_subfolder": "city2",
        "files": {
            "crimes": "crimes.csv",
            "pois": "pois_city.csv",
            "faccoes": "faccoes_city.csv"
        },
        "faccao_prefixes": ["faccao_a", "faccao_b"] 
    }
}

ACTIVE_CONFIG = CITY_CONFIGS[CURRENT_CITY]

RAW_DIR = DATA_DIR / "raw" / ACTIVE_CONFIG["raw_subfolder"]
PROCESSED_DIR = DATA_DIR / "processed" / CURRENT_CITY

RESULTS_DIR = ROOT_DIR / "results" / CURRENT_CITY
MODELS_DIR = RESULTS_DIR / "models"
RANKINGS_DIR = RESULTS_DIR / "rankings"
FIGURES_DIR = RESULTS_DIR / "figures"

for dir_path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RANKINGS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CSV_POIS = RAW_DIR / ACTIVE_CONFIG["files"]["pois"]
CSV_FACCOES = RAW_DIR / ACTIVE_CONFIG["files"]["faccoes"]
CSV_CRIMES = RAW_DIR / ACTIVE_CONFIG["files"]["crimes"]

DATASET_FINAL_PARQUET = PROCESSED_DIR / "base" / "dataset_node_month.parquet"

# ===================================
# SPATIAL AND TECHNICAL CONFIGURATIONS
# ===================================

CITY_PLACE = ACTIVE_CONFIG["place_name"]
GEODETIC_EPSG = 4326 
PROJECTED_EPSG = 99999 #Replace with the city's UTM code.

POI_BUFFER_METERS = 200
FACCAO_BUFFER_M = 200
FACCAO_MAX_DIST_M = 1000.0


REQUIRED_CRIME_COLUMNS = [
    "id_atendimento", 
    "data_hora_fato", 
    "latitude", 
    "longitude", 
    "classificacoes"
]

# Ablation Scenarios and Features


FEATURES_MODELO_A = [
    "betweenness", "closeness_mean", "degree_mean", "length_m",
    "hw_living_street", "hw_motorway", "hw_motorway_link", "hw_primary",
    "hw_primary_link", "hw_residential", "hw_secondary", "hw_secondary_link",
    "hw_service", "hw_tertiary", "hw_tertiary_link", "hw_trunk", "hw_trunk_link", "hw_unclassified"
]

FEATURES_MODELO_B = FEATURES_MODELO_A + [
    "poi_banco", "poi_bar", "poi_clinica_posto_de_saude", "poi_creche_jardim_de_infancia",
    "poi_delegacia_de_policia", "poi_edificio_governamental", "poi_escola", "poi_hospital",
    "poi_hotel", "poi_local_de_eventos", "poi_posto_de_gasolina", "poi_quartel_(militar_policia)",
    "poi_quartel_exercito", "poi_restaurante", "poi_teatro", "poi_transporte", "poi_universidade_faculdade"
]

def get_modelo_c_features(base_features: list, city_name: str) -> list:
    
    faccoes_map = {
        "City, State, Country": ["faccao_a", "faccao_b", "faccao_c"],
        "City2, State2, Country2": ["faccao_c", "faccao_d"]
    }
    
    prefixes = faccoes_map.get(city_name, [])
    fac_features = []
    
    for prefix in prefixes:
        fac_features.extend([
            f"cnt_{prefix}_200m", 
            f"cnt_{prefix}_200m_nbr_mean",
            f"dist_{prefix}_m", 
            f"dist_{prefix}_m_nbr_mean"
        ])
        
    return base_features + fac_features

FEATURES_MODELO_C = get_modelo_c_features(FEATURES_MODELO_B, CITY_PLACE)

ABLATION_SCENARIOS = {
    "Modelo_A": FEATURES_MODELO_A,
    "Modelo_B": FEATURES_MODELO_B,
    "Modelo_C": FEATURES_MODELO_C,
}

# MODEL TEMPORAL SETTINGS AND HYPERPARAMETERS

TRAIN_MONTHS = list(range(0, 18))
EVAL_MONTHS = [18, 19]
TEST_MONTHS = [20, 21, 22, 23]

# Parâmetros default para o LightGBM
LGBM_PARAMS = {
    "n_estimators": 500, 
    "learning_rate": 0.05, 
    "max_depth": -1, 
    "num_leaves": 127,
    "min_child_samples": 80, 
    "subsample": 0.85, 
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0, 
    "reg_alpha": 0.3, 
    "n_jobs": -1, 
    "verbose": -1, 
    "random_state": 42
}

MODEL_PARAMS = LGBM_PARAMS

DICIONARIO_VARIAVEIS = {
    "betweenness": "Betweenness",
    "closeness_mean": "Closeness",
    "degree_mean": "Degree",
    "length_m": "Street Length",
    "dist_faccao_a_m": "Distance Gang A",
    "dist_faccao_a_m_nbr_mean": "Distance Gang A (Neighbors)",
    "cnt_faccao_a_200m": "Density Gang A",
    "cnt_faccao_a_200m_nbr_mean": "Density Gang A (Neighbors)",
    "dist_faccao_b_m": "Distance Gang B",
    "dist_faccao_b_m_nbr_mean": "Distance Gang B (Neighbors)",
    "cnt_faccao_b_200m": "Density Gang B",
    "cnt_faccao_b_200m_nbr_mean": "Density Gang B (Neighbors)",
    "dist_faccao_c_m": "Distance Gang C",
    "dist_faccao_c_m_nbr_mean": "Distance Gang C (Neighbors)",
    "cnt_faccao_c_200m": "Density Gang C",
    "cnt_faccao_c_200m_nbr_mean": "Density Gang C (Neighbors)",
    "dist_faccao_d_m": "Distance Gang D",
    "dist_faccao_d_m_nbr_mean": "Distance Gang D (Neighbors)",
    "cnt_faccao_d_200m": "Density Gang D",
    "cnt_faccao_d_200m_nbr_mean": "Density Gang D (Neighbors)",
    "hw_living_street": "Living Street",
    "hw_motorway": "Motorway",
    "hw_motorway_link": "Motorway Link",
    "hw_primary": "Primary Road",
    "hw_primary_link": "Primary Link",
    "hw_residential": "Residential",
    "hw_secondary": "Secondary Road",
    "hw_secondary_link": "Secondary Link",
    "hw_service": "Service Road",
    "hw_tertiary": "Tertiary Road",
    "hw_tertiary_link": "Tertiary Link",
    "hw_trunk": "Trunk Road",
    "hw_trunk_link": "Trunk Link",
    "hw_unclassified": "Unclassified",
    "poi_banco": "Bank",
    "poi_bar": "Bar",
    "poi_clinica_posto_de_saude": "Health Clinic",
    "poi_creche_jardim_de_infancia": "Daycare",
    "poi_delegacia_de_policia": "Police Station",
    "poi_edificio_governamental": "Government Building",
    "poi_escola": "School",
    "poi_hospital": "Hospital",
    "poi_hotel": "Hotel",
    "poi_local_de_eventos": "Event Venue",
    "poi_posto_de_gasolina": "Gas Station",
    "poi_quartel_(militar_policia)": "Military/Police Barracks",
    "poi_quartel_exercito": "Army Barracks",
    "poi_restaurante": "Restaurant",
    "poi_teatro": "Theater",
    "poi_transporte": "Public Transport",
    "poi_universidade_faculdade": "University"
}

