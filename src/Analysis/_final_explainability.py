"""
Description:
    Executes sequentially all the post-training analysis scripts
    for the general crime prediction model. Consolidates tests of 
    significance, SHAP values, coverage metrics (Top-K) and
    spatial attention of TabNet.
"""

import sys
import warnings
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CITY_PLACE, RESULTS_DIR

warnings.filterwarnings("ignore")

try:
    from src.Analysis import bootstrap_auprc
    from src.Analysis import compute_shap_analysis
    from src.Analysis import compute_topk_coverage
    from src.Analysis import tabnet_explain
except ImportError as e:
    print(f"\n❌ Erro de importação: {e}")
    sys.exit(1)

def main():
    print(f"\nIniciando Pipeline de Explicabilidade e Análise Estatística - {CITY_PLACE}")
    
    try:
        print("\n[1/4] Executando Teste de Significância.")
        bootstrap_auprc.consolidate_results()

        print("\n[2/4] Gerando SHAP Analysis.")
        compute_shap_analysis.main()

        print("\n[3/4] Computando Top-K Coverage.")
        compute_topk_coverage.main()

        print("\n[4/4] Extraindo Atenção da Rede.")
        tabnet_explain.main()

    except Exception as e:
        print(f"\n❌ Erro na execução do pipeline de análise.")
        print(f"  ➡️ Detalhes ({type(e).__name__}): {e}\n")
        sys.exit(1)

    print(f"\nPipeline de Análise para {CITY_PLACE} Concluído com Sucesso.")
    print(f"📁 Artefatos e resultados salvos em: {RESULTS_DIR}\n")

if __name__ == "__main__":
    main()