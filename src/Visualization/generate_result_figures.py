"""
Description:
    Reads the output artifacts from the training pipeline (ablation summaries 
    and TabNet history). Generates consolidated visualizations, including hierarchical 
    performance comparison (Ablation AUPRC/AUROC) and learning curves, using 
    the Seaborn framework.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR, MODELS_DIR, FIGURES_DIR, CITY_PLACE

warnings.filterwarnings("ignore")


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300})


VIGIA_PALETTE = {"LightGBM": "#e67e22", "TabNet": "#9b59b6"}

OUT_FIG_DIR = FIGURES_DIR / "final_plots"

def load_tabnet_history() -> pd.DataFrame:
    """Carrega o histórico de treino do TabNet (Modelo C - Geral)."""
    history_path = MODELS_DIR / "ablation_tabnet" / "modelo_c_history.csv"
    
    if not history_path.exists():
        print(f"  ⚠️ Aviso: Histórico do TabNet não encontrado em {history_path}")
        return None
        
    df_history = pd.read_csv(history_path)
    df_history.columns = [col.lower().replace(' ', '_') for col in df_history.columns]
    
    # Mapeamento dinâmico e seguro das colunas do TabNet
    rename_map = {}
    if 'loss' in df_history.columns:
        rename_map['loss'] = 'train_loss'
        
    if 'val_0_auc' in df_history.columns:
        rename_map['val_0_auc'] = 'valid_auc'
    elif 'val_auc' in df_history.columns:
        rename_map['val_auc'] = 'valid_auc'
        
    if 'val_0_loss' in df_history.columns:
        rename_map['val_0_loss'] = 'valid_loss'
    elif 'val_loss' in df_history.columns:
        rename_map['val_loss'] = 'valid_loss'
        
    df_history.rename(columns=rename_map, inplace=True)
    return df_history

def load_consolidated_ablation_data() -> pd.DataFrame:
    """Carrega e une os resumos de ablação do LGBM e TabNet (Geral)."""
    lgbm_sum_path = MODELS_DIR / "ablation_lgbm" / "ablation_summary.csv"
    tabnet_sum_path = MODELS_DIR / "ablation_tabnet" / "ablation_summary.csv"
    
    dfs_to_concat = []
    
    if lgbm_sum_path.exists():
        df_lgbm = pd.read_csv(lgbm_sum_path)
        df_lgbm['model_arch'] = 'LightGBM'
        dfs_to_concat.append(df_lgbm)
        
    if tabnet_sum_path.exists():
        df_tabnet = pd.read_csv(tabnet_sum_path)
        df_tabnet['model_arch'] = 'TabNet'
        dfs_to_concat.append(df_tabnet)
        
    if not dfs_to_concat:
        print("  ❌ Erro Crítico: Nenhum resumo de ablação encontrado (LGBM ou TabNet).")
        return None
        
    df_consolidated = pd.concat(dfs_to_concat, ignore_index=True)
    
    df_consolidated['scenario'] = df_consolidated['scenario'].replace({
        'Modelo_A': 'Modelo A', 
        'Modelo_B': 'Modelo B', 
        'Modelo_C': 'Modelo C',
        'modelo_a': 'Modelo A', 
        'modelo_b': 'Modelo B', 
        'modelo_c': 'Modelo C'
    })
    
    return df_consolidated

def add_value_labels(ax, format_str='{:.3f}'):
    """Adiciona rótulos de valor em cima das barras do Seaborn."""
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(format_str.format(p.get_height()), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 7), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

def plot_optimized_ablation_comparison(df_ablation: pd.DataFrame):

    print("  [1/2] Gerando gráficos consolidados de ablação (layout vertical corrigido)... ")
    
    # Transforma o DataFrame para formato longo para facilitar o processamento
    df_long = df_ablation.melt(
        id_vars=['scenario', 'model_arch'], 
        value_vars=['auprc', 'auroc'], 
        var_name='metric', value_name='score'
    )
    df_long['metric'] = df_long['metric'].str.upper()
    
    # Dicionário de cores para consistência visual
    # Usando os nomes das arquiteturas diretamente no dicionário
    VIGIA_PALETTE = {"LightGBM": "#e67e22", "TabNet": "#9b59b6"}
    metrics = ['AUPRC', 'AUROC']
    architectures = ['LightGBM', 'TabNet']
    
    # ---------------------------------------------------------
    # AQUI ESTAVA O PROBLEMA! Atualizado para bater com o replace
    scenarios_order = ['Modelo A', 'Modelo B', 'Modelo C'] 
    # ---------------------------------------------------------
    
    for metric in metrics:
        # Cria uma figura com 2 gráficos separados empilhados verticalmente (2 linhas, 1 coluna)
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"{metric}\nCrimes Gerais - {CITY_PLACE}", fontsize=18, y=0.98, fontweight='bold')
        
        # Define a escala do eixo Y para ser consistente para todas as arquiteturas
        y_max = 0.45 if metric == 'AUPRC' else 1.05
        
        for i, arch in enumerate(architectures):
            ax = axes[i]
            # Filtra os dados apenas para a métrica e arquitetura atual
            subset = df_long[(df_long['metric'] == metric) & (df_long['model_arch'] == arch)]
            
            if subset.empty:
                continue
                
            # Garante a ordem correta dos cenários e reindexa
            subset = subset.set_index('scenario').reindex(scenarios_order).reset_index()
            scores = subset['score'].values
            
            # Valor base para calcular a diferença percentual (Δ%)
            base_score = scores[0]

            # Plota as barras
            sns.barplot(
                ax=ax,
                data=subset, 
                x='scenario', 
                y='score', 
                color=VIGIA_PALETTE.get(arch, '#bdc3c7'), # Usa a cor do dicionário VIGIA_PALETTE
                edgecolor='black',
                linewidth=1,
                order=scenarios_order
            )
            
            # Adiciona títulos secundários claros para cada arquitetura
            ax.set_title(f"Arquitetura: {arch}", fontsize=16, pad=10, fontweight='bold')
            ax.set_ylabel(f"Score {metric}", fontsize=14)
            ax.set_xlabel("Cenário de Ablação", fontsize=14)
            ax.set_ylim(0, y_max)
            ax.grid(axis='y', linestyle='-', alpha=0.5)

            # Adiciona rótulos numéricos e variação percentual (Δ%) cumulativa vs Base (A)
            for j, p in enumerate(ax.patches):
                height = p.get_height()
                if pd.isna(height) or height == 0:
                    continue
                
                # O cenário A (Base) mostra apenas o valor. B e C mostram a diferença % vs A (Base).
                if j == 0:
                    label_text = f"{height:.3f}"
                else:
                    pct_diff = ((height - base_score) / base_score) * 100
                    # O formato {:+.1f} força a exibição do sinal de + ou - com uma casa decimal
                    label_text = f"{height:.3f}\n({pct_diff:+.1f}%)"
                
                # Adiciona o texto no topo da barra com deslocamento vertical ajustado para caber o texto quebrado
                ax.annotate(label_text, 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 20), 
                            textcoords='offset points', fontsize=12, fontweight='bold')

        # Ajuste dinâmico para evitar que elementos fiquem de fora
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        file_name = OUT_FIG_DIR / f"ablation_comparison_{metric.lower()}_geral_unificado.png"
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()
        
    print(f"    ✓ {len(metrics)} gráficos de ablação unificados e corrigidos salvos em: {OUT_FIG_DIR.name}")

def plot_optimized_tabnet_history(df_history: pd.DataFrame):
    """Gera subplots com o histórico de Loss e AUC do TabNet (Modelo C)."""
    print("  [2/2] Gerando curvas de aprendizado do TabNet...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss de Treino 
    if 'train_loss' in df_history.columns:
        sns.lineplot(ax=axes[0], data=df_history, x=df_history.index, y='train_loss', label='Treino', color='gray', linestyle='--')
    if 'valid_loss' in df_history.columns:
        sns.lineplot(ax=axes[0], data=df_history, x=df_history.index, y='valid_loss', label='Validação', color=VIGIA_PALETTE["TabNet"], linewidth=2.5)
    
    axes[0].set_title("Evolução da Função de Perda (Loss)", fontsize=14)
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Época")
    
    if 'train_loss' in df_history.columns or 'valid_loss' in df_history.columns:
        axes[0].legend()

    # Plot 2: AUC de Validação
    if 'train_auc' in df_history.columns:
        sns.lineplot(ax=axes[1], data=df_history, x=df_history.index, y='train_auc', label='Treino', color='gray', linestyle='--')
    if 'valid_auc' in df_history.columns:
        sns.lineplot(ax=axes[1], data=df_history, x=df_history.index, y='valid_auc', label='Validação', color="#2ecc71", linewidth=2.5)
    
    axes[1].set_title("Evolução da Métrica AUC", fontsize=14)
    axes[1].set_ylabel("AUC Score")
    axes[1].set_xlabel("Época")
    
    if 'train_auc' in df_history.columns or 'valid_auc' in df_history.columns:
        axes[1].legend()

    fig.suptitle(f"Curvas de Aprendizado TabNet (Modelo C) - {CITY_PLACE}\nTarget: Crimes Gerais", fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    file_name = OUT_FIG_DIR / "tabnet_learning_curves_geral.png"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    
    print(f"    ✓ Curvas de aprendizado salvas.")

def main():
    parser = argparse.ArgumentParser(description="Geração de Figuras de Resultados Finais")
    parser.add_argument("--process_id", type=str, default="base", help="ID da pasta de processamento do ETL")
    args = parser.parse_args()
    
    print(f"\nIniciando Geração de Artefatos Visuais Finais - {CITY_PLACE}")
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    df_ablation = load_consolidated_ablation_data()
    if df_ablation is not None:
        plot_optimized_ablation_comparison(df_ablation)
        
    df_tabnet_hist = load_tabnet_history()
    if df_tabnet_hist is not None:
        plot_optimized_tabnet_history(df_tabnet_hist)
        
    print(f"\n✅ Pipeline de visualização concluído. Figuras salvas em: {OUT_FIG_DIR.name}\n")

if __name__ == "__main__":
    main()