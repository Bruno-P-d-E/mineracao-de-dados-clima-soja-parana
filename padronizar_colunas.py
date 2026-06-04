#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Padronização de Nomes de Colunas para snake_case
=================================================
Converte colunas sem "dec" para snake_case para consistência.

Mapeamento:
  Município → municipio
  Ano → ano
  Área plantada ou destinada à colheita (Hectares) → area_plantada_ha
  Área plantada ou destinada à colheita - percentual do total geral → area_plantada_pct
  Área colhida (Hectares) → area_colhida_ha
  Área colhida - percentual do total geral → area_colhida_pct
  Quantidade produzida (Toneladas) → quantidade_produzida_ton
  Rendimento médio da produção (Quilogramas por Hectare) → rendimento_kg_ha
  Valor da produção (Mil Reais) → valor_producao_mil_reais
  Valor da produção - percentual do total geral → valor_producao_pct
"""

import pandas as pd
from pathlib import Path

# Mapeamento de nomes antigos → novos (snake_case)
RENAME_MAP = {
    'Município': 'municipio',
    'Ano': 'ano',
    'Área plantada ou destinada à colheita (Hectares)': 'area_plantada_ha',
    'Área plantada ou destinada à colheita - percentual do total geral': 'area_plantada_pct',
    'Área colhida (Hectares)': 'area_colhida_ha',
    'Área colhida - percentual do total geral': 'area_colhida_pct',
    'Quantidade produzida (Toneladas)': 'quantidade_produzida_ton',
    'Rendimento médio da produção (Quilogramas por Hectare)': 'rendimento_kg_ha',
    'Valor da produção (Mil Reais)': 'valor_producao_mil_reais',
    'Valor da produção - percentual do total geral': 'valor_producao_pct',
}

def padronizar_colunas(df):
    """Aplica padronização de snake_case ao DataFrame."""
    # Renomear colunas conhecidas
    df = df.rename(columns=RENAME_MAP)
    return df

if __name__ == "__main__":
    print("="*70)
    print("PADRONIZAÇÃO DE COLUNAS: snake_case")
    print("="*70)
    
    # Atualizar PAM_SIDRA.csv
    pam_path = Path('data/interim/PAM_SIDRA/PAM_SIDRA.csv')
    print(f"\n[1/1] Padronizando {pam_path}...")
    df_pam = pd.read_csv(pam_path)
    print(f"      Antes: {list(df_pam.columns[:5])}")
    
    df_pam = padronizar_colunas(df_pam)
    df_pam.to_csv(pam_path, index=False, encoding='utf-8-sig')
    print(f"      Depois: {list(df_pam.columns[:5])}")
    print(f"      ✓ Salvo: {pam_path}")
    
    print("\n" + "="*70)
    print("✓ PADRONIZAÇÃO COMPLETA")
    print("="*70)
    print("\nProximos passos:")
    print("  1. Executar novamente: python src/preprocessing/merge.py")
    print("  2. Iniciar dashboard: streamlit run src/dashboard/dashboard.py")
