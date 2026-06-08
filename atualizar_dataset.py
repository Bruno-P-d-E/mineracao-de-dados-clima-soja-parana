#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para atualizar o dataset com:
1. Coluna valor_producao_ipca_mil_reais (corrigido por IPCA)
2. Coluna valor_producao_ipca_mil_reais_ha
3. Matriz de correlação de Pearson
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Índices Acumulados de IPCA (consolidados para 2024 como base) ─────
# Cada taxa representa a variação que ocorreu DURANTE aquele ano
# Para deflacionar: Valor_ano * FATORES_ACUMULADOS[ano]
# Exemplo: Valor_2023 * 1.0483 = Valor em reais de 2024
IPCA_DATA = {
    2018: 3.75,
    2019: 4.31,
    2020: 4.52,
    2021: 10.06,
    2022: 5.79,
    2023: 4.62,
    2024: 4.83
}

def calcular_fatores_acumulados(ano_base=2024):
    """
    Calcula dinamicamente os fatores de correção acumulados até ano_base.
    A inflação de um ano é a variação que ocorreu DURANTE aquele ano.
    Para trazer um valor do ano Y para ano_base, multiplica-se pela inflação de Y+1, Y+2, ..., ano_base.
    """
    fatores = {}
    anos_ordenados = sorted(IPCA_DATA.keys())
    
    for ano in anos_ordenados:
        if ano < ano_base:
            fator = 1.0
            # Multiplica pela inflação dos anos APÓS o ano origem até ano_base
            for y in range(ano + 1, ano_base + 1):
                if y in IPCA_DATA:
                    fator *= (1 + IPCA_DATA[y] / 100)
            fatores[ano] = fator
        elif ano == ano_base:
            # Valor já está na base, sem correção necessária
            fatores[ano] = 1.0
    
    return fatores

# ─── PRÉ-CÁLCULO DOS FATORES CONSOLIDADOS (base 2024) ──────────────────
FATORES_ACUMULADOS = calcular_fatores_acumulados(ano_base=2024)

def main():
    print("=" * 80)
    print("ATUALIZAÇÃO DO DATASET - ANÁLISE ECONÔMICA E CLIMÁTICA")
    print("=" * 80)
    
    # ─── 1. Carregar dados ───────────────────────────────────────────────
    print("\n[1/5] Carregando dados PAM_SIDRA...")
    df = pd.read_csv('data/interim/PAM_SIDRA/PAM_SIDRA.csv')
    print(f"      ✓ Carregado: {df.shape[0]:,} linhas × {df.shape[1]:,} colunas")
    
    # ─── 2. Exibir fatores consolidados de correção ──────────────────────
    print("\n[2/5] Fatores consolidados de correção para 2024:")
    print("      Ano → Fator (rigoroso, sem rounding em loop)")
    for ano in sorted(FATORES_ACUMULADOS.keys()):
        ipca_pct = IPCA_DATA[ano]
        print(f"        {ano}: IPCA {ipca_pct:>6.2f}% → Fator = {FATORES_ACUMULADOS[ano]:>10.6f}")
    
    # ─── 3. Criar diretórios de saída se não existirem ────────────────────
    print("\n[3/5] Verificando diretórios de saída...")
    os.makedirs('data/interim/PAM_SIDRA/', exist_ok=True)
    os.makedirs('data/outputs_teste/', exist_ok=True)
    print("      ✓ Diretórios confirmados/criados")
    
    # ─── 4. Criar novas colunas ──────────────────────────────────────────
    print("\n[4/5] Criando novas colunas...")
    
    # Usar nomes originais do CSV para evitar KeyError
    col_valor = 'Valor da produção (Mil Reais)'
    col_area = 'Área plantada ou destinada à colheita (Hectares)'
    col_rendimento = 'Rendimento médio da produção (Quilogramas por Hectare)'
    col_ano = 'ano'
    
    # Aplicar fator de correção consolidado
    df['fator_correcao_ipca'] = df[col_ano].map(FATORES_ACUMULADOS)
    df['valor_producao_ipca_mil_reais'] = df[col_valor] * df['fator_correcao_ipca']
    
    # Calcular valor por hectare
    df['valor_producao_ipca_mil_reais_ha'] = df['valor_producao_ipca_mil_reais'] / df[col_area]
    
    # Substituir infinitos e NaN por 0
    df['valor_producao_ipca_mil_reais_ha'] = df['valor_producao_ipca_mil_reais_ha'].replace([np.inf, -np.inf], np.nan)
    
    print("      ✓ Coluna 'valor_producao_ipca_mil_reais' criada (base 2024)")
    print("      ✓ Coluna 'valor_producao_ipca_mil_reais_ha' criada")
    
    # ─── 5. Matriz de Correlação de Pearson ───────────────────────────────
    print("\n[5/5] Calculando matriz de correlação de Pearson...")
    
    # Selecionar colunas para análise
    variaveis_analise = ['valor_producao_ipca_mil_reais_ha', col_rendimento]
    
    # Adicionar variáveis climáticas (primeiras 10 como exemplo)
    variaveis_climaticas = [col for col in df.columns if any(
        pattern in col for pattern in ['T2M', 'WS', 'PRECTOT', 'ALLSKY', 'RH2M', 'TOA']
    )][:10]  # Primeiras 10 variáveis climáticas
    
    variaveis_analise.extend(variaveis_climaticas)
    
    # Remover NaN
    df_corr = df[variaveis_analise].dropna()
    
    # Calcular matriz de correlação
    matriz_correlacao = df_corr.corr(method='pearson')
    
    print(f"      ✓ Matriz {matriz_correlacao.shape[0]} × {matriz_correlacao.shape[1]} calculada")
    
    # ─── 6. Salvar resultados ────────────────────────────────────────────
    print("\n[6/6] Salvando resultados...")
    
    # Salvar dataset atualizado
    df.to_csv('data/interim/PAM_SIDRA/PAM_SIDRA_ATUALIZADO.csv', index=False)
    print(f"      ✓ Dataset salvo: data/interim/PAM_SIDRA/PAM_SIDRA_ATUALIZADO.csv")
    
    # Salvar matriz de correlação
    matriz_correlacao.to_csv('data/outputs_teste/matriz_correlacao_pearson.csv')
    print(f"      ✓ Matriz de correlação: data/outputs_teste/matriz_correlacao_pearson.csv")
    
    # ─── 7. Exibir resumo ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESUMO DOS DADOS")
    print("=" * 80)
    
    print("\nÚltimas 5 linhas do dataset atualizado:")
    cols_exibir = ['Município', col_ano, col_area, col_valor, 'valor_producao_ipca_mil_reais', 'valor_producao_ipca_mil_reais_ha', col_rendimento]
    print(df[cols_exibir].tail())
    
    print(f"\nEstatísticas - valor_producao_ipca_mil_reais_ha (base 2024):")
    print(f"  Média:      R$ {df['valor_producao_ipca_mil_reais_ha'].mean():.2f} mil/ha")
    print(f"  Mediana:    R$ {df['valor_producao_ipca_mil_reais_ha'].median():.2f} mil/ha")
    print(f"  Mínimo:     R$ {df['valor_producao_ipca_mil_reais_ha'].min():.2f} mil/ha")
    print(f"  Máximo:     R$ {df['valor_producao_ipca_mil_reais_ha'].max():.2f} mil/ha")
    print(f"  Desvio Pad: R$ {df['valor_producao_ipca_mil_reais_ha'].std():.2f} mil/ha")
    
    print(f"\nEstatísticas - {col_rendimento}:")
    print(f"  Média:      {df[col_rendimento].mean():.2f} kg/ha")
    print(f"  Mediana:    {df[col_rendimento].median():.2f} kg/ha")
    print(f"  Mínimo:     {df[col_rendimento].min():.2f} kg/ha")
    print(f"  Máximo:     {df[col_rendimento].max():.2f} kg/ha")
    
    print("\nCorrelações com valor_producao_ipca_mil_reais_ha (Top 10):")
    print(matriz_correlacao['valor_producao_ipca_mil_reais_ha'].sort_values(ascending=False).head(10))
    
    print(f"\nCorrelações com {col_rendimento} (Top 10):")
    print(matriz_correlacao[col_rendimento].sort_values(ascending=False).head(10))
    
    print("\n" + "=" * 80)
    print("✓ ATUALIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)

if __name__ == "__main__":
    main()
