# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

"""
Módulo de Correção Econômica por IPCA
======================================
Realiza deflação de valores nominais para base real (2024).

Uso:
    from src.preprocessing.correcao_ipca import aplicar_correcao_ipca
    
    df_corrigido = aplicar_correcao_ipca(df, ano_base=2024)
"""

import pandas as pd
import numpy as np

# ─── Índices de Inflação IPCA ────────────────────────────────────────────────
IPCA_DATA = {
    2018: 3.75,   # Inflação durante 2018
    2019: 4.31,   # Inflação durante 2019
    2020: 4.52,   # ...
    2021: 10.06,
    2022: 5.79,
    2023: 4.62,
    2024: 4.83    # Inflação durante 2024
}


def calcular_fatores_acumulados(ano_base=2024):
    """
    Calcula dinamicamente os fatores de correção acumulados até ano_base.
    
    A inflação de um ano é a variação de preço que ocorreu DURANTE aquele ano.
    Para trazer um valor do ano Y para ano_base, multiplica-se pela inflação 
    de Y+1, Y+2, ..., ano_base.
    
    Exemplo: 
        Valor_2023 → 2024: Valor_2023 * (1 + IPCA_2024/100)
    
    Parameters
    ----------
    ano_base : int, default 2024
        Ano base para deflação
    
    Returns
    -------
    dict
        {ano: fator_acumulado} para cada ano em IPCA_DATA
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


def aplicar_correcao_ipca(df, 
                         col_valor='valor_producao_mil_reais',
                         col_area='area_plantada_ha',
                         col_ano='ano',
                         ano_base=2024,
                         verbose=True):
    """
    Aplica correção IPCA a um DataFrame de produção agrícola.
    
    Cria 3 novas colunas:
    - fator_correcao_ipca: fator aplicado ao ano
    - valor_producao_ipca_mil_reais: valor nominal corrigido para a base
    - valor_producao_ipca_mil_reais_ha: valor corrigido dividido pela área plantada
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados de produção (ex: PAM_SIDRA)
    
    col_valor : str, default 'Valor da produção (Mil Reais)'
        Nome da coluna de valor nominal
    
    col_area : str, default 'Área plantada ou destinada à colheita (Hectares)'
        Nome da coluna de área plantada
    
    col_ano : str, default 'Ano'
        Nome da coluna de ano
    
    ano_base : int, default 2024
        Ano base para deflação
    
    verbose : bool, default True
        Se True, exibe resumo das transformações
    
    Returns
    -------
    pd.DataFrame
        DataFrame com as 3 novas colunas adicionadas
    
    Examples
    --------
    >>> import pandas as pd
    >>> from src.preprocessing.correcao_ipca import aplicar_correcao_ipca
    >>> 
    >>> df = pd.read_csv('data/interim/PAM_SIDRA/PAM_SIDRA.csv')
    >>> df_corrigido = aplicar_correcao_ipca(df)
    >>> df_corrigido[['Município', 'Ano', 'valor_producao_ipca_mil_reais', 'valor_producao_ipca_mil_reais_ha']].head()
    """
    
    # Validar colunas obrigatórias
    colunas_obrigatorias = [col_valor, col_area, col_ano]
    colunas_faltantes = [c for c in colunas_obrigatorias if c not in df.columns]
    if colunas_faltantes:
        raise ValueError(f"Colunas obrigatórias não encontradas: {colunas_faltantes}")
    
    # Copiar DataFrame para não modificar original
    df = df.copy()
    
    # Calcular fatores consolidados
    fatores = calcular_fatores_acumulados(ano_base=ano_base)
    
    # Aplicar correção
    df['fator_correcao_ipca'] = df[col_ano].map(fatores)
    df['valor_producao_ipca_mil_reais'] = df[col_valor] * df['fator_correcao_ipca']
    
    # Calcular valor por hectare
    df['valor_producao_ipca_mil_reais_ha'] = df['valor_producao_ipca_mil_reais'] / df[col_area]
    
    # Substituir infinitos e NaN
    df['valor_producao_ipca_mil_reais_ha'] = df['valor_producao_ipca_mil_reais_ha'].replace([np.inf, -np.inf], np.nan)
    
    if verbose:
        print("\n" + "─" * 70)
        print(f"✓ CORREÇÃO IPCA APLICADA (base {ano_base})")
        print("─" * 70)
        print("\nFatores de Correção:")
        for ano in sorted(fatores.keys()):
            ipca = IPCA_DATA[ano]
            print(f"  {ano}: IPCA {ipca:>6.2f}% → Fator = {fatores[ano]:>10.6f}")
        
        print(f"\nEstatísticas - valor_producao_ipca_mil_reais_ha (base {ano_base}):")
        print(f"  Média:      R$ {df['valor_producao_ipca_mil_reais_ha'].mean():.2f} mil/ha")
        print(f"  Mediana:    R$ {df['valor_producao_ipca_mil_reais_ha'].median():.2f} mil/ha")
        print(f"  Range:      [R$ {df['valor_producao_ipca_mil_reais_ha'].min():.2f}, R$ {df['valor_producao_ipca_mil_reais_ha'].max():.2f}] mil/ha")
        print("─" * 70 + "\n")
    
    return df


if __name__ == "__main__":
    # Teste rápido
    df_teste = pd.read_csv('data/interim/PAM_SIDRA/PAM_SIDRA.csv')
    df_corrigido = aplicar_correcao_ipca(df_teste)
    print(df_corrigido[['Município', 'Ano', 'valor_producao_ipca_mil_reais', 'valor_producao_ipca_mil_reais_ha']].head(10))
