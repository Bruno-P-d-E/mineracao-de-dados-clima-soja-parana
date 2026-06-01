from decimal import Decimal

import numpy as np
import pandas as pd
from scipy import stats


def calcular_pearson(df_temp, col_x, col_y):
    """Retorna correlacao de Pearson e p-valor para duas colunas numericas."""
    dados = df_temp[[col_x, col_y]].dropna()
    n = len(dados)
    if n <= 2:
        return np.nan, np.nan
    x = dados[col_x]
    y = dados[col_y]
    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan, np.nan
    try:
        corr, p_valor = stats.pearsonr(x, y)
    except Exception:
        return np.nan, np.nan
    return corr, p_valor


def formatar_pvalor(p_valor, decimais_pvalor=3, notacao_cientifica=False):
    """Formata p-valor sem esconder valores pequenos como zero."""
    if notacao_cientifica:
        return f"{p_valor:.{decimais_pvalor}e}".replace('.', ',')

    p_fmt = f"{p_valor:.{decimais_pvalor}f}"
    if p_valor > 0 and Decimal(p_fmt) == 0:
        p_fmt = format(Decimal(str(p_valor)), 'f')
    return p_fmt.replace('.', ',')


def formatar_correlacao_pvalor(
    corr,
    p_valor,
    decimais_corr=2,
    decimais_pvalor=3,
    usar_limite_pvalor=True,
    notacao_cientifica_pvalor=False,
):
    """Formata correlacao e p-valor no padrao pt-BR."""
    if pd.isna(corr):
        return "-"

    corr_fmt = f"{corr:.{decimais_corr}f}".replace('.', ',')
    if pd.isna(p_valor):
        return f"rho= {corr_fmt}, p−value= -"

    limite = 10 ** (-decimais_pvalor)
    if usar_limite_pvalor and p_valor < limite:
        limite_fmt = f"{limite:.{decimais_pvalor}f}".replace('.', ',')
        return f"rho= {corr_fmt}, p−value= < {limite_fmt}"

    p_fmt = formatar_pvalor(
        p_valor,
        decimais_pvalor,
        notacao_cientifica=notacao_cientifica_pvalor,
    )
    return f"rho= {corr_fmt}, p−value= {p_fmt}"
