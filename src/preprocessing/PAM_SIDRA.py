# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

import pandas as pd
import os
import re

def extrair_nome_variavel(excel, nome_pagina):
    """Lê a célula A2 e extrai o nome após 'Variável - '"""
    df_raw = pd.read_excel(excel, sheet_name=nome_pagina, header=None, nrows=3)
    celula_a2 = str(df_raw.iloc[1, 0]).strip()
    match = re.search(r'[Vv]ariável\s*-\s*(.+)', celula_a2)
    if match:
        return match.group(1).strip()
    print(f"  Aviso: padrão 'Variável - ' não encontrado em A2 de '{nome_pagina}'. Conteúdo: '{celula_a2}'")
    return celula_a2


def converter_excel_para_csv(caminho_arquivo, caminho_saida='./data/interim/PAM_SIDRA/PAM_SIDRA.csv'):
    excel = pd.ExcelFile(caminho_arquivo)
    dfs_transformados = []

    for nome_pagina in excel.sheet_names:
        if not nome_pagina.strip().lower().startswith('tabela'):
            print(f"Pulando aba não-dados: {nome_pagina}")
            continue

        nome_variavel = extrair_nome_variavel(excel, nome_pagina)
        print(f"Processando: '{nome_pagina}' → variável: '{nome_variavel}'")

        df = pd.read_excel(excel, sheet_name=nome_pagina, header=[2, 3])

        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                top, sub = str(col[0]).strip(), str(col[1]).strip()
                new_cols.append(top if 'Unnamed' in sub else sub)
            else:
                new_cols.append(str(col).strip())
        df.columns = new_cols

        colunas_anos = [str(ano) for ano in range(2018, 2025)]
        anos_presentes = [a for a in colunas_anos if a in df.columns]

        if not anos_presentes:
            print(f"  Aviso: Nenhum ano encontrado. Pulando.")
            continue

        df.rename(columns={df.columns[0]: 'Cód.', df.columns[1]: 'Município'}, inplace=True)
        df = df[['Cód.', 'Município'] + anos_presentes]
        df = df[df['Município'].notna()]

        df_longo = pd.melt(
            df,
            id_vars=['Cód.', 'Município'],
            value_vars=anos_presentes,
            var_name='Ano',
            value_name=nome_variavel
        )

        dfs_transformados.append(df_longo)

    if not dfs_transformados:
        print("Erro: Nenhuma página válida encontrada.")
        return None

    df_final = dfs_transformados[0]
    for df in dfs_transformados[1:]:
        df_final = pd.merge(df_final, df, on=['Cód.', 'Município', 'Ano'], how='outer')

    df_final['Ano'] = df_final['Ano'].astype(int)
    df_final = df_final.sort_values(by=['Cód.', 'Município', 'Ano']).reset_index(drop=True)
    df_final = df_final.rename(columns={'Cód.': 'cod_ibge'})

    # ─── Limpeza: substitui "-" (e variantes com espaços) por 0 ───────────────
    colunas_variaveis = [c for c in df_final.columns if c not in ['cod_ibge', 'Município', 'Ano']]
    df_final[colunas_variaveis] = (
        df_final[colunas_variaveis]
        .replace(r'^\s*-\s*$', 0, regex=True)  # traço isolado (ex: "-", " - ")
        .apply(pd.to_numeric, errors='coerce')  # converte tudo para número
        .fillna(0)                              # NaN residuais → 0
    )
    # ──────────────────────────────────────────────────────────────────────────

    df_final.to_csv(caminho_saida, index=False, encoding='utf-8-sig')
    print(f"\nCSV salvo: {caminho_saida}")
    print(f"{len(df_final)} linhas × {len(df_final.columns)} colunas")
    print(f"Variáveis encontradas: {colunas_variaveis}")

    return df_final


if __name__ == "__main__":
    caminho_do_arquivo = "./data/raw/PAM_SIDRA/tabela5457.xlsx"

    if not os.path.exists(caminho_do_arquivo):
        print(f"Erro: Arquivo não encontrado: {caminho_do_arquivo}")
    else:
        df_resultado = converter_excel_para_csv(caminho_do_arquivo)
        if df_resultado is not None:
            print("\nPrimeiras linhas:")
            print(df_resultado.head())