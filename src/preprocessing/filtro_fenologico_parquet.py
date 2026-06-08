# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

import re
import pandas as pd

# Carregar o CSV
df = pd.read_csv("data/processed/dataset_final.csv")

def deve_manter(col):
    # Mantém colunas sem "dec" e sem "ano" (ex: id, cultura, etc.)
    if col == "ano":
        return True
    if "dec" not in col and "ano" not in col:
        return True

    # Extrai o número do dec e o ano da coluna (ex: "precipitacao_dec26_ano1")
    match = re.search(r"dec(\d+)_ano(\d+)", col)
    if not match:
        return False

    num_dec = int(match.group(1))
    num_ano = int(match.group(2))

    if num_ano == 1 and 26 <= num_dec <= 36:
        return True
    if num_ano == 2 and 1 <= num_dec <= 15:
        return True

    return False

colunas_para_manter = [col for col in df.columns if deve_manter(col)]
df = df[colunas_para_manter]

# Salvar em Parquet com compressão Snappy
df.to_parquet("data/processed/dataset_final.parquet", engine="pyarrow", compression="snappy")