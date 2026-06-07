# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

import pandas as pd

# Carregar o CSV
df = pd.read_csv("data/processed/dataset_final.csv")

# Filtro fenológico mantenha as colunas que contém "dec" "ano" em suas nomenclaturas apenas as do do dec26 ao dec36 do ano1 e dec1 ao dec15 do ano2 as que não tem dec ou ano mantenha todas
colunas_para_manter = [
    col for col in df.columns
    if col == 'ano'  # mantém a coluna ano explicitamente
    or (
        "dec" in col and (
            ("ano1" in col and any(f"dec{n}" in col for n in range(26, 37)))
            or
            ("ano2" in col and any(f"dec{n}" in col for n in range(1, 16)))
        )
    )
    or ("ano" not in col and "dec" not in col)
]
df = df[colunas_para_manter]

# Salvar em Parquet com compressão Snappy
df.to_parquet("data/processed/dataset_final.parquet", engine="pyarrow", compression="snappy")
