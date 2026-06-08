# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

"""
Script de construção do dataset unificado
=========================================
Fontes:
  - PAM_SIDRA   : dados de produção agrícola (IBGE/SIDRA)
  - NASA_POWER  : variáveis climáticas por decêndio de safra
  - IBGE        : dados geográficos dos municípios (PR)

Chaves de join:
  - PAM_SIDRA ↔ IBGE       : cod_ibge
  - (PAM+IBGE) ↔ NASA_POWER : cod_ibge == codigo_ibge  AND  Ano == ano_safra
"""

import pandas as pd
import re
from pathlib import Path
from correcao_ipca import aplicar_correcao_ipca
from padronizar_colunas import padronizar_colunas


def deve_manter_coluna_fenologica(coluna):
    """Mantem somente o periodo fenologico da soja nas colunas climaticas."""
    if coluna == "ano":
        return True
    if "dec" not in coluna and "ano" not in coluna:
        return True

    match = re.search(r"dec(\d+)_ano(\d+)", coluna)
    if not match:
        return False

    num_dec = int(match.group(1))
    num_ano = int(match.group(2))

    if num_ano == 1 and 26 <= num_dec <= 36:
        return True
    if num_ano == 2 and 1 <= num_dec <= 15:
        return True

    return False


def aplicar_filtro_fenologico(df):
    """Filtra o dataset para o ciclo da soja: set-dez do ano 1 e jan-mai do ano 2."""
    colunas_para_manter = [col for col in df.columns if deve_manter_coluna_fenologica(col)]
    return df[colunas_para_manter].copy()

# ── Caminhos ────────────────────────────────────────────────────────────────
PAM_PATH  = Path("data/interim/PAM_SIDRA/PAM_SIDRA.csv")
NASA_PATH = Path("data/interim/NASA_POWER/NASA_POWER_decendio_safra.csv")
IBGE_PATH = Path("data/raw/IBGE/municipios_pr.csv")
OUT_PATH  = Path("data/processed/dataset_final.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Leitura ──────────────────────────────────────────────────────────────────
print("Lendo arquivos...")
pam  = pd.read_csv(PAM_PATH,  dtype={"cod_ibge": str})
nasa = pd.read_csv(NASA_PATH, dtype={"codigo_ibge": str})
ibge = pd.read_csv(IBGE_PATH, dtype={"cod_ibge": str})

print(f"  PAM_SIDRA   : {pam.shape[0]:,} linhas  x {pam.shape[1]} colunas")
print(f"  NASA_POWER  : {nasa.shape[0]:,} linhas  x {nasa.shape[1]} colunas")
print(f"  IBGE        : {ibge.shape[0]:,} linhas  x {ibge.shape[1]} colunas")

# ── Normalização dos códigos IBGE ─────────────────────────────────────────────
# Garante 7 dígitos para comparação segura
print("\n[PADRONIZACAO] Normalizando colunas PAM/SIDRA para snake_case...")
pam = padronizar_colunas(pam)

pam["cod_ibge"]      = pam["cod_ibge"].str.strip().str.zfill(7)
nasa["codigo_ibge"]  = nasa["codigo_ibge"].str.strip().str.zfill(7)
ibge["cod_ibge"]     = ibge["cod_ibge"].str.strip().str.zfill(7)

# ── Normalização do ano ───────────────────────────────────────────────────────
pam["ano"]          = pam["ano"].astype(int)
nasa["ano_safra"]   = nasa["ano_safra"].astype(int)

# ── Aplicar Correção IPCA (deflaciona valores nominais para base 2024) ────────
print("\n[CORREÇÃO ECONÔMICA] Aplicando deflação IPCA (base 2024)...")
pam = aplicar_correcao_ipca(pam, verbose=False)  # verbose=False para não poluir output

# ── Merge 1: PAM_SIDRA + IBGE (enriquece com lat/lon e mesorregião) ──────────
print("\nMerge 1: PAM_SIDRA ↔ IBGE (cod_ibge)...")
pam_ibge = pam.merge(
    ibge[["cod_ibge", "cod_meso", "mesorregiao", "latitude", "longitude"]],
    on="cod_ibge",
    how="left",
    validate="m:1",   # muitos anos por município
)

n_sem_geo = pam_ibge["latitude"].isna().sum()
if n_sem_geo:
    print(f"  ⚠  {n_sem_geo} registros PAM sem correspondência no IBGE")

# ── Merge 2: PAM+IBGE + NASA_POWER (cod_ibge + ano) ──────────────────────────
print("Merge 2: (PAM+IBGE) ↔ NASA_POWER (cod_ibge + ano_safra)...")

# Renomeia chave nasa para igualar o nome
nasa_renamed = nasa.rename(columns={"codigo_ibge": "cod_ibge",
                                     "ano_safra":   "ano"})

# Remove colunas redundantes da NASA que já temos no PAM/IBGE
cols_to_drop_nasa = ["municipio"]   # nome já vem do PAM
nasa_renamed = nasa_renamed.drop(
    columns=[c for c in cols_to_drop_nasa if c in nasa_renamed.columns]
)

dataset = pam_ibge.merge(
    nasa_renamed,
    on=["cod_ibge", "ano"],
    how="left",
    validate="1:1",   # 1 registro climático por município-ano
)

n_sem_clima = dataset.iloc[:, nasa_renamed.shape[1]:].isna().all(axis=1).sum()
if n_sem_clima:
    print(f"  ⚠  {n_sem_clima} registros sem dados climáticos NASA")

# ── Reordenação de colunas ───────────────────────────────────────────────────
# Identificadores primeiro, depois geo, depois PAM (targets), depois clima
id_cols      = ["cod_ibge", "municipio", "ano"]
geo_cols     = ["cod_meso", "mesorregiao", "latitude", "longitude"]
pam_targets  = [c for c in pam.columns if c not in id_cols]
nasa_cols    = [c for c in nasa_renamed.columns if c not in ["cod_ibge", "ano"]]

# Garante que só colunas que existem entram na lista final
final_cols = (
    [c for c in id_cols    if c in dataset.columns] +
    [c for c in geo_cols   if c in dataset.columns] +
    [c for c in pam_targets if c in dataset.columns] +
    [c for c in nasa_cols  if c in dataset.columns]
)

dataset = dataset[final_cols]

print("\n[FILTRO FENOLOGICO] Mantendo ciclo da soja (Ano 1: dec26-36 | Ano 2: dec1-15)...")
colunas_antes_filtro = dataset.shape[1]
dataset = aplicar_filtro_fenologico(dataset)
colunas_removidas = colunas_antes_filtro - dataset.shape[1]
colunas_climaticas_filtradas = [c for c in dataset.columns if re.search(r"_dec\d+_ano\d+", c)]
print(f"  Colunas removidas pelo filtro fenologico: {colunas_removidas:,}")

# ── Relatório ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Dataset final: {dataset.shape[0]:,} linhas x {dataset.shape[1]} colunas")
print(f"  Municípios únicos : {dataset['cod_ibge'].nunique():,}")
print(f"  Anos cobertos     : {sorted(dataset['ano'].unique())}")
print(f"  Colunas climáticas: {len(colunas_climaticas_filtradas)}")
print(f"  Valores nulos     : {dataset.isna().sum().sum():,}")
print(f"{'='*55}")

# ── Salvar ───────────────────────────────────────────────────────────────────
dataset.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"\n✓ Dataset salvo em: {OUT_PATH}")

# ── Salvar também em Parquet (otimizado para o Dashboard) ──────────────────────
PARQUET_PATH = OUT_PATH.parent / "dataset_final.parquet"
dataset.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
print(f"✓ Dataset salvo em (Parquet): {PARQUET_PATH}")

# ── Amostra ──────────────────────────────────────────────────────────────────
print("\nPrimeiras colunas (amostra):")
print(dataset[final_cols[:10]].head(3).to_string())
