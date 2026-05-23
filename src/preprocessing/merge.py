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
import numpy as np
from pathlib import Path

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
pam["cod_ibge"]      = pam["cod_ibge"].str.strip().str.zfill(7)
nasa["codigo_ibge"]  = nasa["codigo_ibge"].str.strip().str.zfill(7)
ibge["cod_ibge"]     = ibge["cod_ibge"].str.strip().str.zfill(7)

# ── Normalização do ano ───────────────────────────────────────────────────────
pam["Ano"]          = pam["Ano"].astype(int)
nasa["ano_safra"]   = nasa["ano_safra"].astype(int)

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
                                     "ano_safra":   "Ano"})

# Remove colunas redundantes da NASA que já temos no PAM/IBGE
cols_to_drop_nasa = ["municipio"]   # nome já vem do PAM
nasa_renamed = nasa_renamed.drop(
    columns=[c for c in cols_to_drop_nasa if c in nasa_renamed.columns]
)

dataset = pam_ibge.merge(
    nasa_renamed,
    on=["cod_ibge", "Ano"],
    how="left",
    validate="1:1",   # 1 registro climático por município-ano
)

n_sem_clima = dataset.iloc[:, nasa_renamed.shape[1]:].isna().all(axis=1).sum()
if n_sem_clima:
    print(f"  ⚠  {n_sem_clima} registros sem dados climáticos NASA")

# ── Reordenação de colunas ───────────────────────────────────────────────────
# Identificadores primeiro, depois geo, depois PAM (targets), depois clima
id_cols      = ["cod_ibge", "Município", "Ano"]
geo_cols     = ["cod_meso", "mesorregiao", "latitude", "longitude"]
pam_targets  = [c for c in pam.columns if c not in id_cols]
nasa_cols    = [c for c in nasa_renamed.columns if c not in ["cod_ibge", "Ano"]]

# Garante que só colunas que existem entram na lista final
final_cols = (
    [c for c in id_cols    if c in dataset.columns] +
    [c for c in geo_cols   if c in dataset.columns] +
    [c for c in pam_targets if c in dataset.columns] +
    [c for c in nasa_cols  if c in dataset.columns]
)

dataset = dataset[final_cols]

# ── Relatório ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Dataset final: {dataset.shape[0]:,} linhas x {dataset.shape[1]} colunas")
print(f"  Municípios únicos : {dataset['cod_ibge'].nunique():,}")
print(f"  Anos cobertos     : {sorted(dataset['Ano'].unique())}")
print(f"  Colunas climáticas: {len(nasa_cols)}")
print(f"  Valores nulos     : {dataset.isna().sum().sum():,}")
print(f"{'='*55}")

# ── Salvar ───────────────────────────────────────────────────────────────────
dataset.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"\n✓ Dataset salvo em: {OUT_PATH}")

# ── Amostra ──────────────────────────────────────────────────────────────────
print("\nPrimeiras colunas (amostra):")
print(dataset[final_cols[:10]].head(3).to_string())