# Copyright (C) 2026 Bruno Proença de Souza
# Licenciado sob GNU AGPL v3 - veja o arquivo LICENSE

import pandas as pd
import glob
import os
import gc
import shutil
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def func_agg(param):
    p = param.lower()
    if "max" in p:
        return "max"
    elif "min" in p:
        return "min"
    elif "range" in p:
        return lambda x: x.max() - x.min() if len(x) > 0 else 0
    elif "prec" in p:
        if "prob" in p:
            return "mean"
        elif "count" in p:
            return "sum"
        else:
            return "sum"
    else:
        return "mean"

def processar_e_salvar_direto(args):
    arquivo, pasta_saida = args
    COLS = ['ibge', 'municipio', 'lat', 'lon', 'ano', 'mes', 'decendio', 'parametro', 'valor']

    try:
        nome_base  = os.path.splitext(os.path.basename(arquivo))[0]
        pasta_temp = os.path.join(pasta_saida, 'temp')

        # Checar se já foi 100% processado via arquivo sentinela
        sentinela = os.path.join(pasta_temp, 'processados', f"{nome_base}.done")
        if os.path.exists(sentinela):
            return True, None  # pula leitura do CSV inteiro

        df = pd.read_csv(
            arquivo,
            usecols=['data', 'ibge', 'municipio', 'lat', 'lon', 'parametro', 'valor'],
            dtype={'ibge': 'int32', 'lat': 'float32', 'lon': 'float32',
                   'municipio': 'category', 'parametro': 'category', 'valor': 'float32'},
            na_values=['-999', '-999.0', -999]
        )

        datas          = pd.to_datetime(df['data'], format='%Y-%m-%d', cache=True)
        df['ano']      = datas.dt.year.astype('int16')
        df['mes']      = datas.dt.month.astype('int8')
        dia            = datas.dt.day.astype('int8')
        df['decendio'] = np.where(dia <= 10, 1, np.where(dia <= 20, 2, 3)).astype('int8')
        df.drop(columns='data', inplace=True)

        for ano, idx in df.groupby('ano', sort=False).groups.items():
            pasta_ano = os.path.join(pasta_saida, 'temp', str(ano))
            os.makedirs(pasta_ano, exist_ok=True)
            caminho = os.path.join(pasta_ano, f"{nome_base}.parquet")
            df.iloc[idx][COLS].to_parquet(caminho, index=False)

        # Marcar como concluído
        os.makedirs(os.path.join(pasta_temp, 'processados'), exist_ok=True)
        open(sentinela, 'w').close()

        return True, None

    except Exception as e:
        return False, f"{arquivo}: {e}"


def agregar_dados_decendio(df):
    """Agrega dados diários para decêndios — versão vetorizada em lotes por município"""
    parametros = df['parametro'].unique()

    params_sum   = [p for p in parametros if "prec" in p.lower() and "prob" not in p.lower()]
    params_count = [p for p in parametros if "count" in p.lower()]
    params_mean  = [p for p in parametros if "prec" in p.lower() and "prob" in p.lower()]
    params_max   = [p for p in parametros if "max" in p.lower()]
    params_min   = [p for p in parametros if "min" in p.lower()]
    params_range = [p for p in parametros if "range" in p.lower()]

    categorizados = set(params_sum + params_count + params_mean + params_max + params_min + params_range)
    params_mean   = params_mean + [p for p in parametros if p not in categorizados]

    chave = ['ibge', 'municipio', 'lat', 'lon', 'ano', 'mes', 'decendio']

    def agregar_lote(df_lote):
        from functools import reduce
        partes = []

        def pivot_agg(sub_df, params, func):
            if not params:
                return None
            sub = sub_df[sub_df['parametro'].isin(params)]
            if sub.empty:
                return None
            pt = sub.pivot_table(
                index=chave, columns='parametro', values='valor', aggfunc=func
            )
            pt.columns.name = None
            return pt.reset_index()

        if params_range:
            r = pivot_agg(df_lote, params_range, lambda x: x.max() - x.min())
            if r is not None:
                partes.append(r)

        for params, func in [
            (params_sum + params_count, 'sum'),
            (params_mean,               'mean'),
            (params_max,                'max'),
            (params_min,                'min'),
        ]:
            r = pivot_agg(df_lote, params, func)
            if r is not None:
                partes.append(r)

        if not partes:
            return pd.DataFrame()

        return reduce(lambda a, b: pd.merge(a, b, on=chave, how='outer'), partes)

    municipios = df['municipio'].unique()
    LOTE_SIZE  = 50
    resultados = []

    for i in range(0, len(municipios), LOTE_SIZE):
        lote_munic = municipios[i : i + LOTE_SIZE]
        df_lote    = df[df['municipio'].isin(lote_munic)]
        resultado  = agregar_lote(df_lote)
        if not resultado.empty:
            resultados.append(resultado)
        del df_lote
        gc.collect()

    if not resultados:
        return pd.DataFrame()

    return pd.concat(resultados, ignore_index=True)


def transformar_para_safra(df_total, pasta_saida):
    print("\n=== TRANSFORMANDO PARA FORMATO SAFRA ===")
    df_total = df_total.rename(columns={'ibge': 'codigo_ibge'})

    colunas_nao_climaticas = ['ano', 'municipio', 'codigo_ibge', 'mes', 'lat', 'lon', 'decendio']
    variaveis_climaticas   = [col for col in df_total.columns if col not in colunas_nao_climaticas]
    print(f"Variáveis climáticas: {variaveis_climaticas}")

    # ← CORREÇÃO: criar coluna composta mes_decendio (1 a 36)
    df_total['mes_dec'] = (df_total['mes'] - 1) * 3 + df_total['decendio']
    # mes_dec vai de 1 (jan/dec1) a 36 (dez/dec3)

    dfs_transformados = []
    for variavel in variaveis_climaticas:
        df_pivot = df_total.pivot_table(
            index=['ano', 'municipio', 'codigo_ibge'],
            columns='mes_dec',          # ← era 'decendio'
            values=variavel,
            aggfunc='mean'
        )
        df_pivot.columns = [f"{variavel}_dec{int(col)}" for col in df_pivot.columns]
        # ex: tmax_dec01 (jan/dec1) ... tmax_dec36 (dez/dec3)
        dfs_transformados.append(df_pivot)

    clima_transformado = pd.concat(dfs_transformados, axis=1).reset_index()
    print(f"Formato largo: {clima_transformado.shape}")

    anos_disponiveis = sorted(clima_transformado['ano'].unique())
    dados_safra = []

    for municipio in clima_transformado['municipio'].unique():
        df_mun = clima_transformado[clima_transformado['municipio'] == municipio]
        for ano_safra in anos_disponiveis[1:]:
            d_ant = df_mun[df_mun['ano'] == ano_safra - 1]
            d_atu = df_mun[df_mun['ano'] == ano_safra]
            if len(d_ant) > 0 and len(d_atu) > 0:
                registro = {
                    'ano_safra':   ano_safra,
                    'municipio':   municipio,
                    'codigo_ibge': d_atu['codigo_ibge'].iloc[0]
                }
                for col in d_ant.columns:
                    if col not in ['ano', 'municipio', 'codigo_ibge']:
                        registro[f"{col}_ano1"] = d_ant[col].iloc[0]
                for col in d_atu.columns:
                    if col not in ['ano', 'municipio', 'codigo_ibge']:
                        registro[f"{col}_ano2"] = d_atu[col].iloc[0]
                dados_safra.append(registro)

    df_safra = pd.DataFrame(dados_safra)
    arquivo_safra = f"{pasta_saida}/NASA_POWER_decendio_safra.csv"
    df_safra.to_csv(arquivo_safra, index=False, float_format="%.4f")
    print(f"Salvo: {arquivo_safra} ({df_safra.shape[0]} registros, {df_safra.shape[1]} colunas)")
    gc.collect()


def main():
    print("=== PROCESSAMENTO OTIMIZADO DE CSVs PARA DECÊNDIOS ===")

    PASTA_ENTRADA  = "./data/raw/NASA_POWER"
    PASTA_SAIDA    = "./data/interim/NASA_POWER"
    # 
    N_WORKERS      = max(1, os.cpu_count())
    LOTE_PARQUETS  = 500

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    arquivos = glob.glob(f"{PASTA_ENTRADA}/*.csv")
    print(f"Encontrados {len(arquivos)} arquivos CSV")
    if not arquivos:
        return

    # ── ETAPA 1: cada worker lê e salva parquet por ano ──────────────────
    print("\n=== ETAPA 1: LENDO E PARTICIONANDO ARQUIVOS ===")
    args = [(arq, PASTA_SAIDA) for arq in arquivos]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        resultados = executor.map(processar_e_salvar_direto, args, chunksize=50)
        erros = 0
        for ok, erro in tqdm(resultados, total=len(arquivos), desc="Lendo arquivos"):
            if not ok:
                print(f"Erro: {erro}")
                erros += 1

    print(f"Etapa 1 concluída. Erros: {erros}/{len(arquivos)}")

# ── ETAPA 2: agregar por ano e salvar decendio_YYYY.csv ──────────────
    print("\n=== ETAPA 2: AGREGANDO POR ANO ===")
    pasta_temp = os.path.join(PASTA_SAIDA, 'temp')
    anos = sorted([int(d) for d in os.listdir(pasta_temp) if d.isdigit()])
    print(f"Anos encontrados: {anos}")

    for ano in tqdm(anos, desc="Agregando anos"):
        pasta_ano = os.path.join(pasta_temp, str(ano))
        parquets  = glob.glob(f"{pasta_ano}/*.parquet")
        print(f"\nAno {ano}: {len(parquets)} parquets encontrados")

        if not parquets:
            continue

        resultados_ano = []
        total_lotes = (len(parquets) + LOTE_PARQUETS - 1) // LOTE_PARQUETS

        for i in range(0, len(parquets), LOTE_PARQUETS):
            lote = parquets[i : i + LOTE_PARQUETS]
            print(f"  Lote {i//LOTE_PARQUETS + 1}/{total_lotes} — lendo {len(lote)} parquets...", flush=True)

            # Ler um por um em vez de gerador (mais seguro para diagnóstico)
            dfs = []
            for j, p in enumerate(lote):
                try:
                    dfs.append(pd.read_parquet(p))
                except Exception as e:
                    print(f"    Erro ao ler {p}: {e}")
                    continue

            print(f"  Lidos {len(dfs)} parquets, concatenando...", flush=True)
            df_lote = pd.concat(dfs, ignore_index=True)
            del dfs
            gc.collect()

            print(f"  Concat OK — shape: {df_lote.shape}, agregando...", flush=True)
            resultado = agregar_dados_decendio(df_lote)
            print(f"  Agregação OK — shape: {resultado.shape}", flush=True)
            resultados_ano.append(resultado)
            del df_lote
            gc.collect()

        print(f"  Todos os lotes prontos, concat final...", flush=True)
        df_agregado = pd.concat(resultados_ano, ignore_index=True)
        del resultados_ano
        gc.collect()

        arquivo_ano = f"{PASTA_SAIDA}/decendio_{ano}.csv"
        df_agregado.to_csv(arquivo_ano, index=False, float_format="%.4f")
        print(f"Salvo: {arquivo_ano} ({len(df_agregado)} registros)")
        del df_agregado
        gc.collect()

    # ── ETAPA 3: consolidar anuais + safra ────────────────────────────────
    print("\n=== ETAPA 3: CONSOLIDANDO E SAFRA ===")
    arquivos_anuais = sorted(glob.glob(f"{PASTA_SAIDA}/decendio_[0-9]*.csv"))

    if arquivos_anuais:
        df_total = pd.concat(
            (pd.read_csv(f) for f in tqdm(arquivos_anuais, desc="Lendo anuais")),
            ignore_index=True
        )
        df_total.to_csv(f"{PASTA_SAIDA}/decendio_total.csv", index=False, float_format="%.4f")
        print(f"Salvo: decendio_total.csv ({len(df_total)} registros)")
        transformar_para_safra(df_total, PASTA_SAIDA)
        del df_total
        gc.collect()

    # ── Limpeza dos temporários ───────────────────────────────────────────
    shutil.rmtree(pasta_temp, ignore_errors=True)

    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    for arquivo in glob.glob(f"{PASTA_SAIDA}/*.csv"):
        tamanho = os.path.getsize(arquivo) / (1024 * 1024)
        print(f"  - {os.path.basename(arquivo)} ({tamanho:.1f} MB)")


if __name__ == "__main__":
    main()