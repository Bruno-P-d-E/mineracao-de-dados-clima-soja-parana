import pandas as pd
import glob
import os
import gc
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def decendio(data_str):
    """Converte data YYYYMMDD para decêndio"""
    dia = int(str(data_str)[-2:])
    if dia <= 10:
        return 1
    elif dia <= 20:
        return 2
    else:
        return 3

def func_agg(param):
    """Define função de agregação baseada no nome do parâmetro"""
    p = param.lower()
    
    if "max" in p:
        return "max"
    elif "min" in p:
        return "min"
    elif "range" in p:
        return lambda x: x.max() - x.min() if len(x) > 0 else 0
    
    # ---- Tratamento especial para precipitação ----
    elif "prec" in p:
        if "prob" in p:     # Probabilidade → média
            return "mean"
        elif "count" in p:  # Contagem → soma
            return "sum"
        else:               # Totais (chuva, neve, etc.) → soma
            return "sum"
    
    else:
        return "mean"

def processar_chunk_arquivos(arquivos_chunk, chunk_size=1000):
    """Processa um chunk de arquivos e retorna dados agregados por ano"""
    dados_por_ano = defaultdict(list)
    
    for arquivo in arquivos_chunk:
        try:
            # Ler arquivo em chunks menores se for muito grande
            chunk_list = []
            for chunk in pd.read_csv(arquivo, chunksize=chunk_size):
                # Converter data para formato adequado
                chunk['ano'] = chunk['data'].str[:4].astype(int)
                chunk['mes'] = chunk['data'].str[5:7].astype(int)
                chunk['dia'] = chunk['data'].str[8:10].astype(int)
                
                # Calcular decêndio
                chunk['decendio'] = (chunk['mes'] - 1) * 3 + chunk['dia'].apply(lambda x: 1 if x <= 10 else (2 if x <= 20 else 3))
                
                # Garantir que valor seja float
                chunk['valor'] = pd.to_numeric(chunk['valor'], errors='coerce')
                # Substituir -999 por NaN
                chunk['valor'] = chunk['valor'].replace(-999, np.nan)
                chunk['valor'] = chunk['valor'].replace(-999.0, np.nan)
                
                chunk_list.append(chunk[['ibge', 'municipio', 'lat', 'lon', 'ano', 'mes', 'decendio', 'parametro', 'valor']])
            
            if chunk_list:
                df_arquivo = pd.concat(chunk_list, ignore_index=True)

                # Agrupar por ano
                for ano in df_arquivo['ano'].unique():
                    dados_ano = df_arquivo[df_arquivo['ano'] == ano].copy()
                    dados_por_ano[ano].append(dados_ano)

                # Limpar memória
                del df_arquivo, chunk_list
                gc.collect()

        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
            continue

    return dados_por_ano

def agregar_dados_decendio(df):
    """Agrega dados diários para decêndios"""
    # Obter todos os parâmetros únicos
    parametros = df['parametro'].unique()   
    parametros = [p for p in parametros]
    # Criar dicionário de agregação
    agg_dict = {}
    for param in parametros:
        agg_dict[param] = func_agg(param)

    # Preparar dados para pivot
    dados_agregados = []

    # Agrupar por localização e período
    grupos = df.groupby(['ibge', 'municipio', 'lat', 'lon', 'ano', 'mes', 'decendio'])

    for (ibge, municipio, lat, lon, ano, mes, decendio), grupo in grupos:
        linha = {
            'ibge': ibge,
            'municipio': municipio, 
            'lat': lat,
            'lon': lon,
            'ano': ano,
            'mes': mes,
            'decendio': decendio
        }

        # Agregar cada parâmetro
        for param in parametros:
            dados_param = grupo[grupo['parametro'] == param]['valor'].dropna()
            if not dados_param.empty:
                func_agregacao = agg_dict[param]
                if callable(func_agregacao):
                    try:
                        linha[param] = func_agregacao(dados_param)
                    except:
                        linha[param] = dados_param.mean()
                elif func_agregacao == 'max':
                    linha[param] = dados_param.max()
                elif func_agregacao == 'min':
                    linha[param] = dados_param.min()
                elif func_agregacao == 'sum':
                    linha[param] = dados_param.sum()
                else:  # mean
                    linha[param] = dados_param.mean()
            else:
                linha[param] = np.nan

        dados_agregados.append(linha)

    return pd.DataFrame(dados_agregados)

def transformar_para_safra(df_total, pasta_saida):
    """Transforma dados diários agregados em formato safra de dois anos"""
    print("\n=== TRANSFORMANDO PARA FORMATO SAFRA ===")
    
    df_total = df_total.rename(columns={'ibge': 'codigo_ibge'})
    
    # 1. Formato largo: uma coluna por variável+decêndio
    colunas_nao_climaticas = ['ano', 'municipio', 'codigo_ibge', 'mes', 'lat', 'lon', 'decendio']
    variaveis_climaticas = [col for col in df_total.columns if col not in colunas_nao_climaticas]
    print(f"Variáveis climáticas: {variaveis_climaticas}")

    dfs_transformados = []
    for variavel in variaveis_climaticas:
        df_pivot = df_total.pivot_table(
            index=['ano', 'municipio', 'codigo_ibge'],
            columns='decendio',
            values=variavel,
            aggfunc='mean'
        )
        df_pivot.columns = [f"{variavel}_dec{int(col)}" for col in df_pivot.columns]
        dfs_transformados.append(df_pivot)

    clima_transformado = pd.concat(dfs_transformados, axis=1).reset_index()
    print(f"Formato largo: {clima_transformado.shape}")

    # 2. Organizar por safra de dois anos
    anos_disponiveis = sorted(clima_transformado['ano'].unique())
    dados_safra = []

    for municipio in clima_transformado['municipio'].unique():
        df_mun = clima_transformado[clima_transformado['municipio'] == municipio]

        for ano_safra in anos_disponiveis[1:]:
            ano_anterior = ano_safra - 1
            d_ant = df_mun[df_mun['ano'] == ano_anterior]
            d_atu = df_mun[df_mun['ano'] == ano_safra]

            if len(d_ant) > 0 and len(d_atu) > 0:
                registro = {
                    'ano_safra': ano_safra,
                    'municipio': municipio,
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
    arquivo_safra = f"{pasta_saida}/decendio_safra.csv"
    df_safra.to_csv(arquivo_safra, index=False, float_format="%.4f")
    print(f"Salvo: {arquivo_safra} ({df_safra.shape[0]} registros, {df_safra.shape[1]} colunas)")
    
    gc.collect()

def main():
    print("=== PROCESSAMENTO OTIMIZADO DE CSVs PARA DECÊNDIOS ===")
    
    # Configurações
    PASTA_ENTRADA = "data/raw/NASA_POWER"
    PASTA_SAIDA = "dados_decendio"
    CHUNK_SIZE_ARQUIVOS = 10  # Processar 100 arquivos por vez
    CHUNK_SIZE_LINHAS = 5000   # Ler 5000 linhas por vez de cada arquivo
    
    # Criar pasta de saída
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    
    # Listar todos os arquivos CSV
    arquivos = glob.glob(f"{PASTA_ENTRADA}/*.csv")
    print(f"Encontrados {len(arquivos)} arquivos CSV")
    
    if not arquivos:
        print("Nenhum arquivo CSV encontrado!")
        return
    
    # Processar arquivos em chunks
    dados_consolidados = defaultdict(list)
    total_chunks = (len(arquivos) + CHUNK_SIZE_ARQUIVOS - 1) // CHUNK_SIZE_ARQUIVOS
    
    for i in tqdm(range(0, len(arquivos), CHUNK_SIZE_ARQUIVOS), desc="Processando chunks de arquivos"):
        chunk_arquivos = arquivos[i:i + CHUNK_SIZE_ARQUIVOS]
        print(f"\nProcessando chunk {i//CHUNK_SIZE_ARQUIVOS + 1}/{total_chunks} ({len(chunk_arquivos)} arquivos)")
        
        # Processar chunk de arquivos
        dados_por_ano = processar_chunk_arquivos(chunk_arquivos, CHUNK_SIZE_LINHAS)
        
        # Consolidar dados por ano
        for ano, lista_dfs in dados_por_ano.items():
            if lista_dfs:
                df_ano = pd.concat(lista_dfs, ignore_index=True)
                dados_consolidados[ano].append(df_ano)
                del lista_dfs
        
        # Limpeza de memória
        gc.collect()
    
    # Processar e salvar dados por ano
    print("\n=== AGREGANDO E SALVANDO DADOS POR ANO ===")
    todos_dados = []
    
    for ano in sorted(dados_consolidados.keys()):
        print(f"Processando ano {ano}...")
        
        # Consolidar dados do ano
        print(f"Consolidando dados do ano {ano}...")
        df_ano = pd.concat(dados_consolidados[ano], ignore_index=True)
        
        # Agregar para decêndios
        print(f"Agregando dados do ano {ano} para decêndios...")
        df_agregado = agregar_dados_decendio(df_ano)
        
        # Salvar arquivo do ano
        print(f"Salvando dados do ano {ano}...")
        arquivo_ano = f"{PASTA_SAIDA}/decendio_{ano}.csv"
        print(f"Salvando: {arquivo_ano} ({len(df_agregado)} registros)")
        df_agregado.to_csv(arquivo_ano, index=False, float_format="%.4f")
        print(f"Salvo: {arquivo_ano} ({len(df_agregado)} registros)")

        # Guardar para consolidação total (apenas uma amostra se muito grande)
        if len(df_agregado) > 100000:  # Se muito grande, pega amostra
            todos_dados.append(df_agregado.sample(n=50000))
        else:
            todos_dados.append(df_agregado)

        # Limpar memória
        del df_ano, df_agregado
        gc.collect()

    # Salvar arquivo consolidado total
    print("\n=== CRIANDO ARQUIVO CONSOLIDADO TOTAL ===")
    if todos_dados:
        df_total = pd.concat(todos_dados, ignore_index=True)
        arquivo_total = f"{PASTA_SAIDA}/decendio_total.csv"
        df_total.to_csv(arquivo_total, index=False, float_format="%.4f")
        print(f"Salvo: {arquivo_total} ({len(df_total)} registros)")

        transformar_para_safra(df_total, PASTA_SAIDA)

    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Arquivos salvos na pasta: {PASTA_SAIDA}")
    print("Arquivos gerados:")
    for arquivo in glob.glob(f"{PASTA_SAIDA}/*.csv"):
        tamanho = os.path.getsize(arquivo) / (1024*1024)  # MB
        print(f"  - {os.path.basename(arquivo)} ({tamanho:.1f} MB)")

if __name__ == "__main__":
    main()