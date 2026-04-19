import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from dotenv import load_dotenv


load_dotenv()

USER = os.getenv("POSTGRES_USER")
PASSWORD_ENCODED = quote_plus(os.getenv("POSTGRES_PASSWORD"))
DB = os.getenv("POSTGRES_DB")

# Se rodar por dentro do container (docker exec), use "postgres"
HOST = "postgres" 
PORT = "5432"

engine = create_engine(f"postgresql://{USER}:{PASSWORD_ENCODED}@{HOST}:{PORT}/{DB}")

def carregar_staging():
    print("Iniciando carga das tabelas Staging...")

    # 1. Carga da Dimensão de Municípios
    # USE BARRAS NORMAIS (/)
    caminho_mun = "data/raw/IBGE/municipios_pr.csv"
    if not os.path.exists(caminho_mun):
        # AQUI O SCRIPT GRITA E MORRE, PROTEGENDO O BANCO DE DADOS
        raise FileNotFoundError(f"ERRO FATAL: Arquivo não encontrado em {caminho_mun}. O container Linux não enxerga este arquivo.")

    print("Carregando Municípios...")
    df_mun = pd.read_csv(caminho_mun)
    df_mun.to_sql('stg_municipios', engine, if_exists='replace', index=False)

    # 2. Carga do SIDRA PAM
    print
    caminho_sidra = "data/raw/PAM_SIDRA/PAM_SIDRA.csv" # Ajuste se o SIDRA não estiver na pasta IBGE
    if not os.path.exists(caminho_sidra):
        raise FileNotFoundError(f"ERRO FATAL: Arquivo não encontrado em {caminho_sidra}.")

    print("Carregando SIDRA PAM...")
    df_sidra = pd.read_csv(caminho_sidra) 
    df_sidra.columns = [
        'cod_ibge', 'municipio_nome', 'ano', 
        'area_plantada_ha', 'area_plantada_perc', 
        'area_colhida_ha', 'area_colhida_perc', 
        'qtd_produzida_ton', 'rendimento_kg_ha', 
        'valor_producao_mil_rs', 'valor_producao_perc'
    ]
    df_sidra.to_sql('stg_sidra_pam', engine, if_exists='replace', index=False)

    # 3. Carga do NASA POWER
    print("Carregando NASA POWER...")
    pasta_nasa = "data/raw/NASA_POWER/" # USE BARRAS NORMAIS (/)
    if not os.path.exists(pasta_nasa):
        raise FileNotFoundError(f"ERRO FATAL: A pasta {pasta_nasa} não existe dentro do container.")


    arquivos_nasa = glob.glob(os.path.join(pasta_nasa, "*.csv"))
    
    if not arquivos_nasa:
        raise FileNotFoundError(f"ERRO FATAL: A pasta {pasta_nasa} existe, mas está vazia. Cadê os milhares de CSVs?")
    
    print(f"Encontrados {len(arquivos_nasa)} arquivos do NASA POWER. Iniciando ingestão em lotes...")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS stg_nasa_power;"))
    
    primeiro = True
    for arquivo in arquivos_nasa:
        df_nasa = pd.read_csv(arquivo)
        modo = 'replace' if primeiro else 'append'
        df_nasa.to_sql('stg_nasa_power', engine, if_exists=modo, index=False)
        primeiro = False
        
    print("Carga do NASA POWER concluída.")

def executar_processamento_sql():
    print("Executando transformações no banco (Limpeza de Nulos)...")
    
    # NASA: A estrutura é "parametro" e "valor". O -999 está na coluna valor.
    query_nasa = """
        DROP TABLE IF EXISTS f_clima;
        CREATE TABLE f_clima AS
        SELECT 
            ibge::INT AS cod_ibge,
            data::DATE,
            parametro,
            NULLIF(valor, -999) AS valor
        FROM stg_nasa_power;
        
        CREATE INDEX idx_f_clima_ibge_data ON f_clima(cod_ibge, data);
    """
    
    # SIDRA: Limpando o "-" de todas as colunas numéricas que importam
    query_sidra = """
        DROP TABLE IF EXISTS f_producao;
        CREATE TABLE f_producao AS
        SELECT 
            cod_ibge::INT,
            ano::INT,
            CASE WHEN area_plantada_ha = '-' THEN 0 ELSE area_plantada_ha::NUMERIC END AS area_plantada_ha,
            CASE WHEN area_colhida_ha = '-' THEN 0 ELSE area_colhida_ha::NUMERIC END AS area_colhida_ha,
            CASE WHEN qtd_produzida_ton = '-' THEN 0 ELSE qtd_produzida_ton::NUMERIC END AS qtd_produzida_ton,
            CASE WHEN rendimento_kg_ha = '-' THEN 0 ELSE rendimento_kg_ha::NUMERIC END AS rendimento_kg_ha,
            CASE WHEN valor_producao_mil_rs = '-' THEN 0 ELSE valor_producao_mil_rs::NUMERIC END AS valor_producao_mil_rs
        FROM stg_sidra_pam;
        
        CREATE INDEX idx_f_producao_ibge ON f_producao(cod_ibge, ano);
    """

    # MUNICÍPIOS: Apenas tipando e movendo para a camada final
    query_mun = """
        DROP TABLE IF EXISTS dim_municipios;
        CREATE TABLE dim_municipios AS
        SELECT 
            cod_ibge::INT,
            nome_municipio,
            cod_meso::INT,
            mesorregiao,
            latitude::FLOAT,
            longitude::FLOAT
        FROM stg_municipios;
        
        ALTER TABLE dim_municipios ADD PRIMARY KEY (cod_ibge);
    """
    
    with engine.begin() as conn:
        conn.execute(text(query_mun))
        conn.execute(text(query_nasa))
        conn.execute(text(query_sidra))
        print("Modelagem Star Schema concluída com sucesso!")

if __name__ == "__main__":
    carregar_staging()
    executar_processamento_sql()