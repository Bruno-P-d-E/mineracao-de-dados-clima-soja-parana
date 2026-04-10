import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from dotenv import load_dotenv

load_dotenv()

# Credenciais
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_name = os.getenv("DB_NAME", "postgres")
db_port = os.getenv("DB_PORT", "6543")

# 1. Spark com memória reforçada
spark = SparkSession.builder \
    .appName("TCC-Soja-Limpeza-Total") \
    .config("spark.jars", "/opt/spark/jars/postgresql-42.7.2.jar") \
    .config("spark.driver.memory", "4g") \
    .master("local[*]") \
    .getOrCreate()

try:
    print("--- Iniciando processamento do CSV NASA ---")

    # 2. Lendo o arquivo. 
    # Como o cabeçalho é bagunçado, lemos sem 'header' primeiro para poder filtrar o lixo.
    df_raw = spark.read.text("data/raw/*.csv")

    # 3. FILTRO: Pegamos apenas as linhas que começam com um ano (ex: 2020)
    # Isso joga fora o "-BEGIN HEADER-", as explicações e o "-END HEADER-"
    # O regex ^\d{4}, garante que a linha comece com 4 dígitos e uma vírgula
    df_clean = df_raw.filter(F.col("value").rlike(r"^\d{4},"))

    # 4. PARSER: Transformar a linha de texto em colunas reais
    # Usamos o esquema que você postou no Postman
    cols = ["YEAR", "DOY", "AIRMASS", "ALLSKY_KT", "ALLSKY_NKT", "ALLSKY_SFC_LW_DWN", 
            "ALLSKY_SFC_LW_UP", "ALLSKY_SFC_PAR_DIFF", "ALLSKY_SFC_PAR_DIRH", 
            "ALLSKY_SFC_PAR_TOT", "ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DIRH", 
            "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_UP", 
            "ALLSKY_SFC_UVA", "ALLSKY_SFC_UVB", "ALLSKY_SFC_UV_INDEX", 
            "ALLSKY_SRF_ALB", "AOD_55", "AOD_55_ADJ"]

    # Divide a string 'value' pelas vírgulas e atribui os nomes
    df_split = df_clean.select(F.split(F.col("value"), ",").alias("split_cols"))
    
    for i, col_name in enumerate(cols):
        df_split = df_split.withColumn(col_name, F.col("split_cols").getItem(i).cast("double"))

    # 5. CONVERSÃO DE DATA: YEAR + DOY -> YYYY-MM-DD
    # A lógica: pega o primeiro dia do ano e soma o (DOY - 1)
    df_final = df_split.withColumn("data", 
        F.expr("date_add(to_date(cast(cast(YEAR as int) as string), 'yyyy'), cast(DOY as int) - 1)")
    )

    # Selecionar apenas as colunas de dados para o banco
    df_export = df_final.select("data", *cols[2:]) # Pula YEAR e DOY, já temos 'data'

    print("Dados prontos para o banco (amostra):")
    df_export.select("data", "ALLSKY_SFC_PAR_TOT", "T2M" if "T2M" in cols else "ALLSKY_KT").show(5)

    # 6. ENVIAR PARA O SUPABASE (Com correção de SSL)
    # O segredo do 'sslfactory' evita o erro do root.crt
    jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}?ssl=true&sslmode=require&sslfactory=org.postgresql.ssl.NonValidatingFactory&prepareThreshold=0"
    
    print("Conectando e enviando ao Supabase...")
    df_export.write.format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "clima_teste_nasa") \
        .option("user", db_user) \
        .option("password", db_pass) \
        .option("driver", "org.postgresql.Driver") \
        .option("batchsize", "10000") \
        .mode("overwrite") \
        .save()

    print("--- SUCESSO! Banco de dados atualizado ---")

except Exception as e:
    print(f"ERRO NO PIPELINE: {e}")

finally:
    spark.stop()