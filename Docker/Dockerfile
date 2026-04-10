# Usamos a versão slim do Debian Bullseye
FROM python:3.9-slim-bullseye

# Evita arquivos .pyc e permite logs em tempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 1. Instalar dependências do sistema e o Java (OpenJDK 11)
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jdk-headless \
    curl \
    procps \
    build-essential \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. CRIAR A PASTA DO SPARK E BAIXAR O DRIVER JDBC (Correção do Erro 23)
# Criamos a pasta primeiro com o mkdir -p
RUN mkdir -p /opt/spark/jars && \
    curl -L https://jdbc.postgresql.org/download/postgresql-42.7.2.jar -o /opt/spark/jars/postgresql-42.7.2.jar

# 3. Configurar variáveis de ambiente
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# Adicionamos o JAR ao CLASSPATH do Spark para ele reconhecer automaticamente
ENV SPARK_DIST_CLASSPATH=/opt/spark/jars/postgresql-42.7.2.jar

WORKDIR /app

# 4. Instalar as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o restante do código
COPY . .

CMD ["python", "test_pipeline.py"]