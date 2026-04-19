import os
import re
import sys
import time
import unicodedata
import urllib3
import requests
import multiprocessing
import pandas as pd
import io
from datetime import timedelta

urllib3.disable_warnings()

# --- CONFIGURAÇÕES ---
RAW_DIR = "data/raw/NASA_POWER"
os.makedirs(RAW_DIR, exist_ok=True)

MAX_PROCESSES = 5 
DELAY_BASE = 1     
START_DATE = "20170101"
END_DATE = "20241231"
COMMUNITY = "AG"

# Formato CSV solicitado
REQUEST_TEMPLATE = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    "parameters={parameters}" 
    "&community=" + COMMUNITY +
    "&longitude={longitude}"
    "&latitude={latitude}"
    "&start=" + START_DATE +
    "&end=" + END_DATE +
    "&format=CSV"
)

def sanitize_nome(nome: str) -> str:
    nfkd = unicodedata.normalize("NFKD", nome)
    ascii_nome = nfkd.encode("ASCII", "ignore").decode()
    return re.sub(r"[^\w\-]", "_", ascii_nome)

def download_one(task: dict):
    params_str = ",".join(task["parametros"])
    url = REQUEST_TEMPLATE.format(
        parameters=params_str,
        latitude=task["lat"],
        longitude=task["lon"]
    )
    
    max_tentativas = 5 
    ult_erro = "Desconhecido"
    
    for tentativa in range(max_tentativas):
        try:
            time.sleep(DELAY_BASE)
            response = requests.get(url=url, verify=False, timeout=60.0)

            if response.status_code == 429:
                time.sleep(DELAY_BASE * (tentativa + 2))
                continue

            if response.status_code != 200:
                ult_erro = f"HTTP_{response.status_code}"
                time.sleep(DELAY_BASE)
                continue

            raw_text = response.text
            if "-END HEADER-" not in raw_text:
                ult_erro = "CABECALHO_INVALIDO"
                continue

            # Corta o cabeçalho explicativo
            csv_data = raw_text.split("-END HEADER-")[-1].strip()
            
            df_api = pd.read_csv(io.StringIO(csv_data))
            
            # --- CONVERSÃO DE YEAR + DOY PARA DATA ISO ---
            # %Y é o ano com 4 dígitos, %j é o dia do ano (001-366)
            if "YEAR" in df_api.columns and "DOY" in df_api.columns:
                df_api["data"] = pd.to_datetime(
                    df_api["YEAR"].astype(str) + df_api["DOY"].astype(str).str.zfill(3), 
                    format='%Y%j'
                ).dt.strftime('%Y-%m-%d')
            else:
                # Fallback caso a NASA mude o formato dinamicamente para MO/DY
                ult_erro = "COLUNAS_DATA_NAO_ENCONTRADAS"
                continue

            for param_name in task["parametros"]:
                if param_name not in df_api.columns:
                    continue
                
                nome_safe = sanitize_nome(task["municipio"])
                filepath = os.path.join(RAW_DIR, f"{task['ibge']}_{nome_safe}_{param_name}.csv")
                
                df_final = df_api[["data", param_name]].copy()
                df_final.columns = ["data", "valor"]
                df_final["ibge"] = task["ibge"]
                df_final["municipio"] = task["municipio"]
                df_final["lat"] = task["lat"]
                df_final["lon"] = task["lon"]
                df_final["parametro"] = param_name
                
                df_final = df_final[["ibge", "municipio", "lat", "lon", "data", "parametro", "valor"]]
                df_final.to_csv(filepath, index=False, encoding="utf-8")

            return {**task, "status": "OK"}

        except Exception as e:
            ult_erro = str(e)
            time.sleep(DELAY_BASE)
            continue
            
    return {**task, "status": f"ABORT_FATAL: {ult_erro}"}

class NasaPowerExtractor:
    def execute(self):
        print(f"[!] Stop após 5 falhas consecutivas | Delay base: {DELAY_BASE}s")
        
        try:
            df_muni = pd.read_csv("./data/raw/IBGE/municipios_pr.csv")
            df_attr = pd.read_csv("atributos.csv")
            lista_attr = df_attr["Atributo"].dropna().unique().tolist()
        except Exception as e:
            print(f"[ERRO] Falha nos arquivos base: {e}")
            return

        chunks = [lista_attr[i:i + 18] for i in range(0, len(lista_attr), 18)]
        tasks = []
        for _, muni in df_muni.iterrows():
            for chunk in chunks:
                tasks.append({
                    "ibge": muni["cod_ibge"], "municipio": muni["nome_municipio"],
                    "lat": muni["latitude"], "lon": muni["longitude"], "parametros": chunk
                })

        total = len(tasks)
        start_time = time.time()
        concluidos = 0
        
        pool = multiprocessing.Pool(MAX_PROCESSES)
        results = pool.imap_unordered(download_one, tasks)

        try:
            for result in results:
                if "ABORT_FATAL" in result["status"]:
                    print(f"\n\n[ERRO CRÍTICO] Falha definitiva no IBGE {result['ibge']}: {result['status']}")
                    print("[!] Encerrando todos os requests imediatamente conforme solicitado.")
                    pool.terminate()
                    sys.exit(1)

                concluidos += 1
                elapsed = time.time() - start_time
                avg = elapsed / concluidos
                eta = (total - concluidos) * avg
                
                sys.stdout.write(
                    f"\r[{concluidos}/{total}] {concluidos/total:>4.1%} | "
                    f"Decorrido: {str(timedelta(seconds=int(elapsed)))} | "
                    f"Restante: {str(timedelta(seconds=int(eta)))} | "
                    f"IBGE {result['ibge']}: OK    "
                )
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n[!] Parada manual.")
            pool.terminate()
        finally:
            pool.close()
            pool.join()

if __name__ == "__main__":
    NasaPowerExtractor().execute()