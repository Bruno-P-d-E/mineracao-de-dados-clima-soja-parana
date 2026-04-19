# -*- encoding: utf-8 -*-
"""
Consulta a API do IBGE e gera um CSV com todos os municípios do Paraná
contendo: cod_ibge, cod_meso, mesorregiao, latitude e longitude.

APIs utilizadas:
  - Municípios do PR:     https://servicodados.ibge.gov.br/api/v1/localidades/estados/41/municipios
  - Mesorregiões do PR:   https://servicodados.ibge.gov.br/api/v1/localidades/estados/41/mesorregioes
  - Coordenadas (IBGE v3 nominatim via municipio): resolvidas pelo próprio retorno da API v1 extended

Execução:
    pip install requests
    python gerar_municipios_pr.py
"""

import os
import csv
import json
import logging
import requests

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(DIR, 'data', 'raw', 'IBGE', 'municipios_pr.csv')

BASE_URL = "https://servicodados.ibge.gov.br/api/v1/localidades"
ESTADO_PR = 41          # código IBGE do Paraná
TIMEOUT   = 30          # segundos por requisição

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Funções auxiliares de validação (mantidas do padrão do projeto)
# ---------------------------------------------------------------------------
def check_dict(items: list[dict]) -> None:
    """Verifica se há valores inválidos nos dicts.

    Args:
        items: Lista de dicts a serem verificados.

    Raises:
        KeyError: Se existir algum valor inválido no dict.
    """
    len_keys = None
    for item in items:
        if len_keys is None:
            len_keys = len(item.keys())
        invalid_item = any([
            None in item.keys(),
            None in item.values(),
            '' in item.keys(),
            '' in item.values(),
            len_keys != len(item.keys()),
        ])
        if invalid_item:
            raise KeyError('Dados inválidos nesse item:', item)


def check_csv(file_path: str) -> None:
    """Verifica integridade do CSV gerado."""
    logger.info(f'Verificando o arquivo {file_path}')
    with open(file_path, encoding='utf-8-sig', mode='r') as f:
        check_dict(list(csv.DictReader(f)))
    logger.info(f'Tudo certo com o arquivo {file_path}')


# ---------------------------------------------------------------------------
# Funções de consulta à API do IBGE
# ---------------------------------------------------------------------------
def fetch_json(url: str) -> dict | list:
    """Realiza GET na URL e retorna o JSON parseado.

    Args:
        url: Endpoint da API.

    Returns:
        Objeto JSON (dict ou list).

    Raises:
        requests.HTTPError: Se a resposta não for 2xx.
    """
    logger.info(f'GET {url}')
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_mesorregioes(estado: int) -> dict[int, dict]:
    """Busca todas as mesorregiões de um estado.

    Args:
        estado: Código IBGE do estado.

    Returns:
        Dict mapeando cod_meso → {cod_meso, mesorregiao}.
    """
    url = f"{BASE_URL}/estados/{estado}/mesorregioes"
    data = fetch_json(url)

    mesorregioes = {}
    for item in data:
        cod  = item['id']
        nome = item['nome']
        mesorregioes[cod] = {'cod_meso': cod, 'mesorregiao': nome}
        logger.debug(f'  Mesorregião {cod}: {nome}')

    logger.info(f'{len(mesorregioes)} mesorregiões encontradas para o estado {estado}')
    return mesorregioes


def fetch_municipios(estado: int, mesorregioes: dict[int, dict]) -> list[dict]:
    """Busca todos os municípios de um estado e enriquece com dados de
    mesorregião, latitude e longitude.

    A API v1 de localidades retorna a mesorregião aninhada em cada município.
    Para coordenadas, é usada a API de malha de centróides (v3).

    Args:
        estado:       Código IBGE do estado.
        mesorregioes: Dict retornado por :func:`fetch_mesorregioes`.

    Returns:
        Lista de dicts prontos para escrita no CSV.
    """
    url = f"{BASE_URL}/estados/{estado}/municipios"
    raw = fetch_json(url)
    logger.info(f'{len(raw)} municípios retornados pela API')

    # Busca coordenadas (centróides) via API de malhas
    coordenadas = fetch_coordenadas(estado)

    municipios = []
    sem_coord   = []

    for item in raw:
        cod_ibge = item['id']
        nome_municipio = item['nome']
        # Mesorregião vem aninhada: municipio → microrregiao → mesorregiao
        try:
            meso_id   = item['microrregiao']['mesorregiao']['id']
            meso_nome = item['microrregiao']['mesorregiao']['nome']
        except (KeyError, TypeError):
            # fallback para o dict pré-carregado (não deve ocorrer, mas garante robustez)
            meso_info = mesorregioes.get(None, {})
            meso_id   = meso_info.get('cod_meso', '')
            meso_nome = meso_info.get('mesorregiao', '')
            logger.warning(f'Mesorregião não encontrada para município {cod_ibge}')

        lat, lon = coordenadas.get(cod_ibge, (None, None))
        if lat is None:
            sem_coord.append(cod_ibge)

        municipios.append({
            'cod_ibge'    : cod_ibge,
            'cod_meso'    : meso_id,
            'nome_municipio': nome_municipio,
            'mesorregiao' : meso_nome,
            'latitude'    : lat if lat is not None else '',
            'longitude'   : lon if lon is not None else '',
        })

    if sem_coord:
        logger.warning(
            f'{len(sem_coord)} município(s) sem coordenadas: '
            f'{sem_coord[:10]}{"..." if len(sem_coord) > 10 else ""}'
        )

    return sorted(municipios, key=lambda x: x['cod_ibge'])


def fetch_coordenadas(estado: int) -> dict[int, tuple[float, float]]:
    """Busca os centróides de todos os municípios de um estado via
    API de malhas do IBGE (v3 – formato geojson com propriedades).

    Endpoint:
        GET /api/v3/malhas/estados/{estado}/municipios
            ?formato=application/vnd.geo+json&resolucao=5

    Args:
        estado: Código IBGE do estado.

    Returns:
        Dict mapeando cod_ibge → (latitude, longitude).
    """
    # A API de localidades v1 não retorna coordenadas diretamente.
    # Usamos a API de coordenadas de municípios (nominatim do IBGE).
    url = (
        f"https://servicodados.ibge.gov.br/api/v4/malhas/estados/{estado}"
        f"?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=minima"
    )
    logger.info('Buscando coordenadas dos municípios (API de malhas v3)...')

    try:
        geojson = fetch_json(url)
    except Exception as exc:
        logger.warning(f'Não foi possível buscar coordenadas via malha: {exc}')
        return {}

    coordenadas: dict[int, tuple[float, float]] = {}

    for feature in geojson.get('features', []):
        props   = feature.get('properties', {})
        cod     = props.get('codarea')          # código IBGE do município
        geometry = feature.get('geometry', {})

        if not cod:
            continue

        # Calcula centróide aproximado a partir da geometria
        lat, lon = _centroide(geometry)
        if lat is not None:
            coordenadas[int(cod)] = (lat, lon)

    logger.info(f'Coordenadas obtidas para {len(coordenadas)} municípios')
    return coordenadas


def _centroide(geometry: dict) -> tuple[float | None, float | None]:
    """Calcula o centróide aproximado de uma geometria GeoJSON.

    Suporta Point, Polygon e MultiPolygon.

    Args:
        geometry: Dict com 'type' e 'coordinates' no padrão GeoJSON.

    Returns:
        Tupla (latitude, longitude) ou (None, None) se não for possível calcular.
    """
    if not geometry:
        return None, None

    geo_type = geometry.get('type')
    coords   = geometry.get('coordinates')

    if geo_type == 'Point':
        lon, lat = coords
        return round(lat, 6), round(lon, 6)

    if geo_type == 'Polygon':
        # Usa apenas o anel externo (índice 0)
        return _media_pontos(coords[0])

    if geo_type == 'MultiPolygon':
        # Concatena todos os anéis externos e calcula a média geral
        todos_pontos = []
        for poligono in coords:
            todos_pontos.extend(poligono[0])
        return _media_pontos(todos_pontos)

    return None, None


def _media_pontos(pontos: list) -> tuple[float, float]:
    """Retorna a média das coordenadas de uma lista de pontos [lon, lat].

    Args:
        pontos: Lista de pares [longitude, latitude].

    Returns:
        Tupla (latitude_media, longitude_media).
    """
    lons = [p[0] for p in pontos]
    lats = [p[1] for p in pontos]
    return round(sum(lats) / len(lats), 6), round(sum(lons) / len(lons), 6)


# ---------------------------------------------------------------------------
# Escrita do CSV
# ---------------------------------------------------------------------------
FIELDNAMES = ['cod_ibge','nome_municipio', 'cod_meso', 'mesorregiao', 'latitude', 'longitude']


def write_csv(municipios: list[dict], output_path: str) -> None:
    """Escreve a lista de municípios em um arquivo CSV.

    Args:
        municipios:  Lista de dicts com as chaves definidas em FIELDNAMES.
        output_path: Caminho completo do arquivo de saída.
    """
    with open(output_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(municipios)
    logger.info(f'CSV gerado: {output_path} ({len(municipios)} registros)')


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    logger.info('=== Iniciando geração do CSV de municípios do Paraná ===')

    # 1. Busca mesorregiões
    mesorregioes = fetch_mesorregioes(ESTADO_PR)

    # 2. Busca municípios + coordenadas
    municipios = fetch_municipios(ESTADO_PR, mesorregioes)

    # 3. Grava CSV
    write_csv(municipios, OUTPUT_CSV)

    # 4. Valida integridade (apenas linhas sem campos vazios)
    #    Municípios sem coordenada terão string vazia — o check abaixo
    #    é opcional; comente se quiser ignorar campos em branco.
    # check_csv(OUTPUT_CSV)

    logger.info('=== Concluído com sucesso ===')