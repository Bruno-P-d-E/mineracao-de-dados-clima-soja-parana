# Nota metodológica — K-Means não supervisionado

Objetivo:
- Agrupar municípios por padrões climáticos decendiais, sem usar rendimento, produção, área, coordenadas ou identificadores no ajuste.

Algoritmo:
- K-Means foi o único algoritmo de clustering utilizado.
- A variação de k entre 2 e 10 foi tratada como seleção de hiperparâmetro, não como comparação de algoritmos.

Pré-processamento:
- Agregação por município usando média histórica.
- Imputação por mediana nas variáveis climáticas.
- Padronização z-score.
- PCA com 6 componentes, explicando 87.11% da variância.

Escolha de k:
- k selecionado: 2.
- Silhouette: 0.4117.
- Davies-Bouldin: 1.0526.
- Calinski-Harabasz: 325.90.
- Menor participação de cluster: 45.11%.
- Estabilidade por sementes, ARI médio: 1.0000.
- Estabilidade por bootstrap, ARI médio: 0.9933.

Validação externa pós-hoc:
- Rendimento foi usado apenas depois do clustering.
- ARI contra quartis de rendimento: 0.1568.
- NMI contra quartis de rendimento: 0.1671.
- Kruskal-Wallis p-valor: 4.37939e-23.
- Epsilon squared: 0.2441.

Interpretação:
- Os clusters descrevem estrutura climática nos municípios.
- Associação com rendimento deve ser interpretada como evidência exploratória, não causal.
- Fatores não climáticos, como solo, manejo, cultivar, tecnologia, pragas e mercado, podem influenciar o rendimento.
