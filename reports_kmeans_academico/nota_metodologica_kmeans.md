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
- PCA com 6 componentes, explicando 86.80% da variância.

Escolha de k:
- k selecionado: 2.
- Silhouette: 0.4115.
- Calinski-Harabasz: 319.12.
- Menor participação de cluster: 44.18%.
- Estabilidade por sementes, ARI médio: 1.0000.
- Estabilidade por bootstrap, ARI médio: 1.0000.

Validação externa pós-hoc:
- Rendimento foi usado apenas depois do clustering.
- ARI contra quartis de rendimento: 0.1978.
- NMI contra quartis de rendimento: 0.2175.
- Kruskal-Wallis p-valor: 7.43225e-32.
- Epsilon squared: 0.3643.

Interpretação:
- Os clusters descrevem estrutura climática nos municípios.
- Associação com rendimento deve ser interpretada como evidência exploratória, não causal.
- Fatores não climáticos, como solo, manejo, cultivar, tecnologia, pragas e mercado, podem influenciar o rendimento.
