# Relatorio metodologico - clustering nao supervisionado

Objetivo principal:
- Agrupar municipios por padroes climaticos decendiais, sem usar rendimento,
  area, producao, coordenadas ou identificadores como features de clustering.

Validacao principal:
- Metricas internas: silhouette, Davies-Bouldin, Calinski-Harabasz e equilibrio
  de tamanho dos clusters.
- Estabilidade: ARI entre seeds e ARI entre solucao completa e subamostras
  bootstrap sem reposicao.

Validacao externa/post-hoc:
- O rendimento medio entra apenas depois do clustering.
- Quartis de rendimento sao classes externas de referencia, nao ground truth.
- ARI/NMI/AMI contra quartis medem associacao, nao causalidade.
- Kruskal-Wallis e epsilon quadrado medem separacao de rendimento entre clusters.

Escolha final:
- Solucao escolhida por ranking consensual interno + estabilidade.
- O rendimento nao foi usado para selecionar a solucao final.

Melhor solucao:
- solution_id: pca_85_raw_dec__kmeans__k2
- representacao: pca_85_raw_dec
- algoritmo: kmeans
- k: 2
- silhouette: 0.4117
- Davies-Bouldin: 1.0526
- Calinski-Harabasz: 325.90
- estabilidade bootstrap ARI: 0.9937
- ARI externo vs quartis de rendimento: 0.1568
- Kruskal p-valor rendimento: 4.37939e-23

Limitacoes:
- Clustering descreve estrutura dos dados climaticos; nao prova causalidade.
- Rendimento tambem depende de solo, manejo, cultivar, tecnologia, pragas e mercado.
- Resultados externos devem ser apresentados como associacao entre clima e produtividade.
