# textCoclustering
Artefatos da dissertação de Mestrado em Sistemas de Informação

##############################################
Nomenclatura dos Datasets: dataset-n-m-tax.txt
Em que,

 dataset: Tipo de Cogrupo
  A: Único bigrupo
  B: Bigrupos com linhas e colunas exclusivas
  C: Bigrupos com estrutura de tabuleiro de xadrez
  D: Bigrupos com linhas exclusivas
  E: Bigrupos com colunas exclusivas
  F: Bigrupos sem sobreposição com estrutura em árvore
  G: Bigrupos não exclusivos e sem sobreposição
  H: Bigrupos com sobreposição e com estrutura hierárquica
  I: Bigrupos com sobreposição e arbitrariamente posicionados

 n e m: Numero de linhas e numero de colunas da matriz, respectivamente

 tax: Taxa de esparsidade da matriz



#########################################################################################################
Nomenclatura dos arquivos para calcular Rand Index: dataset-RIRow-n-m-tax.txt e dataset-RICol-n-m-tax.txt
Em que,

 dataset: Tipo de Cogrupo
  A: Único bigrupo
  B: Bigrupos com linhas e colunas exclusivas
  C: Bigrupos com estrutura de tabuleiro de xadrez
  D: Bigrupos com linhas exclusivas
  E: Bigrupos com colunas exclusivas
  F: Bigrupos sem sobreposição com estrutura em árvore
  G: Bigrupos não exclusivos e sem sobreposição
  H: Bigrupos com sobreposição e com estrutura hierárquica
  I: Bigrupos com sobreposição e arbitrariamente posicionados

 RIRow e RICol: Rand Index de Linha e Rand Index de Coluna, respectivamente

 n e m: Numero de linhas e numero de colunas da matriz, respectivamente

 tax: Taxa de esparsidade da matriz



#################################################################################################################
Nomenclatura dos arquivos resultantes da execução dos algoritmos: fatMatrix-algorithm-k-l-itr-dataset-n-m-tax.txt
Em que,

 fatMatrix: Matrizes que fazem parte do processo de fatoração
  U: Matriz de coeficiente de linhas
  S: Matriz com estrutura em blocos
  V: Matriz de coeficiente de colunas

 algorithm: Algoritmos de clustering e coclustering utilizados em fatoracao de matrizes
  NBVD: Non Negative Block Value Decomposition
  ONM3F: Orthogonal Non Negative Matrix Factorization baseado em atualizacao multiplicativa
  ONMTF: Orthogonal Non Negative Matrix Factorization baseado em atualizacao multiplicativa e na teoria de derivacao na superficie com restricoes
  OvNMTF: Overlapped Non Negative Matrix Factorization

 k e l: Numero de grupos de linhas e grupos de colunas, respectivamente

 itr: Numero de iterações que o respectivo algoritmo convergiu

 dataset: Tipo de Cogrupo
  A: Único bigrupo
  B: Bigrupos com linhas e colunas exclusivas
  C: Bigrupos com estrutura de tabuleiro de xadrez
  D: Bigrupos com linhas exclusivas
  E: Bigrupos com colunas exclusivas
  F: Bigrupos sem sobreposição com estrutura em árvore
  G: Bigrupos não exclusivos e sem sobreposição
  H: Bigrupos com sobreposição e com estrutura hierárquica
  I: Bigrupos com sobreposição e arbitrariamente posicionados

 n e m: Numero de linhas e numero de colunas da matriz, respectivamente

 tax: Taxa de esparsidade da matriz
