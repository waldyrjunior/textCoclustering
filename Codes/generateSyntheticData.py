#######################################################################################
#######################################################################################
####### ====================================================================== ########
####### AUTOR: Waldyr Lourenço de Freitas Junior ============================= ########
####### ORIENTANDO: Waldyr Lourenço de Freitas Junior ======================== ########
####### ORIENTADOR: Sarajane Marques Peres =================================== ########
####### PROGRAMA: Programa de Pós-graduação em Sistemas de Informação - PPgSI  ########
####### UNIVERSIDADE: Universidade de São Paulo - USP ======================== ########
####### ESCOLA: Escola de Artes, Ciências e Humanidades - EACH =============== ########
####### ANO: 2021 ============================================================ ########
####### ====================================================================== ########
#######################################################################################
#######################################################################################

import numpy as np
from commons import createSubMatrix, uniformDistribution, sparcity, generateCoMatrix, eliminateDiag

directory = 'DataSets'


# Método para gerar arquivos de dados sintéticos com base em 9 estruturas de bigrupos/cogrupos
# 1. Único bigrupo (A)
# 2. Bigrupos com linhas e colunas exclusivas (B)
# 3. Bigrupos com estrutura de tabuleiro de xadrez (C)
# 4. Bigrupos com linhas exclusivas (D)
# 5. Bigrupos com colunas exclusivas (E)
# 6. Bigrupos sem sobreposição com estrutura em árvore (F)
# 7. Bigrupos não exclusivos e sem sobreposição (G)
# 8. Bigrupos com sobreposição e com estrutura hierárquica (H)
# 9. Bigrupos com sobreposição e arbitrariamente posicionados (I)
def generateSyntheticData(n, m, tax):

    # Único bigrupo
    C = [150]
    X = uniformDistribution(n, m, 0, 1)
    AMatrixRIRow = np.zeros([n, 1])
    AMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    AMatrix = createSubMatrix(round(n * 0.300), round(n * 0.700), round(m * 0.300), round(m * 0.850), X, C)
    AMatrix = sparcity(AMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.300, 0.700, 0.300, 0.850, CoMatrix)

    AMatrixRIRow[round(n * 0.000):round(n * 0.300)] = 1
    AMatrixRIRow[round(n * 0.300):round(n * 0.700)] = 2
    AMatrixRIRow[round(n * 0.700):round(n * 1.000)] = 1

    AMatrixRICol[0][round(m * 0.000):round(m * 0.300)] = 1
    AMatrixRICol[0][round(m * 0.300):round(m * 0.850)] = 2
    AMatrixRICol[0][round(m * 0.850):round(m * 1.000)] = 1

    np.savetxt(directory + '/' + 'A-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'A-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'A-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', AMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'A-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', AMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'A-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', AMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos com linhas e colunas exclusivas
    C = [50, 150, 250]
    X = uniformDistribution(n, m, 0, 1)
    BMatrixRIRow = np.zeros([n, 1])
    BMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    BMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.000), round(m * 0.333), X, C)
    BMatrix = createSubMatrix(round(n * 0.333), round(n * 0.667), round(m * 0.333), round(m * 0.667), BMatrix, C)
    BMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.667), round(m * 1.000), BMatrix, C)
    BMatrix = sparcity(BMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 0.667, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.667, 1.000, CoMatrix)

    BMatrixRIRow[round(n * 0.000):round(n * 0.333)] = 1
    BMatrixRIRow[round(n * 0.333):round(n * 0.667)] = 2
    BMatrixRIRow[round(n * 0.667):round(n * 1.000)] = 3

    BMatrixRICol[0][round(m * 0.000):round(m * 0.333)] = 1
    BMatrixRICol[0][round(m * 0.333):round(m * 0.667)] = 2
    BMatrixRICol[0][round(m * 0.667):round(m * 1.000)] = 3

    np.savetxt(directory + '/' + 'B-CMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'B-CMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'B-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', BMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'B-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', BMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'B-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', BMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos com estrutura de tabuleiro de xadrez
    C = [75, 100, 125, 150, 175, 200, 225, 250, 275]
    X = uniformDistribution(n, m, 0, 1)
    CMatrixRIRow = np.zeros([n, 1])
    CMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    CMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.000), round(m * 0.333), X, C)
    CMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.333), round(m * 0.667), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.667), round(m * 1.000), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.333), round(n * 0.667), round(m * 0.000), round(m * 0.333), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.333), round(n * 0.667), round(m * 0.333), round(m * 0.667), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.333), round(n * 0.667), round(m * 0.667), round(m * 1.000), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.000), round(m * 0.333), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.333), round(m * 0.667), CMatrix, C)
    CMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.667), round(m * 1.000), CMatrix, C)
    CMatrix = sparcity(CMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.667, 1.000, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 0.667, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 0.667, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 0.667, 0.667, 1.000, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.667, 1.000, CoMatrix)

    CMatrixRIRow[round(n * 0.000):round(n * 0.333)] = 1
    CMatrixRIRow[round(n * 0.333):round(n * 0.667)] = 2
    CMatrixRIRow[round(n * 0.667):round(n * 1.000)] = 3

    CMatrixRICol[0][round(m * 0.000):round(m * 0.333)] = 1
    CMatrixRICol[0][round(m * 0.333):round(m * 0.667)] = 2
    CMatrixRICol[0][round(m * 0.667):round(m * 1.000)] = 3

    np.savetxt(directory + '/' + 'C-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'C-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'C-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', CMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'C-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', CMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'C-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', CMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos com linhas exclusivas
    C = [50, 150, 250]
    X = uniformDistribution(n, m, 0, 1)
    DMatrixRIRow = np.zeros([n, 1])
    DMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    DMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.000), round(m * 0.450), X, C)
    DMatrix = createSubMatrix(round(n * 0.333), round(n * 0.667), round(m * 0.333), round(m * 0.667), DMatrix, C)
    DMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.550), round(m * 1.000), DMatrix, C)
    DMatrix = sparcity(DMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.000, 0.450, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 0.667, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.550, 1.000, CoMatrix)

    DMatrixRIRow[round(n * 0.000):round(n * 0.333)] = 1
    DMatrixRIRow[round(n * 0.333):round(n * 0.667)] = 2
    DMatrixRIRow[round(n * 0.667):round(n * 1.000)] = 3

    DMatrixRICol[0][round(m * 0.000):round(m * 0.333)] = 1
    DMatrixRICol[0][round(m * 0.333):round(m * 0.450)] = 2
    DMatrixRICol[0][round(m * 0.450):round(m * 0.550)] = 3
    DMatrixRICol[0][round(m * 0.550):round(m * 0.667)] = 4
    DMatrixRICol[0][round(m * 0.667):round(m * 1.000)] = 5

    np.savetxt(directory + '/' + 'D-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'D-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'D-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', DMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'D-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', DMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'D-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', DMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos com colunas exclusivas
    C = [50, 150, 250]
    X = uniformDistribution(n, m, 0, 1)
    EMatrixRIRow = np.zeros([n, 1])
    EMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    EMatrix = createSubMatrix(round(n * 0.000), round(n * 0.450), round(m * 0.000), round(m * 0.333), X, C)
    EMatrix = createSubMatrix(round(n * 0.333), round(n * 0.667), round(m * 0.333), round(m * 0.667), EMatrix, C)
    EMatrix = createSubMatrix(round(n * 0.550), round(n * 1.000), round(m * 0.667), round(m * 1.000), EMatrix, C)
    EMatrix = sparcity(EMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.450, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 0.667, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.550, 1.000, 0.667, 1.000, CoMatrix)

    EMatrixRIRow[round(n * 0.000):round(n * 0.333)] = 1
    EMatrixRIRow[round(n * 0.333):round(n * 0.450)] = 2
    EMatrixRIRow[round(n * 0.450):round(n * 0.550)] = 3
    EMatrixRIRow[round(n * 0.550):round(n * 0.667)] = 4
    EMatrixRIRow[round(n * 0.667):round(n * 1.000)] = 5

    EMatrixRICol[0][round(m * 0.000):round(m * 0.333)] = 1
    EMatrixRICol[0][round(m * 0.333):round(m * 0.667)] = 2
    EMatrixRICol[0][round(m * 0.667):round(m * 1.000)] = 3

    np.savetxt(directory + '/' + 'E-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'E-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'E-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', EMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'E-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', EMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'E-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', EMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos sem sobreposição com estrutura em árvore
    C = [75, 150, 225, 300]
    X = uniformDistribution(n, m, 0, 1)
    FMatrixRIRow = np.zeros([n, 1])
    FMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    FMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.000), round(m * 0.333), X, C)
    FMatrix = createSubMatrix(round(n * 0.333), round(n * 1.000), round(m * 0.000), round(m * 0.333), FMatrix, C)
    FMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.333), round(m * 0.667), FMatrix, C)
    FMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.667), round(m * 1.000), FMatrix, C)
    FMatrix = sparcity(FMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 1.000, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.333, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.667, 1.000, CoMatrix)

    FMatrixRIRow[round(n * 0.000):round(n * 0.333)] = 1
    FMatrixRIRow[round(n * 0.333):round(n * 0.667)] = 2
    FMatrixRIRow[round(n * 0.667):round(n * 1.000)] = 3

    FMatrixRICol[0][round(m * 0.000):round(m * 0.333)] = 1
    FMatrixRICol[0][round(m * 0.333):round(m * 0.667)] = 2
    FMatrixRICol[0][round(m * 0.667):round(m * 1.000)] = 3

    np.savetxt(directory + '/' + 'F-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'F-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'F-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', FMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'F-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', FMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'F-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', FMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos não exclusivos e sem sobreposição
    C = [75, 150, 225, 300]
    X = uniformDistribution(n, m, 0, 1)
    GMatrixRIRow = np.zeros([n, 1])
    GMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    GMatrix = createSubMatrix(round(n * 0.000), round(n * 0.667), round(m * 0.000), round(m * 0.333), X, C)
    GMatrix = createSubMatrix(round(n * 0.000), round(n * 0.333), round(m * 0.333), round(m * 1.000), GMatrix, C)
    GMatrix = createSubMatrix(round(n * 0.667), round(n * 1.000), round(m * 0.000), round(m * 0.667), GMatrix, C)
    GMatrix = createSubMatrix(round(n * 0.333), round(n * 1.000), round(m * 0.667), round(m * 1.000), GMatrix, C)
    GMatrix = sparcity(GMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.667, 0.000, 0.333, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.000, 0.333, 0.333, 1.000, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.667, 1.000, 0.000, 0.667, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.333, 1.000, 0.667, 1.000, CoMatrix)

    GMatrixRIRow[round(n * 0.000):round(n * 0.333)] = 1
    GMatrixRIRow[round(n * 0.333):round(n * 0.667)] = 2
    GMatrixRIRow[round(n * 0.667):round(n * 1.000)] = 3

    GMatrixRICol[0][round(m * 0.000):round(m * 0.333)] = 1
    GMatrixRICol[0][round(m * 0.333):round(m * 0.667)] = 2
    GMatrixRICol[0][round(m * 0.667):round(m * 1.000)] = 3

    np.savetxt(directory + '/' + 'G-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'G-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'G-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', GMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'G-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', GMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'G-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', GMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos com sobreposição e com estrutura hierárquica
    C = [50, 100, 150, 200, 250]
    X = uniformDistribution(n, m, 0, 1)
    HMatrixRIRow = np.zeros([n, 1])
    HMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    HMatrix = createSubMatrix(round(n * 0.000), round(n * 0.500), round(m * 0.000), round(m * 0.500), X, C)
    HMatrix = createSubMatrix(round(n * 0.000), round(n * 0.800), round(m * 0.650), round(m * 1.000), HMatrix, C)
    HMatrix = createSubMatrix(round(n * 0.650), round(n * 0.800), round(m * 0.175), round(m * 0.500), HMatrix, C)
    HMatrix = createSubMatrix(round(n * 0.000), round(n * 0.325), round(m * 0.800), round(m * 1.000), HMatrix, C)
    HMatrix = createSubMatrix(round(n * 0.175), round(n * 0.325), round(m * 0.175), round(m * 0.325), HMatrix, C)
    HMatrix = sparcity(HMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.000, 0.500, 0.000, 0.500, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.000, 0.800, 0.650, 1.000, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.650, 0.800, 0.175, 0.500, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.000, 0.325, 0.800, 1.000, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.175, 0.325, 0.175, 0.325, CoMatrix)

    HMatrixRIRow[round(n * 0.000):round(n * 0.175)] = 1
    HMatrixRIRow[round(n * 0.175):round(n * 0.325)] = 2
    HMatrixRIRow[round(n * 0.325):round(n * 0.500)] = 3
    HMatrixRIRow[round(n * 0.500):round(n * 0.650)] = 4
    HMatrixRIRow[round(n * 0.650):round(n * 0.800)] = 5
    HMatrixRIRow[round(n * 0.800):round(n * 1.000)] = 6

    HMatrixRICol[0][round(m * 0.000):round(m * 0.175)] = 1
    HMatrixRICol[0][round(m * 0.175):round(m * 0.325)] = 2
    HMatrixRICol[0][round(m * 0.325):round(m * 0.500)] = 3
    HMatrixRICol[0][round(m * 0.500):round(m * 0.650)] = 4
    HMatrixRICol[0][round(m * 0.650):round(m * 0.800)] = 5
    HMatrixRICol[0][round(m * 0.800):round(m * 1.000)] = 6

    np.savetxt(directory + '/' + 'H-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'H-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'H-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', HMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'H-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', HMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'H-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', HMatrixRICol, delimiter=';', fmt='%5.10f')

    # Bigrupos com sobreposição e arbitrariamente posicionados
    C = [75, 150, 225, 300]
    X = uniformDistribution(n, m, 0, 1)
    IMatrixRIRow = np.zeros([n, 1])
    IMatrixRICol = np.zeros([1, m])
    CoMatrix = np.zeros([m, m])

    IMatrix = createSubMatrix(round(n * 0.150), round(n * 0.500), round(m * 0.150), round(m * 0.500), X, C)
    IMatrix = createSubMatrix(round(n * 0.150), round(n * 0.500), round(m * 0.625), round(m * 1.000), IMatrix, C)
    IMatrix = createSubMatrix(round(n * 0.325), round(n * 0.625), round(m * 0.325), round(m * 0.825), IMatrix, C)
    IMatrix = createSubMatrix(round(n * 0.825), round(n * 1.000), round(m * 0.150), round(m * 0.625), IMatrix, C)
    IMatrix = sparcity(IMatrix, tax/100)

    CoMatrix = generateCoMatrix(n, m, 0.150, 0.500, 0.150, 0.500, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.150, 0.500, 0.625, 1.000, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.325, 0.625, 0.325, 0.825, CoMatrix)
    CoMatrix = generateCoMatrix(n, m, 0.825, 1.000, 0.150, 0.625, CoMatrix)

    IMatrixRIRow[round(n * 0.000):round(n * 0.150)] = 1
    IMatrixRIRow[round(n * 0.150):round(n * 0.325)] = 2
    IMatrixRIRow[round(n * 0.325):round(n * 0.500)] = 3
    IMatrixRIRow[round(n * 0.500):round(n * 0.650)] = 4
    IMatrixRIRow[round(n * 0.650):round(n * 0.825)] = 1
    IMatrixRIRow[round(n * 0.825):round(n * 1.000)] = 5

    IMatrixRICol[0][round(m * 0.000):round(m * 0.150)] = 1
    IMatrixRICol[0][round(m * 0.150):round(m * 0.325)] = 2
    IMatrixRICol[0][round(m * 0.325):round(m * 0.500)] = 3
    IMatrixRICol[0][round(m * 0.500):round(m * 0.650)] = 4
    IMatrixRICol[0][round(m * 0.650):round(m * 0.825)] = 5
    IMatrixRICol[0][round(m * 0.825):round(m * 1.000)] = 6

    np.savetxt(directory + '/' + 'I-CoMatrix-' + str(n) + '-' + str(m) + '-Diag.txt', CoMatrix, delimiter=';', fmt='%5.10f')
    CoMatrix = eliminateDiag(CoMatrix, m)
    np.savetxt(directory + '/' + 'I-CoMatrix-' + str(n) + '-' + str(m) + '.txt', CoMatrix, delimiter=';', fmt='%5.10f')

    np.savetxt(directory+'/'+'I-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', IMatrix, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'I-RIRow-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', IMatrixRIRow, delimiter=';', fmt='%5.10f')
    np.savetxt(directory+'/'+'I-RICol-'+str(n)+'-'+str(m)+'-'+str(tax)+'.txt', IMatrixRICol, delimiter=';', fmt='%5.10f')


# generateSyntheticData(100, 100, 0)
# generateSyntheticData(100, 100, 50)
# generateSyntheticData(100, 100, 95)
# generateSyntheticData(100, 300, 0)
# generateSyntheticData(100, 300, 50)
# generateSyntheticData(100, 300, 95)
# generateSyntheticData(300, 100, 0)
# generateSyntheticData(300, 100, 50)
# generateSyntheticData(300, 100, 95)
# generateSyntheticData(300, 300, 0)
# generateSyntheticData(300, 300, 50)
# generateSyntheticData(300, 300, 95)

