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
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def geraMatrizSeletora(k):
    iTMP = []
    for i in range(k):
        iTMP.append(np.zeros([k, k]))
    for j in range(k):
        iTMP[j][j][j] = 1
    return iTMP


def generateC():
    return [20, 60, 100, 140, 180, 220, 260, 300, 340]


def selectC(C):
    if not C:
        return 0
    else:
        chosen = np.random.choice(C)
        C.remove(chosen)
        return chosen


def createSubMatrix(lowRow, upRow, lowColumn, upColumn, X, C):
    c = selectC(C)
    for i in range(lowRow, upRow, 1):
        for j in range(lowColumn, upColumn, 1):
            X[i][j] = uniformDistribution(1, 1, 0, 10) + c
    return X


def createRIVectorRowCENT(matrix, SVT, k):
    n, m = matrix.shape
    vetAux = np.zeros([1, n])
    MIN = 0

    for i in range(n):
        for j in range(k):
            if j == 0:
                vetAux[0][i] = j + 1
                MIN = np.linalg.norm(matrix[i] - SVT[j])
            else:
                if (np.linalg.norm(matrix[i] - SVT[j])) <= MIN:
                    vetAux[0][i] = j + 1
                    MIN = np.linalg.norm(matrix[i] - SVT[j])
    return vetAux


def createRIVectorRowFAC(U):
    n = U.shape[0]
    vetAux = np.zeros([1, n])

    for i in range(n):
        vetAux[0][i] = np.argmax(U[i]) + 1

    return vetAux


def createRIVectorColCENT(matrix, US, l):
    n, m = matrix.shape
    vetAux = np.zeros([1, m])
    MIN = 0

    for i in range(m):
        for j in range(l):
            if j == 0:
                vetAux[0][i] = j + 1
                MIN = np.linalg.norm(matrix.T[i] - US.T[j])
            else:
                if (np.linalg.norm(matrix.T[i] - US.T[j])) <= MIN:
                    vetAux[0][i] = j + 1
                    MIN = np.linalg.norm(matrix.T[i] - US.T[j])
    return vetAux


def createRIVectorColFAC(V):
    m = V.shape[0]
    vetAux = np.zeros([1, m])

    for i in range(m):
        vetAux[0][i] = np.argmax(V[i]) + 1

    return vetAux


def printCluster(X, n, m):
    printMatrix(np.array([X.T[0], ] * m).transpose())


def printCoCluster(X, n, m):
    printMatrix(np.array([X[0], ] * n))


def printMatrix(matrix):
    fig, ax = plt.subplots()
    n, m = matrix.shape
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(n):
        for j in range(m):
            ax.text(i, j, '')


def printUMatrix(matrix, l, m):
    X = matrix.copy()
    Aux = np.array([X.T[0], ] * (int(m / l))).transpose()
    for i in range(1, l, 1):
        Aux = np.concatenate((Aux, np.array([X.T[i], ] * (int(m / l))).transpose()), axis=1)
    printMatrix(Aux)


def uniformDistribution(n, m, x, y):
    return np.random.uniform(low=np.nextafter(x, x+1), high=round(np.nextafter(y, y+1), 15), size=[n, m])


def onesVector(dim):
    return np.ones([dim, 1])


def sparcity(matrix, tax):
    X = matrix.copy()
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            if np.random.rand() < tax:
                X[i][j] = 0
    return X


def restrictionFNMTF(matrix, n, m):
    for i in range(n):
        matrix[i][np.random.randint(m)] = 1
    return matrix


def leTXT(directory, dataset, matrix, algoritmo, k, l, itr, n, m, tax):
    return np.loadtxt(directory + '/' + dataset + '/' + matrix + '-' + algoritmo + '-' + str(k) + '-' + str(l) + '-' +
                      str(itr) + '-' + dataset + '-' + str(n) + '-' + str(m) + '-' + str(tax) + '.txt', delimiter=';')


def generatePMIMatrix(CoMatrix):
    # PMI(wj, wj') = log( (cjj' x c..) / (cj. x c.j')) - Equation 2 Salah, Ailem and Nadif (2018)
    a = CoMatrix  # cjj'
    b = np.array([np.array([np.sum(CoMatrix), ]*CoMatrix.shape[0]), ]*CoMatrix.shape[1])  # c..
    c = np.array([np.sum(CoMatrix, axis=1), ]*CoMatrix.shape[1]).T  # cj.
    d = np.array([np.sum(CoMatrix, axis=0), ]*CoMatrix.shape[1])  # c.j'
    return np.log2((a * b) / (c + 0.000001 * d + 0.000001) + 0.000001)


def generateSPPMIMatrix(PMIMatrix, N):
    # SPPMI = M = (mjj')
    # mjj' = max{PMI(wj, wj') - log(N), 0} - Equation 3 Salah, Ailem and Nadif (2018)
    logN = np.array([np.array([np.log2(N), ]*PMIMatrix.shape[0]), ]*PMIMatrix.shape[1])
    zeros = np.zeros([PMIMatrix.shape[0], PMIMatrix.shape[1]])
    return np.maximum(PMIMatrix - logN, zeros)


def coOcorrenceMatrix(texts, window):
    dictTXT = defaultdict(int)
    vocabulary = set()
    for text in texts:
        textTMP = text[3]
        for i in range(len(textTMP)):
            token = textTMP[i]
            vocabulary.add(token)
            next_token = textTMP[i + 1: i + 1 + window]
            for t in next_token:
                key = tuple(sorted([t, token]))
                dictTXT[key] += 1

    vocabulary = sorted(vocabulary)
    df = pd.DataFrame(data=np.zeros((len(vocabulary), len(vocabulary)), dtype=np.int16), index=vocabulary, columns=vocabulary)
    for key, value in dictTXT.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


def eliminateDiag(CoMatrix, m):
    for i in range(m):
        CoMatrix[i, i] = 0
    return CoMatrix


def generateCoMatrix(n, m, beginN, endN, beginM, endM, CoMatrix):
    for i in range(round(n * beginN), round(n * endN)):
        for j in range(round(m * beginM), round(m * endM)):
            for k in range(j, round(m * endM)):
                CoMatrix[j, k] = CoMatrix[j, k] + 1
    for i in range(round(n * beginN), round(n * endN)):
        for j in range(round(m * endM)-1, round(m * beginM)-1, -1):
            for k in range(j, round(m * beginM)-1, -1):
                CoMatrix[j, k] = CoMatrix[j, k] + 1
    return CoMatrix

