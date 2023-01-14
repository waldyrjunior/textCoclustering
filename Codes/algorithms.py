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
from commons import uniformDistribution, onesVector, geraMatrizSeletora, restrictionFNMTF, generatePMIMatrix, generateSPPMIMatrix

# ================================= #
# Algoritmo K-Means =============== #
# Least squares quantization in PCM #
# LLOYD, S. P. ==================== #
# ================================= #


def KMeans(matrix, k, itrMAX):
    N, M = matrix.shape
    U = np.zeros([N, k])
    C = uniformDistribution(k, M, 0, 1)

    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(C))**2
    
    while True:
        U = np.zeros([N, k])
        for i in range(N):
            U[i][np.argmin(np.sum((np.tile(matrix[i], (k, 1))-C)**2, axis=1))] = 1
            
        C = np.linalg.pinv(U.T.dot(U)).dot(U.T).dot(matrix)

        resANT = resNEW
        resNEW = np.linalg.norm(matrix - U.dot(C))**2
        
        itr = itr + 1
        
        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break
    return U, C, resNEW, itr

# ======================================================= #
# Algoritmo Non-negative Block Value Decomposition - NBVD #
# Co-clustering by block value decomposition ============ #
# LONG, B.; ZHANG, Z. M.; YU, P. S. ===================== #
# ======================================================= #


def NBVD(matrix, k, l, itrMAX):
    N, M = matrix.shape
    U = uniformDistribution(N, k, 0, 1)
    S = uniformDistribution(k, l, 0, 1)
    V = uniformDistribution(M, l, 0, 1)
    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

    while True:
        U = (U * ((matrix.dot(V).dot(S.T)) / (U.dot(S).dot(V.T).dot(V).dot(S.T))))
        V = (V * ((matrix.T.dot(U).dot(S)) / (V.dot(S.T).dot(U.T).dot(U).dot(S))))
        S = (S * ((U.T.dot(matrix).dot(V)) / (U.T.dot(U).dot(S).dot(V.T).dot(V))))
        resANT = resNEW
        resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break

        itr = itr + 1
    return U, S, V, resNEW, itr

# ================================================================== #
# Algoritmo Orthogonal Non-negative Matrix Tri-Factorization - ONM3F #
# Orthogonal nonnegative matrixtri-factorizations for clustering === #
# DING, C. H. Q.; LI, T.; PENG, W.; PARK, H. ======================= #
# ================================================================== #


def ONM3F(matrix, k, l, itrMAX):
    N, M = matrix.shape
    U = uniformDistribution(N, k, 0, 1)
    S = uniformDistribution(k, l, 0, 1)
    V = uniformDistribution(M, l, 0, 1)
    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

    while True:
        U = U * np.sqrt(matrix.dot(V).dot(S.T) / U.dot(U.T).dot(matrix).dot(V).dot(S.T))
        V = V * np.sqrt(matrix.T.dot(U).dot(S) / V.dot(V.T).dot(matrix.T).dot(U).dot(S))
        S = S * np.sqrt(U.T.dot(matrix).dot(V) / U.T.dot(U).dot(S).dot(V.T).dot(V))
        resANT = resNEW
        resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break

        itr = itr + 1
    return U, S, V, resNEW, itr

# ============================================================================================================== #
# Algoritmo Orthogonal Non-negative Matrix Tri-Factorization - ONMTF =========================================== #
# Orthogonal nonnegative matrix tri-factorization for co-clustering: Multiplicative updates on stiefel manifolds #
# YOO, J.; CHOI, S. ============================================================================================ #
# ============================================================================================================== #


def ONMTF(matrix, k, l, itrMAX):
    N, M = matrix.shape
    U = uniformDistribution(N, k, 0, 1)
    S = uniformDistribution(k, l, 0, 1)
    V = uniformDistribution(M, l, 0, 1)
    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

    while True:
        U = U * matrix.dot(V).dot(S.T) / U.dot(S).dot(V.T).dot(matrix.T).dot(U)
        V = V * matrix.T.dot(U).dot(S) / V.dot(S.T).dot(U.T).dot(matrix).dot(V)
        S = S * U.T.dot(matrix).dot(V) / U.T.dot(U).dot(S).dot(V.T).dot(V)
        resANT = resNEW
        resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break

        itr = itr + 1

    U = U.dot(np.diag(S.dot(np.diag(onesVector(M).T.dot(V)[0])).dot(onesVector(l)).T[0]))
    V = V.dot(np.diag(onesVector(k).T.dot(np.diag(onesVector(N).T.dot(U)[0])).dot(S)[0]))

    return U, S, V, resNEW, itr

# ======================================================================================= #
# Algoritmo Overlapping Non-negative Matrix Tri-Factorization - OvNMTF ================== #
# OvNMTF algorithm: an overlapping non-negative matrix tri-factorization for coclustering #
# FREITAS JR., W. L.; PERES, S. M.; SILVA, V. F. da; BRUNIALTI, L. F. =================== #
# ======================================================================================= #


def OvNMTF(matrix, k, l, itrMAX):
    N, M = matrix.shape
    U = uniformDistribution(N, k, 0, 1)
    S = uniformDistribution(k, l, 0, 1)
    I = geraMatrizSeletora(k)
    V = []
    for i in range(k):
        V.append(uniformDistribution(M, l, 0, 1))

    fatTMP = np.zeros([k, M])

    for i in range(k):
        fatTMP = fatTMP + (I[i]).dot(S).dot((V[i]).T)

    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(fatTMP)) ** 2

    while True:
        numTMP = np.zeros([N, k])
        denTMP = np.zeros([N, k])

        for i in range(k):
            numTMP = numTMP + matrix.dot(V[i]).dot(S.T).dot(I[i])

        for i in range(k):
            for j in range(k):
                denTMP = denTMP + U.dot(I[i]).dot(S).dot(V[i].T).dot(V[j]).dot(S.T).dot(I[j])

        U = U * numTMP / denTMP + 0.000001

        numTMP = np.zeros([M, l])
        denTMP = np.zeros([M, l])

        for i in range(k):
            numTMP = matrix.T.dot(U).dot(I[i]).dot(S)

            for j in range(k):
                denTMP = denTMP + (V[j]).dot(S.T).dot(I[j]).dot(U.T).dot(U).dot(I[i]).dot(S)

            V[i] = V[i] * numTMP / denTMP + 0.000001

        numTMP = np.zeros([k, l])
        denTMP = np.zeros([k, l])

        for i in range(k):
            numTMP = numTMP + (I[i]).dot(U.T).dot(matrix).dot(V[i])

        for i in range(k):
            for j in range(k):
                denTMP = denTMP + (I[i]).dot(U.T).dot(U).dot(I[j]).dot(S).dot(V[j].T).dot(V[i])

        S = S * numTMP / denTMP + 0.000001

        resANT = resNEW

        fatTMP = np.zeros([k, M])

        for i in range(k):
            fatTMP = fatTMP + (I[i]).dot(S).dot((V[i]).T)

        resNEW = np.linalg.norm(matrix - U.dot(fatTMP)) ** 2

        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break

        itr = itr + 1
    return U, S, V, resNEW, itr

# ============================================================================ #
# Algoritmo Fast Non-negative Matrix Tri Factorization - FNMTF =============== #
# Fast nonnegative matrix tri-factorization for large-scale data co-clustering #
# WANG, H.; NIE, F.; HUANG, H.; MAKEDON, F. ================================== #
# ============================================================================ #


def FNMTF(matrix, k, l, itrMAX):
    N, M = matrix.shape
    U = np.zeros([N, k])
    U = restrictionFNMTF(U, N, k)
    S = uniformDistribution(k, l, 0, 1)
    V = np.zeros([M, l])
    V = restrictionFNMTF(V, M, l)
    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T))**2
    
    while True:
        S = np.linalg.pinv(U.T.dot(U)).dot(U.T.dot(matrix).dot(V)).dot(np.linalg.pinv(V.T.dot(V)))

        RowPrototype = S.dot(V.T)
        
        U = np.zeros([N, k])
        for i in range(N):
            U[i][np.argmin(np.sum((np.tile(matrix[i], (k, 1))-RowPrototype)**2, axis=1))] = 1
        
        ColPrototype = U.dot(S)
        
        V = np.zeros([M, l])
        for j in range(M):
            V[j][np.argmin(np.sum((np.tile(matrix[:, j], (l, 1))-ColPrototype.T)**2, axis=1))] = 1
        
        resANT = resNEW
        resNEW = np.linalg.norm(matrix - U.dot(S).dot(V.T))**2
        
        itr = itr + 1
        
        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break
    return U, S, V, resNEW, itr

# ======================================================================================================== #
# Algoritmo Overlapped Binary Non-negative Matriz Tri-Factorization - BinOvNMTF ========================== #
# The BinOvNMTF algorithm: Overlapping columns co-clustering based on non-negative matrixtri-factorization #
# BRUNIALTI, L. F.; PERES, S. M.; SILVA, V. F. da; LIMA, C. A. de M. ===================================== #
# ======================================================================================================== #


def BinOvNMTF(matrix, k, l, itrMAX):
    N, M = matrix.shape
    U = np.zeros([N, k])
    U = restrictionFNMTF(U, N, k)
    S = np.zeros([k, l])
    
    I = geraMatrizSeletora(k)
    V = []
    for i in range(k):
        V.append(restrictionFNMTF(np.zeros([M, l]), M, l))
    
    fatTMP = np.zeros([k, M])
    for i in range(k):
        fatTMP = fatTMP + I[i].dot(S).dot(V[i].T)

    itr = 0
    resNEW = np.linalg.norm(matrix - U.dot(fatTMP))**2
        
    while True:
        S = np.zeros([k, l])
        for i in range(k):
            S = S + I[i].dot(np.linalg.pinv(U.T.dot(U))).dot(U.T.dot(matrix).dot(V[i])).dot(np.linalg.pinv(V[i].T.dot(V[i])))        
        
        RowPrototype = np.zeros([k, M])
        for i in range(k):
            RowPrototype = RowPrototype + I[i].dot(S).dot(V[i].T)
        
        U = np.zeros([N, k])
        for i in range(N):
            U[i][np.argmin(np.linalg.norm(np.tile(matrix[i],(k,1))-RowPrototype,axis=1)**2)]=1
        
        ColPrototype = []
        for i in range(k):
            ColPrototype.append(U.dot(I[i]).dot(S))
        
        V = []
        for i in range(k):
            V.append(np.zeros([M, l]))
        for i in range(k):
            for j in range(M):
                V[i][j][np.argmin(np.linalg.norm(np.tile(matrix[:, j], (l, 1))-ColPrototype[i].T, axis=1)**2)] = 1
        
        resANT = resNEW
        
        fatTMP = np.zeros([k, M])
        for i in range(k):
            fatTMP = fatTMP + I[i].dot(S).dot(V[i].T)
        resNEW = np.linalg.norm(matrix - U.dot(fatTMP))**2
        
        itr = itr + 1
        
        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break
    return U, S, V, resNEW, itr


# ======================================================================================================== #
# Algoritmo Word Co-Ocurrence Regularized Non-Negative Matrix Tri-Factorization - WC-NMTF ================ #
# Word Co-Ocurrence Regularized Non-Negative Matrix Tri-Factorization for Text Data Co-Clustering ======== #
# SALAH, A.; AILEM, M.; NADIF, M. ======================================================================== #
# ======================================================================================================== #


def WCNMTF(matrix, k, l, C, itrMAX):
    N, M = matrix.shape
    U = uniformDistribution(N, k, 0, 1)
    S = uniformDistribution(k, l, 0, 1)
    V = uniformDistribution(M, l, 0, 1)
    Q = uniformDistribution(M, l, 0, 1)

    LAMBDA = 1  # Salah, Ailem and Nadif (2018) definiram o parâmetro λ = 1
    PMIMatrix = generatePMIMatrix(C)
    SPPMIMatrix = generateSPPMIMatrix(PMIMatrix, 2)  # Salah, Ailem and Nadif (2018) definiram o parâmetro N = 2
    # SPPMIMatrix = C # Teste

    itr = 0
    resNEW = (1 / 2 * (np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2)) + (LAMBDA / 2 * (np.linalg.norm(SPPMIMatrix - V.dot(Q.T)) ** 2))

    while True:
        itr = itr + 1
        U = (U * ((matrix.dot(V).dot(S.T)) / (U.dot(S).dot(V.T).dot(V).dot(S.T))))
        V = (V * ((matrix.T.dot(U).dot(S) + LAMBDA * SPPMIMatrix.dot(Q)) / (V.dot(S.T.dot(U.T).dot(U).dot(S) + LAMBDA * Q.T.dot(Q)))))
        S = (S * (U.T.dot(matrix).dot(V)) / (U.T.dot(U).dot(S).dot(V.T).dot(V)))
        Q = (Q * (SPPMIMatrix.T.dot(V)) / (Q.dot(V.T).dot(V)) + 0.00001)

        resANT = resNEW
        resNEW = (1 / 2 * (np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2)) + (LAMBDA / 2 * (np.linalg.norm(SPPMIMatrix - V.dot(Q.T)) ** 2))
        resNEW2 = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2

        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break
    return U, S, V, Q, resNEW2, itr


# ======================================================================================================== #
# Algoritmo Word Co-Ocurrence Regularized Fast Non-Negative Matrix Tri-Factorization - WC-FNMTF ========== #
# ======================================================================================================== #
# ======================================================================================================== #
# ======================================================================================================== #


def WCFNMTF(matrix, k, l, C, itrMAX):
    N, M = matrix.shape
    U = np.zeros([N, k])
    U = restrictionFNMTF(U, N, k)
    S = uniformDistribution(k, l, 0, 1)
    V = np.zeros([M, l])
    V = restrictionFNMTF(V, M, l)
    Q = uniformDistribution(M, l, 0, 1)

    LAMBDA = 1 # Salah, Ailem and Nadif (2018) definiram o parâmetro λ = 1
    PMIMatrix = generatePMIMatrix(C)
    SPPMIMatrix = generateSPPMIMatrix(PMIMatrix, 2) # Salah, Ailem and Nadif (2018) definiram o parâmetro N = 2
    #SPPMIMatrix = C # Teste

    itr = 0
    resNEW = (1/2 * (np.linalg.norm(matrix - U.dot(S).dot(V.T))**2)) + (LAMBDA/2 * (np.linalg.norm(SPPMIMatrix - V.dot(Q.T))**2))
    
    while True:
        itr = itr + 1
        
        S = np.linalg.pinv(U.T.dot(U)).dot(U.T.dot(matrix).dot(V)).dot(np.linalg.pinv(V.T.dot(V)))
        Q = ((np.linalg.pinv(V.T.dot(V))).dot(V.T).dot(SPPMIMatrix)).T

        RowPrototype = S.dot(V.T)
        
        U = np.zeros([N, k])
        for i in range(N):
            U[i][np.argmin(np.linalg.norm(np.tile(matrix[i], (k, 1))-RowPrototype, axis=1)**2)] = 1
        
        ColPrototype = U.dot(S)
        
        V = np.zeros([M, l])
        for j in range(M):
            V[j][np.argmin(np.linalg.norm(np.tile(matrix[:, j], (l, 1))-ColPrototype.T, axis=1)**2)] = 1
        
        resANT = resNEW
        resNEW = (1/2 * (np.linalg.norm(matrix - U.dot(S).dot(V.T))**2)) + (LAMBDA/2 * (np.linalg.norm(SPPMIMatrix - V.dot(Q.T))**2))
        resNEW2 = np.linalg.norm(matrix - U.dot(S).dot(V.T)) ** 2
               
        if (resANT - resNEW) < 0.0001 or (itr > itrMAX):
            break
    return U, S, V, Q, resNEW2, itr
