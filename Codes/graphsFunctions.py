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
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn


def AlgorithmCompareGRAPH(data, algorithm1, algorithm2, dataset, sparsity):
    if sparsity == 'All':
        ALG1_0 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == 0)]
        ALG2_0 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == 0)]
        ALG2_50 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == 50)]
        ALG1_50 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == 50)]
        ALG1_95 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == 95)]
        ALG2_95 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == 95)]
    else:
        ALG1 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == sparsity)]
        ALG2 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == sparsity)]
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(7, 7))
    sn.despine(left=True, bottom=True)
    if sparsity == 'All':
        plt.plot(ALG2_0.LogError, ALG2_0.LogError, 'b--', color='midnightblue', linewidth=1)
        plt.scatter(ALG1_0.LogError, ALG2_0.LogError, marker='.', color='orangered')
        plt.scatter(ALG1_50.LogError, ALG2_50.LogError, marker='.', color='slategray')
        plt.scatter(ALG1_95.LogError, ALG2_95.LogError, marker='.', color='cornflowerblue')
    else:
        plt.plot(ALG2.LogError, ALG2.LogError, 'b--', color='midnightblue', linewidth=1)
        plt.scatter(ALG1.LogError, ALG2.LogError, marker='.', color='cornflowerblue')
    plt.xlabel(algorithm1)
    plt.ylabel(algorithm2)
    plt.show()


def ARICompareGRAPH(data, algorithm1, algorithm2, dataset, sparsityTax, ARI):
    # data = dataframe pandas contendo carga de CSV
    # algorithm1 = 'KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF'
    # algorithm2 = 'KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF'
    # dataset =
    #   A - Único bigrupo
    #   B - Bigrupos com linhas e colunas exclusivas
    #   C - Bigrupos com estrutura de tabuleiro de xadrez
    #   D - Bigrupos com linhas exclusivas
    #   E - Bigrupos com colunas exclusivas
    #   F - Bigrupos sem sobreposição com estrutura em árvore
    #   G - Bigrupos não exclusivos e sem sobreposição
    #   H - Bigrupos com sobreposição e com estrutura hierárquica
    #   I - Bigrupos com sobreposição e arbitrariamente posicionados
    # sparsityTax = 'All', '0', '50', '95'
    # ARI = 'linha', 'coluna'
    # nomenclatura exemplo da imagem gerada: Scatterplot-FNMTF-BinOvNMTF-E-All-ARICol.PNG

    if ARI == 'linha':
        ARI = 'ARIRow'
        if sparsityTax == 'All':
            titulo = 'ARI Linhas'+' - Conjunto '+dataset
        else:
            titulo = 'ARI Linhas'+' - Conjunto '+dataset+' (Esparsidade '+sparsityTax+'%)'
    elif ARI == 'coluna':
        ARI = 'ARICol'
        if sparsityTax == 'All':
            titulo = 'ARI Colunas'+' - Conjunto '+dataset
        else:
            titulo = 'ARI Colunas'+' - Conjunto '+dataset+' (Esparsidade '+sparsityTax+'%)'
    
    x = np.zeros([2, 1])
    y = np.zeros([2, 1])
    x[0] = -1
    y[0] = -1
    x[1] = 1
    y[1] = 1
    
    if sparsityTax == 'All':
        ALG1_0 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == '0')]
        ALG2_0 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == '0')]
        ALG1_50 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == '50')]
        ALG2_50 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == '50')]
        ALG1_95 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == '95')]
        ALG2_95 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == '95')]
        a1 = np.array(pd.to_numeric(ALG1_0[ARI], errors='coerce'))
        b1 = np.array(pd.to_numeric(ALG2_0[ARI], errors='coerce'))
        a2 = np.array(pd.to_numeric(ALG1_50[ARI], errors='coerce'))
        b2 = np.array(pd.to_numeric(ALG2_50[ARI], errors='coerce'))
        a3 = np.array(pd.to_numeric(ALG1_95[ARI], errors='coerce'))
        b3 = np.array(pd.to_numeric(ALG2_95[ARI], errors='coerce'))
    else:
        ALG1 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
        ALG2 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
        a1 = np.array(pd.to_numeric(ALG1[ARI], errors='coerce'))
        b1 = np.array(pd.to_numeric(ALG2[ARI], errors='coerce'))
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'b--', color='midnightblue', linewidth=0.7)
    if sparsityTax == 'All':
        plt.scatter(a1, b1, marker='.', color='orangered', label = '0%')
        plt.scatter(a2, b2, marker='.', color='slategray', label = '50%')
        plt.scatter(a3, b3, marker='.', color='cornflowerblue', label = '95%')
        plt.legend(title='Esparsidade')
    else:
        plt.scatter(a1, b1, marker='.', color='cornflowerblue')
    plt.title(titulo)
    plt.xlabel(algorithm1)
    plt.ylabel(algorithm2)
    plt.savefig('Scatterplot-'+algorithm1+'-'+algorithm2+'-'+dataset+'-'+sparsityTax+'-'+ARI+'.PNG', format='png', dpi=1000)
    plt.show()


def ErrorCompareGRAPH(data, algorithm1, algorithm2, dataset, itrmax, sparsity):
    if sparsity == 'All':
        ALG1_0 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == 0)]
        ALG2_0 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == 0)]
        ALG2_50 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == 50)]
        ALG1_50 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == 50)]
        ALG1_95 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == 95)]
        ALG2_95 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == 95)]
    else:
        ALG1 = data[(data.Algorithm == algorithm1) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == sparsity)]
        ALG2 = data[(data.Algorithm == algorithm2) & (data.DatasetName == dataset) & (data.IteratorMax == itrmax) & (data.SparsityTax == sparsity)]
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(20, 6))
    # sn.despine(left=True, bottom=True)
    if sparsity == 'All':
        plt.plot(ALG1_0.LogError.values, 'b', color='darkgray', linewidth=1)
        plt.plot(ALG1_50.LogError.values, 'b', color='orangered', linewidth=1)
        plt.plot(ALG1_95.LogError.values, 'b', color='dodgerblue', linewidth=1)
        plt.plot(ALG2_0.LogError.values, 'b', color='forestgreen', linewidth=1)
        plt.plot(ALG2_50.LogError.values, 'b', color='orchid', linewidth=1)
        plt.plot(ALG2_95.LogError.values, 'b', color='r', linewidth=1)
    else:
        plt.plot(ALG1.LogError.values, 'b', color='darkgray', linewidth=1)
        plt.plot(ALG2.LogError.values, 'b', color='cornflowerblue', linewidth=1)
    plt.xlabel(algorithm1)
    plt.ylabel('LOG Error')
    plt.show()


def boxPlotGraphARI(data, algorithm, dataset, sparsityTax, ARI):
# data = dataframe pandas contendo carga de CSV
# algorithm = 'All', 'KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF'
# datasets = 'All'
    # A - Único bigrupo
    # B - Bigrupos com linhas e colunas exclusivas
    # C - Bigrupos com estrutura de tabuleiro de xadrez
    # D - Bigrupos com linhas exclusivas
    # E - Bigrupos com colunas exclusivas
    # F - Bigrupos sem sobreposição com estrutura em árvore
    # G - Bigrupos não exclusivos e sem sobreposição
    # H - Bigrupos com sobreposição e com estrutura hierárquica
    # I - Bigrupos com sobreposição e arbitrariamente posicionados
# sparsity = 'All', '0', '50', '95'
# ARI = 'linha', 'coluna'

    if ARI == 'linha':
        ARI = 'ARIRow'
        if sparsityTax == ' ':
            titulo = 'Boxplot ARI Linhas'+' - Dataset '+dataset
        else:
            titulo = 'Boxplot ARI Linhas'+' - Dataset '+dataset+' (Esparsidade '+sparsityTax+')'
    elif ARI == 'coluna':
        ARI = 'ARICol'
        if sparsityTax == ' ':
            titulo = 'Boxplot ARI Colunas'+' - Dataset '+dataset
        else:
            titulo = 'Boxplot ARI Colunas'+' - Dataset '+dataset+' (Esparsidade '+sparsityTax+')'

    # Boxplot de ARI de linhas/colunas para todos os datasets, quebrado por algoritmo
    if algorithm == 'All' and dataset == 'All' and sparsityTax == 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        
        if ARI == 'ARIRow':
            labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']
            
            ALG1 = data[(data.Algorithm == labels[0])]
            ALG2 = data[(data.Algorithm == labels[1])]
            ALG3 = data[(data.Algorithm == labels[2])]
            ALG4 = data[(data.Algorithm == labels[3])]
            ALG5 = data[(data.Algorithm == labels[4])]
            ALG6 = data[(data.Algorithm == labels[5])]
            ALG7 = data[(data.Algorithm == labels[6])]        
            
            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')
            g = pd.to_numeric(ALG7[ARI], errors='coerce')
            
            FINAL = [a, b, c, d, e, f, g]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]

        if ARI == 'ARICol':
            labels = ['NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

            ALG1 = data[(data.Algorithm == labels[0])]
            ALG2 = data[(data.Algorithm == labels[1])]
            ALG3 = data[(data.Algorithm == labels[2])]
            ALG4 = data[(data.Algorithm == labels[3])]
            ALG5 = data[(data.Algorithm == labels[4])]
            ALG6 = data[(data.Algorithm == labels[5])]

            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')
                    
            FINAL = [a, b, c, d, e, f]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean()]
    
    # Boxplot de ARI de linhas/colunas para um dataset específico, quebrado por algoritmo
    elif algorithm == 'All' and dataset != 'All' and sparsityTax == 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')

        if ARI == 'ARIRow':
            labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

            ALG1 = data[(data.Algorithm == labels[0]) & (data.DatasetName == dataset)]
            ALG2 = data[(data.Algorithm == labels[1]) & (data.DatasetName == dataset)]
            ALG3 = data[(data.Algorithm == labels[2]) & (data.DatasetName == dataset)]
            ALG4 = data[(data.Algorithm == labels[3]) & (data.DatasetName == dataset)]
            ALG5 = data[(data.Algorithm == labels[4]) & (data.DatasetName == dataset)]
            ALG6 = data[(data.Algorithm == labels[5]) & (data.DatasetName == dataset)]
            ALG7 = data[(data.Algorithm == labels[6]) & (data.DatasetName == dataset)]

            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')
            g = pd.to_numeric(ALG7[ARI], errors='coerce')

            FINAL = [a, b, c, d, e, f, g]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]
            
        if ARI == 'ARICol':
            labels = ['NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

            ALG1 = data[(data.Algorithm == labels[0]) & (data.DatasetName == dataset)]
            ALG2 = data[(data.Algorithm == labels[1]) & (data.DatasetName == dataset)]
            ALG3 = data[(data.Algorithm == labels[2]) & (data.DatasetName == dataset)]
            ALG4 = data[(data.Algorithm == labels[3]) & (data.DatasetName == dataset)]
            ALG5 = data[(data.Algorithm == labels[4]) & (data.DatasetName == dataset)]
            ALG6 = data[(data.Algorithm == labels[5]) & (data.DatasetName == dataset)]

            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')

            FINAL = [a, b, c, d, e, f]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean()]
            
    # Boxplot de ARI de linhas/colunas para todos datasets e um algoritmo específico
    elif algorithm != 'All' and dataset == 'All' and sparsityTax == 'All':
        labels = [algorithm]
        ALG = data[(data.Algorithm == algorithm)]
        x = pd.to_numeric(ALG[ARI], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]
            
    # Boxplot de ARI de linhas/colunas para um dataset específico e um algoritmo específico
    elif algorithm != 'All' and dataset != 'All' and sparsityTax == 'All':
        labels = [algorithm]
        ALG = data[(data.Algorithm == algorithm) & (data.DatasetName == dataset)]
        x = pd.to_numeric(ALG[ARI], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]

    # Boxplot de ARI de linhas/colunas para todos os datasets, esparsidade específica e quebrado por algoritmo
    elif algorithm == 'All' and dataset == 'All' and sparsityTax != 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        
        if ARI == 'ARIRow':
            labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']
            
            ALG1 = data[(data.Algorithm == labels[0]) & (data.SparsityTax == sparsityTax)]
            ALG2 = data[(data.Algorithm == labels[1]) & (data.SparsityTax == sparsityTax)]
            ALG3 = data[(data.Algorithm == labels[2]) & (data.SparsityTax == sparsityTax)]
            ALG4 = data[(data.Algorithm == labels[3]) & (data.SparsityTax == sparsityTax)]
            ALG5 = data[(data.Algorithm == labels[4]) & (data.SparsityTax == sparsityTax)]
            ALG6 = data[(data.Algorithm == labels[5]) & (data.SparsityTax == sparsityTax)]
            ALG7 = data[(data.Algorithm == labels[6]) & (data.SparsityTax == sparsityTax)]
            
            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')
            g = pd.to_numeric(ALG7[ARI], errors='coerce')
            
            FINAL = [a, b, c, d, e, f, g]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]

        if ARI == 'ARICol':
            labels = ['NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

            ALG1 = data[(data.Algorithm == labels[0]) & (data.SparsityTax == sparsityTax)]
            ALG2 = data[(data.Algorithm == labels[1]) & (data.SparsityTax == sparsityTax)]
            ALG3 = data[(data.Algorithm == labels[2]) & (data.SparsityTax == sparsityTax)]
            ALG4 = data[(data.Algorithm == labels[3]) & (data.SparsityTax == sparsityTax)]
            ALG5 = data[(data.Algorithm == labels[4]) & (data.SparsityTax == sparsityTax)]
            ALG6 = data[(data.Algorithm == labels[5]) & (data.SparsityTax == sparsityTax)]

            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')
                    
            FINAL = [a, b, c, d, e, f]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean()]
    
    # Boxplot de ARI de linhas/colunas para um dataset específico, quebrado por algoritmo
    elif algorithm == 'All' and dataset != 'All' and sparsityTax != 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')

        if ARI == 'ARIRow':
            labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

            ALG1 = data[(data.Algorithm == labels[0]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG2 = data[(data.Algorithm == labels[1]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG3 = data[(data.Algorithm == labels[2]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG4 = data[(data.Algorithm == labels[3]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG5 = data[(data.Algorithm == labels[4]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG6 = data[(data.Algorithm == labels[5]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG7 = data[(data.Algorithm == labels[6]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]

            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')
            g = pd.to_numeric(ALG7[ARI], errors='coerce')

            FINAL = [a, b, c, d, e, f, g]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]
            
        if ARI == 'ARICol':
            labels = ['NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

            ALG1 = data[(data.Algorithm == labels[0]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG2 = data[(data.Algorithm == labels[1]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG3 = data[(data.Algorithm == labels[2]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG4 = data[(data.Algorithm == labels[3]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG5 = data[(data.Algorithm == labels[4]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
            ALG6 = data[(data.Algorithm == labels[5]) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]

            a = pd.to_numeric(ALG1[ARI], errors='coerce')
            b = pd.to_numeric(ALG2[ARI], errors='coerce')
            c = pd.to_numeric(ALG3[ARI], errors='coerce')
            d = pd.to_numeric(ALG4[ARI], errors='coerce')
            e = pd.to_numeric(ALG5[ARI], errors='coerce')
            f = pd.to_numeric(ALG6[ARI], errors='coerce')

            FINAL = [a, b, c, d, e, f]
            measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean()]
            
    # Boxplot de ARI de linhas/colunas para todos datasets e um algoritmo específico
    elif algorithm != 'All' and dataset == 'All' and sparsityTax != 'All':
        labels = [algorithm]
        ALG = data[(data.Algorithm == algorithm) & (data.SparsityTax == sparsityTax)]
        x = pd.to_numeric(ALG[ARI], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]
            
    # Boxplot de ARI de linhas/colunas para um dataset específico e um algoritmo específico
    elif algorithm != 'All' and dataset != 'All' and sparsityTax != 'All':
        labels = [algorithm]
        ALG = data[(data.Algorithm == algorithm) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
        x = pd.to_numeric(ALG[ARI], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]
        
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    ax1.set_title(titulo)
    ax1.set_xlabel('Algoritmo')
    ax1.set_ylabel('ARI')
    
    bp = ax1.boxplot(FINAL, labels=labels, vert=True, showmeans=True,  meanprops=meanpointprops)
    
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='8') # . , o 8 * + x 
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_axisbelow(True)
    
    ax1.set_xlim(0.5, len(labels) + 0.5)
    top = 1.2
    bottom = -1
    ax1.set_ylim(bottom, top)
    
    pos = np.arange(7) + 1
    upper_labels = [str(np.round(s, 2)) for s in measureList]

    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], .95, upper_labels[tick], transform=ax1.get_xaxis_transform(), horizontalalignment='center', color='darkgreen')

    plt.savefig('Boxplot-'+algorithm+'-'+dataset+'-'+sparsityTax+'-'+ARI+'.PNG', format='png', dpi = 1000)


def boxPlotGraphError(data, algorithm, dataset, sparsityTax):
# data = dataframe pandas contendo carga de CSV
# algorithm = 'All', 'KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF'
# datasets = 'All'
    # A - Único bigrupo
    # B - Bigrupos com linhas e colunas exclusivas
    # C - Bigrupos com estrutura de tabuleiro de xadrez
    # D - Bigrupos com linhas exclusivas
    # E - Bigrupos com colunas exclusivas
    # F - Bigrupos sem sobreposição com estrutura em árvore
    # G - Bigrupos não exclusivos e sem sobreposição
    # H - Bigrupos com sobreposição e com estrutura hierárquica
    # I - Bigrupos com sobreposição e arbitrariamente posicionados
# sparsity = 'All', '0', '50', '95'

    if sparsityTax == ' ':
        titulo = 'Boxplot Error'+' - Dataset '+dataset
    else:
        titulo = 'Boxplot Error'+' - Dataset '+dataset+' (Esparsidade '+sparsityTax+')'
    
    # Boxplot de erro de reconstrução para todos os datasets, quebrado por algoritmo
    if algorithm == 'All' and dataset == 'All' and sparsityTax == 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']
        
        min_max_scaler = MinMaxScaler()        
        data[["Error"]] = min_max_scaler.fit_transform(data[["Error"]])
        
        ALG1 = data[(data.Algorithm == labels[0])]
        ALG2 = data[(data.Algorithm == labels[1])]
        ALG3 = data[(data.Algorithm == labels[2])]
        ALG4 = data[(data.Algorithm == labels[3])]
        ALG5 = data[(data.Algorithm == labels[4])]
        ALG6 = data[(data.Algorithm == labels[5])]
        ALG7 = data[(data.Algorithm == labels[6])]

        a = pd.to_numeric(ALG1['Error'], errors='coerce')
        b = pd.to_numeric(ALG2['Error'], errors='coerce')
        c = pd.to_numeric(ALG3['Error'], errors='coerce')
        d = pd.to_numeric(ALG4['Error'], errors='coerce')
        e = pd.to_numeric(ALG5['Error'], errors='coerce')
        f = pd.to_numeric(ALG6['Error'], errors='coerce')
        g = pd.to_numeric(ALG7['Error'], errors='coerce')

        FINAL = [a, b, c, d, e, f, g]
        measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]

    # Boxplot de erro de reconstrução para um dataset específico, quebrado por algoritmo
    elif algorithm == 'All' and dataset != 'All' and sparsityTax == 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']

        min_max_scaler = MinMaxScaler()
        ALG = data[(data.DatasetName == dataset)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])
        
        ALG1 = ALG[(ALG.Algorithm == labels[0])]
        ALG2 = ALG[(ALG.Algorithm == labels[1])]
        ALG3 = ALG[(ALG.Algorithm == labels[2])]
        ALG4 = ALG[(ALG.Algorithm == labels[3])]
        ALG5 = ALG[(ALG.Algorithm == labels[4])]
        ALG6 = ALG[(ALG.Algorithm == labels[5])]
        ALG7 = ALG[(ALG.Algorithm == labels[6])]

        a = pd.to_numeric(ALG1['Error'], errors='coerce')
        b = pd.to_numeric(ALG2['Error'], errors='coerce')
        c = pd.to_numeric(ALG3['Error'], errors='coerce')
        d = pd.to_numeric(ALG4['Error'], errors='coerce')
        e = pd.to_numeric(ALG5['Error'], errors='coerce')
        f = pd.to_numeric(ALG6['Error'], errors='coerce')
        g = pd.to_numeric(ALG7['Error'], errors='coerce')

        FINAL = [a, b, c, d, e, f, g]
        measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]
            
    # Boxplot de erro de reconstrução para todos datasets e um algoritmo específico
    elif algorithm != 'All' and dataset == 'All' and sparsityTax == 'All':
        labels = [algorithm]
        min_max_scaler = MinMaxScaler()

        ALG = data[(data.Algorithm == algorithm)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])
        
        x = pd.to_numeric(ALG['Error'], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]
            
    # Boxplot de erro de reconstrução para um dataset específico e um algoritmo específico
    elif algorithm != 'All' and dataset != 'All' and sparsityTax == 'All':
        labels = [algorithm]
        min_max_scaler = MinMaxScaler()

        ALG = data[(data.Algorithm == algorithm) & (data.DatasetName == dataset)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])
            
        x = pd.to_numeric(ALG['Error'], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]

    # Boxplot de erro de reconstrução para todos os datasets, esparsidade específica e quebrado por algoritmo
    elif algorithm == 'All' and dataset == 'All' and sparsityTax != 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']
        
        min_max_scaler = MinMaxScaler()
        ALG = data[(data.SparsityTax == sparsityTax)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])
        
        ALG1 = ALG[(ALG.Algorithm == labels[0])]
        ALG2 = ALG[(ALG.Algorithm == labels[1])]
        ALG3 = ALG[(ALG.Algorithm == labels[2])]
        ALG4 = ALG[(ALG.Algorithm == labels[3])]
        ALG5 = ALG[(ALG.Algorithm == labels[4])]
        ALG6 = ALG[(ALG.Algorithm == labels[5])]
        ALG7 = ALG[(ALG.Algorithm == labels[6])]

        a = pd.to_numeric(ALG1['Error'], errors='coerce')
        b = pd.to_numeric(ALG2['Error'], errors='coerce')
        c = pd.to_numeric(ALG3['Error'], errors='coerce')
        d = pd.to_numeric(ALG4['Error'], errors='coerce')
        e = pd.to_numeric(ALG5['Error'], errors='coerce')
        f = pd.to_numeric(ALG6['Error'], errors='coerce')
        g = pd.to_numeric(ALG7['Error'], errors='coerce')

        FINAL = [a, b, c, d, e, f, g]
        measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]

    # Boxplot de erro de reconstrução para um dataset específico, quebrado por algoritmo
    elif algorithm == 'All' and dataset != 'All' and sparsityTax != 'All':

        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        labels = ['KMeans', 'NBVD', 'ONM3F', 'ONMTF', 'OvNMTF', 'FNMTF', 'BinOvNMTF']
        
        min_max_scaler = MinMaxScaler()
        ALG = data[(data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])

        ALG1 = ALG[(ALG.Algorithm == labels[0])]
        ALG2 = ALG[(ALG.Algorithm == labels[1])]
        ALG3 = ALG[(ALG.Algorithm == labels[2])]
        ALG4 = ALG[(ALG.Algorithm == labels[3])]
        ALG5 = ALG[(ALG.Algorithm == labels[4])]
        ALG6 = ALG[(ALG.Algorithm == labels[5])]
        ALG7 = ALG[(ALG.Algorithm == labels[6])]

        a = pd.to_numeric(ALG1['Error'], errors='coerce')
        b = pd.to_numeric(ALG2['Error'], errors='coerce')
        c = pd.to_numeric(ALG3['Error'], errors='coerce')
        d = pd.to_numeric(ALG4['Error'], errors='coerce')
        e = pd.to_numeric(ALG5['Error'], errors='coerce')
        f = pd.to_numeric(ALG6['Error'], errors='coerce')
        g = pd.to_numeric(ALG7['Error'], errors='coerce')

        FINAL = [a, b, c, d, e, f, g]
        measureList = [a.mean(), b.mean(), c.mean(), d.mean(), e.mean(), f.mean(), g.mean()]

    # Boxplot de erro de reconstrução para todos datasets e um algoritmo específico
    elif algorithm != 'All' and dataset == 'All' and sparsityTax != 'All':
        labels = [algorithm]
        min_max_scaler = MinMaxScaler()
        
        ALG = data[(data.Algorithm == algorithm) & (data.SparsityTax == sparsityTax)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])
        
        x = pd.to_numeric(ALG['Error'], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]
            
    # Boxplot de erro de reconstrução para um dataset específico e um algoritmo específico
    elif algorithm != 'All' and dataset != 'All' and sparsityTax != 'All':
        labels = [algorithm]
        min_max_scaler = MinMaxScaler()
        
        ALG = data[(data.Algorithm == algorithm) & (data.DatasetName == dataset) & (data.SparsityTax == sparsityTax)]
        ALG[["Error"]] = min_max_scaler.fit_transform(ALG[["Error"]])
        
        x = pd.to_numeric(ALG['Error'], errors='coerce')
        FINAL = [x]
        
        meanpointprops = dict(marker='8', markeredgecolor='lightgreen', markerfacecolor='darkgreen')
        measureList = [x.mean()]
        
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    ax1.set_title(titulo)
    ax1.set_xlabel('Algoritmo')
    ax1.set_ylabel('Erro')
    
    bp = ax1.boxplot(FINAL, labels=labels, vert=True, showmeans=True,  meanprops=meanpointprops)
    
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='8') # . , o 8 * + x 
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_axisbelow(True)
    
    ax1.set_xlim(0.5, len(labels) + 0.5)
    top = 1.2
    bottom = -0.2
    ax1.set_ylim(bottom, top)
    
    pos = np.arange(7) + 1
    upper_labels = [str(np.round(s, 2)) for s in measureList]

    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], .95, upper_labels[tick], transform=ax1.get_xaxis_transform(), horizontalalignment='center', color='darkgreen')

    plt.savefig('Boxplot-'+algorithm+'-'+dataset+'-'+sparsityTax+'-Erro.PNG', format='png', dpi = 1000)


def barPlotGraphARI(data, ARI, tax):
# data = dataframe pandas contendo carga de CSV
# ARI = 'linha', 'coluna'
# tax = 0, 0.1, 0.2, ..., 0.9, 1.0

    if ARI == 'linha':
        if tax == 1:
            titulo = 'ARI Linhas = '+str(tax)
        else:
            titulo = 'ARI Linhas >= '+str(tax)
        ARI = 'ARIRow'

        countData = deepcopy(pd.pivot_table(data, index=['DatasetName'], columns='Algorithm', values='KValue', aggfunc=np.count_nonzero))
        
        data['ARIRow'] = pd.to_numeric(data['ARIRow'], errors='coerce')
        data['ARIRow_True'] = 0
        data.loc[:, 'ARIRow_True'] = data.ARIRow.apply(lambda x: 1 if x >= tax else 0)
        
        sumData = deepcopy(pd.pivot_table(data, index=['DatasetName'], columns='Algorithm', values='ARIRow_True', aggfunc='sum', fill_value=0))
        dataAgg = (sumData / countData).fillna(0)
    
    elif ARI == 'coluna':
        if tax == 1:
            titulo = 'ARI Colunas = '+str(tax)
        else:
            titulo = 'ARI Colunas >= '+str(tax)
        ARI = 'ARICol'

        countData = deepcopy(pd.pivot_table(data, index=['DatasetName'], columns='Algorithm', values='KValue', aggfunc=np.count_nonzero))

        data['ARICol'] = pd.to_numeric(data['ARICol'], errors='coerce')
        data['ARICol_True'] = 0
        data.loc[:, 'ARICol_True'] = data.ARICol.apply(lambda x: 1 if x >= tax else 0)

        sumData = deepcopy(pd.pivot_table(data, index=['DatasetName'], columns='Algorithm', values='ARICol_True', aggfunc='sum', fill_value=0))
        dataAgg = (sumData / countData).fillna(0)

    elif ARI == 'All':
        if tax == 1:
            titulo = 'ARI Linhas = '+str(tax)+' + ARI Colunas = '+str(tax)
        else:
            titulo = 'ARI Linhas >= '+str(tax)+' + ARI Colunas >= '+str(tax)

        countData = deepcopy(pd.pivot_table(data, index=['DatasetName'], columns='Algorithm', values='KValue', aggfunc=np.count_nonzero))

        data['ARIRow'] = pd.to_numeric(data['ARIRow'], errors='coerce')
        data['ARICol'] = pd.to_numeric(data['ARICol'], errors='coerce')
        data = data[(data['ARIRow'] >= tax) & (data['ARICol'] >= tax)]
        data['ARIRowCol_True'] = 1.0

        sumData = deepcopy(pd.pivot_table(data, index=['DatasetName'], columns='Algorithm', values='ARIRowCol_True', aggfunc='sum', fill_value=0))
        dataAgg = (sumData / countData).fillna(0)

    barWidth = 0.125
    datasets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    if ARI == 'ARIRow':
        bars3 = dataAgg.iloc[:,2]
        bars2 = dataAgg.iloc[:,1]
        bars1 = dataAgg.iloc[:,0]
        bars4 = dataAgg.iloc[:,3]
        bars7 = dataAgg.iloc[:,6]
        bars5 = dataAgg.iloc[:,4]
        bars6 = dataAgg.iloc[:,5]
    else:
        bars2 = dataAgg.iloc[:,1]
        bars1 = dataAgg.iloc[:,0]
        bars4 = dataAgg.iloc[:,3]
        bars7 = dataAgg.iloc[:,6]
        bars5 = dataAgg.iloc[:,4]
        bars6 = dataAgg.iloc[:,5]
    
    r1 = np.arange(len(datasets))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]
    r8 = [x + barWidth for x in r7]
    r9 = [x + barWidth for x in r8]
    
    plt.figure(figsize=(10, 5))
    if ARI == 'ARIRow':
        plt.bar(r1, bars3, color='darkgoldenrod', width=barWidth, edgecolor='white', label='KMeans')
        plt.bar(r2, bars2, color='darkorange', width=barWidth, edgecolor='white', label='FNMTF')
        plt.bar(r3, bars1, color='maroon', width=barWidth, edgecolor='white', label='BinOvNMTF')
        plt.bar(r4, bars4, color='olivedrab', width=barWidth, edgecolor='white', label='NBVD')
        plt.bar(r5, bars7, color='indigo', width=barWidth, edgecolor='white', label='OvNMTF')
        plt.bar(r6, bars5, color='darkgreen', width=barWidth, edgecolor='white', label='ONM3F')
        plt.bar(r7, bars6, color='steelblue', width=barWidth, edgecolor='white', label='ONMTF')
    else:
        plt.bar(r2, bars2, color='darkorange', width=barWidth, edgecolor='white', label='FNMTF')
        plt.bar(r3, bars1, color='maroon', width=barWidth, edgecolor='white', label='BinOvNMTF')
        plt.bar(r4, bars4, color='olivedrab', width=barWidth, edgecolor='white', label='NBVD')
        plt.bar(r5, bars7, color='indigo', width=barWidth, edgecolor='white', label='OvNMTF')
        plt.bar(r6, bars5, color='darkgreen', width=barWidth, edgecolor='white', label='ONM3F')
        plt.bar(r7, bars6, color='steelblue', width=barWidth, edgecolor='white', label='ONMTF')
    
    plt.xlabel('Conjuntos de dados')
    plt.ylabel('Taxa de acerto')
    plt.xticks(np.arange(len(datasets)) + 3 * barWidth, datasets)

    plt.style.use("seaborn-whitegrid")
    line = np.linspace(0, 1)
    
    plt.plot([0.875,0.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([1.875,1.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([2.875,2.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([3.875,3.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([4.875,4.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([5.875,5.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([6.875,6.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)
    plt.plot([7.875,7.875], [0,1], 'b--', color='midnightblue', linewidth=0.7)

    plt.grid(which='major', axis='x', linestyle='', c='white', linewidth=0.5)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylim(0,1)
    plt.xlim(-0.125, 8.875)
    plt.legend(loc='best', frameon=True)
    plt.title(titulo)
    plt.savefig('BarPlot-'+ARI+'-'+str(tax)+'.PNG', format='png', dpi=1000)
    plt.show()

