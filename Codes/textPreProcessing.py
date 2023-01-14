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

import os
import re
import math
import unidecode

from copy import deepcopy

# Método que recebe como parâmetro um diretório com textos e retorna um dataset
# Criado para o conjunto de dados da Empresa Brasil de Comunicação - EBC
# Necessário adaptações para outros corpus
def loadTexts(sourcepath):
    dataset = []
    for filename in os.listdir(sourcepath):
        file = open(os.path.join(sourcepath, filename), 'r')
        text = file.read().strip()
        file.close()

        # Recupera o código do canal de notícias
        file_channel = re.findall('<channelcode>(.*)</channelcode>', text, re.DOTALL)
        # Recupera o título da notícia
        file_title = re.findall('<newstitle>(.*)</newstitle>', text, re.DOTALL)
        # Recupera o texto da notícia
        file_text = re.findall('<newstext>(.*)</newstext>', text, re.DOTALL)

        # Cria um dataset com "Nome do arquivo", "Código do canal", "Título da notícia" e "Notícia"
        for j in range(len(file_channel)):
            dataset.append([filename, str(file_channel[j]), file_title[j], file_text[j]])

    return dataset

# Método que recebe como parâmetro um texto "bruto" e retorna um texto ajustado
# Também recebe como parâmetro uma lista de stopwords e stemmer (SnowballStemmer)
def processTexts(text, params):
    newText = deepcopy(text)
    for i in range(len(newText)):
		# Retira quebra de linha no final do texto
        newText[i][3] = newText[i][3].rstrip('\n') 
		# Retira acentuação das palavras
        newText[i][3] = unidecode.unidecode(newText[i][3]) 
		# Retira pontuação dos números
        newText[i][3] = re.sub(r'(?<=\d)\.(?=\d)', '', newText[i][3]) 
		# Converte valores de hora para "hora"
        newText[i][3] = re.sub(r'(([0-1]?[0-9]|2[0-3]):)?[0-5]?[0-9]h', 'hora', newText[i][3]) 
		# Converte valores monetários para "moeda"
        newText[i][3] = re.sub(r'[RSU][RSU]*\$[ 1-9]*\d*(\.\d\d\d)*(,\d\d)?', 'moeda', newText[i][3]) 
		# Converte valores numéricos para "numerico"
        newText[i][3] = re.sub(r'(\d+)', 'numerico', newText[i][3]) 
        if params[0]:
			# Reduz os caracateres para minúsculo
            newText[i][3] = newText[i][3].lower()  
		# Tokenização
        listOfTokens = tokenize(newText[i][3])  
		# Stopwords
        listOfTokens = removeStopWords(listOfTokens, params[1])
		# Stemmer		
        # listOfTokens = applyStemming(listOfTokens, params[2]) 
        newText[i][3] = listOfTokens
    return newText


def defineWordSet(text):
    wordSet = set()
    for i in range(len(text)):
        wordSet = wordSet.union(set(text[i][3]))
    return wordSet


def createDict(text, wordSet):
    wordDict = {}
    for i in range(len(text)):
        wordDict[text[i][0]] = dict.fromkeys(wordSet, 0)
    return wordDict


def updateDict(text, wordDict):
    for i, j in zip(text, range(len(text))):
        for word in text[j][3]:
            wordDict[i[0]][word] += 1
    return wordDict


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


def createTFDict(text, wordDict):
    for i, j in zip(text, range(len(text))):
        wordDict[i[0]] = computeTF(wordDict[i[0]], text[j][3])
    return wordDict


def createIDFDict(textList):
    # idfDict = {}
    N = len(textList)

    idfDict = dict.fromkeys(textList[max(textList.keys())].keys(), 0)

    for i in textList:
        for word, val in textList[i].items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


def createTFIDFDict(TFBow, idfs):
    tfidfDict = {}
    for i in TFBow:
        tfidfDict[i] = {}
        for word, val in TFBow[i].items():
            tfidfDict[i][word] = val * idfs[word]
    return tfidfDict


def foldCase(sentence, parameter):
    if parameter:
        sentence = sentence.lower()
    return sentence


def tokenize(sentence):
    sentence = sentence.replace('_', ' ')
    return filter(None, re.split(r'\W+', sentence))


def removeStopWords(listOfTokens, listOfStopWords):
    return [token for token in listOfTokens if token not in listOfStopWords]


def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]


def geraSaidaARIRow(text):
    saida = []
    for i in range(len(text)):
        saida.append(text[i][1])
    return saida
