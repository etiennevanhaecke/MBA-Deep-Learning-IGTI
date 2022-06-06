import numpy as np
import os
import glob
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from PIL import UnidentifiedImageError
from keras.models import load_model

import matplotlib.pyplot as plt

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk

import math

import constants as c

import mtcnn
from mtcnn.mtcnn import MTCNN

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

#Para serializar e deserializar os modelos ML treinados
import pickle

import sys

import time

#Funcoes gerais para montar a lista de label de uma imagem (problema multi-label), codificar esta lista de label (gera o target da imagem),
# decodificar a target (gera a lista de label da imagem)

# Funcao de decodificaca do jogo de saida (target) en lista dos tags da foto
def oneHotDecode(tgt, dicIdTag):
    '''Retorna uma lista dos tags de ID correspondendo as posicoes do jogo de saida com 1'''
    #Recupera as posicoes preenchidas
    posId = np.nonzero(tgt)
    #Monta a lista dos tags da imagem
    lstTag = [dicIdTag[id] for id in posId[0]]
    lstId  = [id for id in posId[0]]
    #OBS: Precisa recuperar o indice 0 de posId prque np.nonzero retorna tuple de array de posicao nao 0, um elemento por dimensao do array
    
    # Retorno lista de tags e ID das classes reconhecidas
    return lstTag, lstId

#Funcao de carregamento e serializacao do jogo de entrada X (pixels das imagens)
def loadImg(maskNameFile="*", sizeImg=128, nomeRepertory="C:\\Users\\Utilisateur\\Downloads\\tmp\\MBA\\"):
    os.chdir(nomeRepertory)
    # Criacao das listas com as imagens e nome destas imagens
    lstImgs = []
    lstNameFiles = []
    gntImages = glob.iglob(f"{maskNameFile}", recursive=False)

    #Para cada glob encontrado
    for pth in gntImages:
        try:
            #Se tratar-se bem de um arquivo
            if os.path.isfile(pth):
                nameFile = os.path.basename(pth)
                #print(nameFile)
                # Visualizacao da imagem sem transformacao
                #img = load_img(fr'{pth}')
                #plt.imshow(img)
                #plt.show()
                #plt.pause(1) 
                # Dados dos pixels desta imagem de tamanho limitado para economizar memoria
                img = image.load_img(fr'{pth}', target_size=(sizeImg,sizeImg))
                # convert to numpy array
                img = image.img_to_array(img)
                #OBS: No exemplo deep_learning_for_computer_vision.pdf, pagina 334, esta indicado de usar dtype uint8, mas como 
                #     o predict sobre os modelos pretreinados sem ajuste, nao foi bem succedido, se deixou os valores originais.
                
                # Adicao da imagem e do nome arquivo as listas
                lstImgs.append(img)
                lstNameFiles.append(nameFile)
        except UnidentifiedImageError:
            print(f'Warning: O arquivo {nameFile} nao pode estar carregado porque nao esta reconhecido como imagem.')
        except OSError:
            print(f'Warning: O arquivo {nameFile} nao pode estar carregado porque gera um erro OS a pesar estar reconhecido como imagem.')
        except:
            print(nameFile)
            print("Unexpected error:", sys.exc_info()[0])
            raise
            
    #Se passa as imagens e nomes arquivo em array numpy
    X = np.asarray(lstImgs)
    npNameFiles = np.asarray(lstNameFiles)

    #Se salva estes arrays em um arquivo compressado de padrao numpy
    np.savez_compressed('X_TestPredict.npz', X, npNameFiles)  

# Funcao de calculo da metrica Fbeta para uma classificacao multi-classe/label
#Se cria a metrica fbeta para avaliar o treinamento do modelo (se vai usar a principio com o beta 1 para ter usar a metrica F1)
#OBS: A funcao nativa foi retirada do keras devido a gerar confusao em funcao do calculo estar realizado ao nivel batch ou por metrica
#     A avaliar se a funcao copiada de deep_learning_for_computer_vision.pdf (pagina 335) esta adequada ou se precisa de uma funcao
#     mais sofisticada como explicada no pdf F-beta Score in Keras Part III. Creating custom F-beta score for… _ by Jolomi Tosanwumi _ Towards Data Science
def fBetaKeyras(YTrue, YPred, beta=1):
    # clip as predicoes entre 0 e 1 (inferior a 0 passa 0 e superior a 1 passa 1)
    YPred = backend.clip(YPred, 0, 1)
    # calcula os elementos que fazem parte do calculo do indicador FBeta
    tp = backend.sum(backend.round(backend.clip(YTrue * YPred, 0, 1)), axis=1)  #Verdadeiros Positivos
    fp = backend.sum(backend.round(backend.clip(YPred - YTrue, 0, 1)), axis=1)  #Falsos Positivos
    fn = backend.sum(backend.round(backend.clip(YTrue - YPred, 0, 1)), axis=1)  #Falsos Negativos
    # Calculo da acuracidade (precisao)
    p = tp / (tp + fp + backend.epsilon())
    # Calculo do recall
    r = tp / (tp + fn + backend.epsilon())
    # Calculo do FBeta, com a media de cada classe
    bb = beta ** 2
    fBetaScore = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    #OBS: Uso de uma constante infinitessima epsilon baixa para nao cair no erro de divisao por zero
        
    return fBetaScore

def loadModels(pathModel = "C:\\Users\\Utilisateur\\Downloads\\tmp\\MBA\\models\\"):
    # Se deserializa o modelo
    #Modelo com jogo de treinamento limitado a 70%, que apresentou a melhor metrica para o jogo de teste
    #model = load_model('modelVGG16DataAugmFineTuning2.h5', custom_objects={'fBetaKeyras':fBetaKeyras})

    #Modelo com jogo de treinamento completo, que treinou com as mesmas tecnicas que o modelo anterior usando callback para 
    # conservar os pesos da melhor epoca e para ter uma fim antecipada depois de 30 epocas sem melhora da metrica
    #Definicao do repertorio home do projet
    os.chdir(pathModel)

    #Modelo das classes comumns previamente treinado
    modelClas = load_model('modelVGG16DataAugmFineTuning2Full.h5', custom_objects={'fBetaKeyras':fBetaKeyras})
    #model.summary()
    
    # Compilacao do modelo para poder usar evaluate
    #opt = SGD(learning_rate=0.01, momentum=0.9)
    #from keras.optimizers import SGD
    #modelClas.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fBetaKeyras])
    modelClas.compile(loss='binary_crossentropy', metrics=[fBetaKeyras])
    print(f'Loaded: Modelo de predicao das classes comuns')
    
    #print(f'Loaded: Modelo de classificacao pessoas')
    
    return modelClas, 1 #modelPes

    
def predictFotos(lstIDClasSel, dirNameSel, valOpcaoFind, modelClas, valLimProb, progFindFoto):
    limProb = valLimProb / 100.
    print(f'Lista ID classes pesquisadas {lstIDClasSel} com opcao {valOpcaoFind} e fracao limite probabilidade de {limProb}')
    
    #Dico a retornar com os dados das fotos de classe predita correspondendo a uma das classes selecionadas pelo usuario
    dicImgFind = dict()
    
    #Lista dos ID selecionadas pel usuario encontradas nas fotos em funcao da opcao OR (qualquer) ou AND (todas)
    lstIDFind = list()
      
    #Se cria o arquivo de compressao com a mascara de imagem e o tamanho de imagens desejados
    #OBS: A descomentar so na primeira execucao, ja que depois se pode ler diretamente do arquivo criado na primeira execucao
    loadImg(maskNameFile="*", sizeImg=c.sizeImg, nomeRepertory=dirNameSel)
    #Se recarrega o jogo de entrada e saida a parte do arquivo de compressao numpy
    data = np.load('X_TestPredict.npz')
    X, npNameFiles = data['arr_0'], data['arr_1']
    print(f'Loaded: X: {X.shape}, Y: npNameFiles: {npNameFiles.shape}\n')
    
    #Predicao com este modelo de todo o jogo de foto selecionado 
    
    # Se centraliza manualmente os dados dos 3 camais de cor dos pixels
    XPrep = X - [123.68, 116.779, 103.939]

    #Predicao com o modelo
    #YPred = modelClas.predict_generator(iterator) is deprecated and will be removed in a future version.
    YPred = modelClas.predict(XPrep)
    
    #Frequencia de fotos preditas para aumento de uma step a barra de progressao. 
    #Definida em funcao do total de fotografias a predir e do total de step de progressao configurado
    freqProg = math.ceil(len(YPred) / c.totStepProg)
    
    #Loop sobre todas as predicoes para apontar as fotos com uma classe predita (considerando um limite de probabilidade cnfigurado)
    # correspondendo a uma das classes selecionadas na aplicacao
    for idx in range(len(YPred)): #range(160, 180)
        #Lista IDs das classes preditas com uma probabilidade superior ao limite configurado em ratio
        lstIDPred = [i for i in range(len(YPred[idx])) if YPred[idx][i] > limProb]
        #Lista %probaabilidade das classes preditas com uma probabilidade superior ao limite configurado em ratio
        lstPercPred = [str(round(YPred[idx][i]*100, 2))+'%' for i in range(len(YPred[idx])) if YPred[idx][i] > limProb]
        #Dico entre as 2 listas para conservar o %predicao por ID classe
        dicIDPercPred = dict(zip(lstIDPred, lstPercPred)) 
        
        #Cruzamento em funcao da opcao de pesquisa selecionada pelo usuario
        if valOpcaoFind == 'or':
            #Se verifica se alguma das classes preditas correspondem as classes selecionadas pelo usuario (cruzando as duas listas de ID)
            lstIDFind = list(set(lstIDClasSel) & set(lstIDPred))
        if valOpcaoFind == 'and':
            #A totalidade das classes selecionadas pelo usuario devem haver sido preditas pelo modelo
            print(f'set lstIDClasSel {set(lstIDClasSel)} - set {set(lstIDPred)}')
            if set(lstIDClasSel) - set(lstIDPred) == set(): 
                lstIDFind = lstIDClasSel
            else:
                lstIDFind = list()
        
        #Se alguma classe foi achada
        if (len(lstIDFind) > 0):
            nameFile = npNameFiles[idx] 
            print(f'- Foto {nameFile} com uma lista de classes preditas correspondendo as classes pesquisadas {lstIDFind}: ')
            lstPercFind = list()
            for i, idxClasse in enumerate(lstIDFind):
                percPred = dicIDPercPred[idxClasse] #Percentual de probabilidade desta classe
                print(f'-> Classe ID {idxClasse} tag {c.dicIdTag[idxClasse]} com a probabilidade {percPred}')
                lstPercFind.append(percPred)   #Lista percentual probabilidade de cada ID de classe predita correspondendo a uma classe selecionada
            
            #Visualizacao da imagem sem transformacao
            #print(f'---> Apresentaca da foto presente no caminho {dirNameSel}/{nameFile}')
            #img = load_img(f'{dirNameSel}/{nameFile}') #que faz fr' como estava antes...
            #plt.imshow(img)
            #plt.show()
            
            #Se adiciona os dados da foto ao dico de foto com uma classe predita correspondendo a uma classe selecionada pelo usuario
            # A chave esta o nome do arquivo foto e o valor o tuple da lista dos ID de classe preditas batendo com a lista correspondente de %predicao
            dicImgFind[nameFile] = (lstIDFind, lstPercFind)
                
            #Criar um label com a foto no frame complementar
        
            #lstClasPred = [(dicIdTag[i], str(round(YPred[idx][i]*100, 2))+'%') for i in range(len(YPred[idx])) if YPred[idx][i] > limProb]
    
        #Se tive predicao da frequencia de fotos calculada para avancar a barra de progressao desde o procedente refresh
        if (idx%freqProg == 0):
            progFindFoto.step()
    
    #Retorno do dico das fotos com uma classe predita correspndendo a uma selecionada pelo usuario
    return dicImgFind

#Apresentacao destas fotos no frame separado topFoto com indicacao das classes encontradas e a probabilidade de predicao destas classes
# tipoClasse indica se dicImgFind tem uma lista de ID (valor default 1) ou de label das classes achadas nas fotos a apresentar 
def showFotosFind(dicImgFind, cnvTopFoto2, dirNameSel, valNumCol, valWidthImg, valHeightImg, tipoClasse=1):
    
    totImg = len(dicImgFind)
    
    global lstImg
    #Lista das imagens achadas e apresentadas no canvas de topFind2
    lstImg = list()
    
    for i, key in enumerate(dicImgFind.keys()):
        nameFile = key
        lstIDFind = dicImgFind[key][0]
        
        #Se foi passado uma lista de ID de classe
        if (tipoClasse == 1):
            #Se ID de temas comuns
            if (dicImgFind[key][0] != [-1]):
                lstClasFind = [c.dicIdTag[idx] for idx in dicImgFind[key][0]]
            #Senao se trata de rosto
            else:
                lstClasFind = ['rosto']
        else:
            lstClasFind = dicImgFind[key][0]
            
        lstPercFind = dicImgFind[key][1]
        #txtImg = f'{nameFile}:\n{lstClasFind}-{lstPercFind}'
        txtImg = f'{nameFile}:\n{tuple(zip(lstClasFind, lstPercFind))}'
        #Criacao de um novo label para apresentar a foto
        img=Image.open(f'{dirNameSel}/{nameFile}')
        #img.thumbnail((300,300),Image.ANTIALIAS)
        img.thumbnail((valWidthImg, valHeightImg))
        img2 = ImageTk.PhotoImage(img)
        
        #Precisa guardar a referencia da imagem em uma variavel global para nao perder as imagens saindo da funcao
        lstImg.append(img2)
        
        #Calculo do posicionamento da foto no grid
        rw = math.trunc(i / valNumCol)
        cl = i % valNumCol
        
        #Adicao no topFoto
        #print(f'{dirNameSel}/{nameFile}')
        #labImage = ttk.Label(topFoto, image=img2, compound='top', text=nameFile, anchor=tk.W)
        #labImage.grid(column=cl, row=rw)        
        
        #Adicao no topFoto2 definindo a posicao em pixels a parte posicao no grid
        width  = c.spaceImgFind*2 + cl*(valWidthImg+c.spaceImgFind)
        height = c.spaceImgFind*2 + rw*(valHeightImg+c.spaceImgFind)
        
        print(f'Posicao no topFoto2: width {width} - height {height}')
        #cnvTopFoto2.create_image(x=cl*valWidthImg, y=rw*valHeightImg, image=img2, anchor='nw') #tags=nameFile
        cnvTopFoto2.create_image(width, height, image=img2, anchor='nw', tags=(nameFile, 'image'))
        cnvTopFoto2.create_text(width, height, text=txtImg, anchor='sw', tags=(nameFile, 'text'))
        #OBS: Da erro de tuple quando se indica as posicoes x e y, em 1era e segunda posicao, usando o nome dos parametros x e y...

    
# extract the faces from a given photograph
def extracaoRostos(modelDet, nameFile, dataImg, dirNameFace, limProb, sizeRosto = (224, 224) ):
    print(f'-> Inicio extracaoRostos a {time.ctime()}')
    lstDicRosto = list()
    dicFaceSave = dict()
    
    # Detecao dos rostos a parte dos dados da imagem passada em parametro
    #print(f'Dentro extracaoRostos com {nameFile}' )
    lstResultRostos = modelDet.detect_faces(dataImg)
    #print(len(lstResultRostos))
    
    for i, resultRosto in enumerate(lstResultRostos):
        #print(resultRosto)
        #Se tem uma boa probabilidade que o pedaco de imagem separado seja um rosto
        #print(resultRosto['confidence'])
        #test = resultRosto['confidence']
        #print(test)
        #print(type(test))
        #print(f'limProb: {limProb}')
        if resultRosto['confidence'] > limProb:
            #Recuperacao das coordenadas da area do rosto identificado
            x1, y1, width, height = resultRosto['box']
            x2, y2 = x1 + width, y1 + height
            #Extracao dos dados do rosto
            dataRosto = dataImg[y1:y2, x1:x2]
            #Se redimensiona o rosto com o tamanho passado em parametro
            # Para isso, se passa no formato de imagem
            imgRosto = Image.fromarray(dataRosto)
            # e se redimensiona a parte do formato imagem
            imgRosto = imgRosto.resize(sizeRosto)
            # Volta em array da imagem de rosto redimensionada
            dataRosto = np.asarray(imgRosto)
            
            #Dicionario do rosto com os dados e as coordenadas
            dicRosto = {'data': dataRosto, 'coord': resultRosto['box']} 
            
            #Se adiciona a lista de retorno da funcao
            lstDicRosto.append(dicRosto) 
            
            #Se o path do repertorio de conservacao dos rostos esta passado em parametro a funcao, 
            # se salva os dados do rosto em um arquivo de imagem de nome derivado do nome da foto original
            if dirNameFace is not None:
                nomeSemExt = os.path.splitext(nameFile)[0]
                #nomeSemExt
                extNome = os.path.splitext(nameFile)[1]
                #extNome
                newName = nomeSemExt+'_FACE'+str(i)+extNome  #Nome sem patch
                newNameFull = dirNameFace+'/'+newName        #Nome com patch
                #newName
                plt.imsave(newNameFull, dataRosto)
                
                #Conservacao em um dico de mesma estrutura que o usado para pesquisar as fotos de temas de comuns
                dicFaceSave[newName] = ([-1] , [resultRosto['confidence']])
                #OBS: ID de classe -1 vai estar reconhecido como rosto na funcao showFotosFind

    print(f'-> Fim extracaoRostos a {time.ctime()}')
    return dicFaceSave, lstDicRosto
    
def orientacaoImagem(img):
    #Recuperacao dos diferentes tags da foto
    exifs = img.getexif()
    #Se a foto foi tirada como um retrato (verificado a traves o tag oficial de orientacao de ID 274)
    if (274 in exifs and exifs[274] == 6):
        #Rotacao de -90 graus para a foto estar no sentido paisagem, necessario para identificar os rostos
        print('-> Rotacao de -90 graus')
        #img.show()
        img = img.rotate(-90)
        #img.show()
    if (274 in exifs and exifs[274] == 8):
        #Rotacao de 90 graus para a foto estar no sentido paisagem, necessario para identificar os rostos
        print('-> Rotacao de +90 graus')
        #img.show()
        img = img.rotate(90)
        #img.show()
    if (274 in exifs and exifs[274] == 3):
        #Rotacao de 180 graus para a foto estar no sentido paisagem, necessario para identificar os rostos
        print('-> Rotacao de +180 graus')
        #img.show()
        img = img.rotate(180)
        #img.show()

    #Retorno da imagem que passou eventualmente por uma rotacao para estar no bom sentido para a identificacao dos rostos
    return img

def predictFaces(dirNameSel, dirNameFace, valLimProb, progFindFoto):
    
    #Carga do modelo de reconhecimento de rosto
    modelDet = MTCNN()  #Modelo ja treinado para reconhecer os rostos (sem classificar eles) em uma foto
    print(f'Created: Modelo de identificacao de rosto')
    
    limProb = valLimProb / 100.
    print(f'Rostos pesquisados com fracao limite probabilidade de {limProb}')
    
    os.chdir(dirNameSel)
    maskNameFile = '*'
    pathFiles = glob.iglob(f"{maskNameFile}", recursive=False)
    
    dicFaceSaveTot = dict()

    for pathFile in pathFiles:
        try:
            #Se tratar-se bem de um arquivo
            if os.path.isfile(pathFile):
                #Nome do arquivo
                nameFile = os.path.basename(pathFile)
                #print(nameFile)

                #Tentativa de carga do arquivo como imagem
                img = Image.open(nameFile)

                #Eventual rotacao da imagem para estar no bom sentido para a identificacao dos rostos
                img = orientacaoImagem(img)

                #Transformacao em array para passar ao extrator de rostos
                dataImg = np.asarray(img)

                #Extracao dos rostos da imagem e armazenamento no storage
                dicFaceSave, _ = extracaoRostos(modelDet, nameFile, dataImg, dirNameFace, limProb, sizeRosto = (160, 160) )
                
                #Se junta o dico dos rostos achados
                dicFaceSaveTot.update(dicFaceSave)

        except UnidentifiedImageError:
            print(f'Warning: O arquivo {nameFile} nao pode estar carregado porque nao esta reconhecido como imagem.')
        except OSError:
            print(f'Warning: O arquivo {nameFile} nao pode estar carregado porque gera um erro OS a pesar estar reconhecido como imagem.')
        except:
            print(nameFile)
            print("Unexpected error:", sys.exc_info()[0])
            raise
    
    #Retorno do dico dos rostos achados em todas as fotos
    return dicFaceSaveTot

def loadRostos(dirRosto):
    print('---> loadRostos')
    lstDataRosto = list()
    for nameFileRosto in os.listdir(dirRosto):
        #print(nameFileRosto)
        dataRosto = plt.imread(f'{dirRosto}/{nameFileRosto}')
        
        lstDataRosto.append(dataRosto)
    print(f'len lstDataRosto: {len(lstDataRosto)}')
    
    return lstDataRosto

# Carga do dataset de rosto correspondendo ao diretorio, formado de um sub-diretorio por classe de rosto, com as imagens desta classe
def loadDatasetRostos(dirDataset):
    print('--> loadDatasetRostos')
    X, Y = list(), list()
    print(dirDataset)
    # Loop sobre cada sub-repertorio, um por classe
    for subDir in os.listdir(dirDataset):
        # path
        path = dirDataset + '/' + subDir + '/'
        #print(path)
        # Bypass dos eventuais arquivos presentes no repertorio do dataset
        if not os.path.isdir(path):
            continue
        # Carga da lista dos dados de todos os rostos presentes no subdiretorio da classe
        lstDataRosto = loadRostos(path)
        # Criacao da lista dos labels destes rostos (= nome da subpasta para cada rosto achado nesta subpasta)
        lstLabelRosto = [subDir for _ in range(len(lstDataRosto))]
        # Log do progresso da carga
        print(f'-> Carga de {len(lstDataRosto)} exemplos da classe rosto {subDir}')
        # Conservacao nas listas para treinamento
        X.extend(lstDataRosto)
        Y.extend(lstLabelRosto)
    print(f'len X: {len(X)} - len Y: {len(Y)}')
    
    return np.asarray(X), np.asarray(Y)

# Calculo do vector "embedding" das features dos dados de um rosto
def getEmbedding(modelEmb, face_pixels):
    #print('--> getEmbedding')
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # Standardizacao dos valores dos pixels a traves os canais de cor (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Se adiciona uma dimensao para poder passar este jogo de pixels de um rosto em entrada do modelo "embedding"
    samples = np.expand_dims(face_pixels, axis=0)

    # Predicao do modelo Facenet afim de ter o vector de feature
    yhat = modelEmb.predict(samples)
    # Retorno do vector de feature que esta em primeira posicao
    return yhat[0]

def trainPersons(dirNameFace, progFindFoto, pathModel = "C:\\Users\\Utilisateur\\Downloads\\tmp\\MBA\\models\\"):
    print('-> trainPersons')
    #Se recupera o jogo completo de rostos, a parte das sub-pastas do repertorio de rostos
    X, Y = loadDatasetRostos(dirNameFace)
    
    #Se separa o jogo completo entre jogo de treinamento (80%) e jogo validacao (20%), respeitando a proporacao das diferentes classes de rosto
    XTrain, XVal, YTrain, YVal = train_test_split(X, Y, test_size=0.2, stratify=Y)
    print(f'XTrain: {len(XTrain)} - XVal: {len(XVal)} - YTrain: {len(YTrain)} - YVal: {len(YVal)}')
    
    #Se conserva em um arquivo compressado do repertorio dos rostos os datasets
    np.savez_compressed(f'{dirNameFace}/datasetRostos.npz', XTrain, YTrain, XVal, YVal)
    
    # Carga dos dados de rosto, do jogo de treinamento e de validacao, a parte deste arquivo de compressao
    datasetRostos = np.load(f'{dirNameFace}/datasetRostos.npz')
    XTrain, YTrain, XVal, YVal = datasetRostos['arr_0'], datasetRostos['arr_1'], datasetRostos['arr_2'], datasetRostos['arr_3']
    print('Loaded: ', XTrain.shape, YTrain.shape, XVal.shape, YVal.shape)
   
    # Carga do modelo facenet so no momento do treinamento que nao deveria acontecer mais que uma vez por sessao e pode nao acontecer
    #  para economizar o tempo desta carga caso nao tem treinamento
    modelEmb = load_model(f'{pathModel}facenet_keras.h5') #Modelo necessario para criar as features dos rostos a usar para
                                                     # o treinamento do modelo personalizado de classificacao dos rostos
    print('Modelo de embedding carregado')

    # Conversao dos dados brutos de cada rosto no seu vector "embedding" de features (128 valores)
    #  Jogo de treinamento
    XTrainNew = list()
    #  Para cada rosto do jogo
    for face_pixels in XTrain:
        #   Se recupera o seu vector embedding a parte do seu array de pixels
        embedding = getEmbedding(modelEmb, face_pixels)
        #   Adicao a lista de vector embedding
        XTrainNew.append(embedding)
    #  Se transforma a lista em um array de vector embedding
    XTrainNew = np.asarray(XTrainNew)
    print(XTrainNew.shape)
    
    #  Jogo de validacao
    XValNew = list()
    #  Para cada rosto do jogo
    for face_pixels in XVal:
        #   Se recupera o seu vector embedding a parte do seu array de pixels
        embedding = getEmbedding(modelEmb, face_pixels)
        #   Adicao a lista de vector embedding
        XValNew.append(embedding)
    #  Se transforma a lista em um array de vector embedding
    XValNew = np.asarray(XValNew)
    print(XValNew.shape)

    # Se salva em arquivo estes vectores de feature, em um formato compressado
    np.savez_compressed(f'{dirNameFace}/datasetRostosEmbeddings.npz', XTrainNew, YTrain, XValNew, YVal)
    
    # Carga a parte deste arquivo compressado do dataset dos vectores "embedding" de feature de rostos (jogo de treinamento e de validacao)
    data = np.load(f'{dirNameFace}/datasetRostosEmbeddings.npz')
    XTrain, YTrain, XVal, YVal = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: ', XTrain.shape, YTrain.shape, XVal.shape, YVal.shape)

    # Normalizacao dos jogos de entrada de treinamento 
    # (objeto de normalizacao calculado sobre o jogo de treinamento e aplicado sobre os 2 jogos, treinamento e validacao)
    encoderIn = Normalizer(norm='l2')
    encoderIn.fit(XTrain)
    XTrain = encoderIn.transform(XTrain)
    XVal = encoderIn.transform(XVal)

    # Codificacao de cada classe de pessoa a identificar (dicionario DE-PARA entre o ID passado ao modelo e o label descrevendo a classe da pessoa)
    encoderOut = LabelEncoder()
    encoderOut.fit(YTrain)
    YTrain = encoderOut.transform(YTrain)
    YVal = encoderOut.transform(YVal)

    # Carga e treinamento de um modelo Support Vector Machine
    modelPes = SVC(kernel='linear', probability=True)
    modelPes.fit(XTrain, YTrain)

    # Predicao do modelo SVC treinado sobre os jogos de treinamento e de validacao
    yhat_train = modelPes.predict(XTrain)
    yhat_val = modelPes.predict(XVal)
    
    # Acuracidade do modelo de classificacao sobre os 2 jogos
    score_train = accuracy_score(YTrain, yhat_train)
    score_val = accuracy_score(YVal, yhat_val)
    print('Accuracy: train=%.3f, val=%.3f' % (score_train*100, score_val*100))

    #Se salva o mdelo de classificacao SVM, os encoders do jogo de entrada (normalizacao) e de saida (DE-PARA ID-LABEL classes rosto)
    # Modelo SVM
    print(f'   -> Objeto SVM Modelo resultando do treinamento:\n      {modelPes}')
    #  Serializacao do modelo em arquivo da pasta models
    with open(f"{pathModel}modelSVC.sav", 'wb') as writeFile:
        pickle.dump(obj=modelPes, file=writeFile)
    
    # Encoder do jogo de entrada (normalizacao)
    print(f'   -> Objeto encoder IN do jogo de treinamento:\n      {encoderIn}')
    #  Serializacao do encoder IN em arquivo da pasta models
    with open(f"{pathModel}encoderIn.sav", 'wb') as writeFile:
        pickle.dump(obj=encoderIn, file=writeFile)
    
    # Encoder do jogo de saida (de-para id-label classes rostos)
    print(f'   -> Objeto encoder OUT do jogo de treinamento:\n      {encoderOut}')
    #  Serializacao do encoder OUT em arquivo da pasta models
    with open(f"{pathModel}encoderOut.sav", 'wb') as writeFile:
        pickle.dump(obj=encoderOut, file=writeFile)
    
    return 'modelSVC.sav', pathModel, round(score_train*100, 2), round(score_val*100, 2)


def getModelos(modelDetect=None, modelEmbed=None, modelClassif = None, encoderIn = None, encoderOut = None,
              pathModel = "C:\\Users\\Utilisateur\\Downloads\\tmp\\MBA\\models\\"):
    print(f'-> Inicio getModelos a {time.ctime()}')
    os.chdir(pathModel)
    
    #  Modelo para detacao dos rostos na fotografia
    if modelDetect is None:
        modelDetect = MTCNN()
    #  Modelo para embedding (=serie de 128 features) dos rostos idenficados na foto
    if modelEmbed is None:
        modelEmbed = load_model('facenet_keras.h5')
    #  Modelo para classificacao dos rostos a parte do enbedding (=serie de 128 features) deles
    if modelClassif is None:
        #  Deserializacao do objeto
        with open(f"modelSVC.sav", 'rb') as readFile:
            modelClassif = pickle.load(file=readFile)    
    #  Encoder do jogo de entrada no modelo SVC de classificacao (normalizacao do jogo de treinamento) 
    if encoderIn is None:
        #  Deserializacao do objeto
        with open(f"encoderIn.sav", 'rb') as readFile:
            encoderIn = pickle.load(file=readFile)  
    #  Encoder do jogo de saida no modelo SVC de classificacao (DE-PARA ID-LABEL classes rosto treinados) 
    if encoderOut is None:
        #  Deserializacao do objeto
        with open(f"encoderOut.sav", 'rb') as readFile:
            encoderOut = pickle.load(file=readFile)  
      
    print(f'-> Fim getModelos a {time.ctime()}')
    return modelDetect, modelEmbed, modelClassif, encoderIn, encoderOut

def embeddingRostos(modelEmbed, lstDicRosto):
    print(f'-> Inicio embeddingRostos a {time.ctime()}')
    #Se transforma os dados dos rostos em uma array numpy de 4 dimensoes 
    # (1era sample, segunda linha, terceira coluna e quarta canal cor)
    lstDataRosto = list()
    for dicRosto in lstDicRosto:
        lstDataRosto.append(dicRosto['data'])
    npDataRosto = np.array(object=lstDataRosto, ndmin=3)
    
    #Preprocessamento dos dados para atender os requisitos do modelo Facenet
    # scale pixel values
    npDataRosto = npDataRosto.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = npDataRosto.mean(), npDataRosto.std()
    npDataRosto = (npDataRosto - mean) / std
    
    #Se realiza a predicao com o modelo Facenet para ter o vector "embedding" de cada rosto
    yhat = modelEmbed.predict(npDataRosto)
    
    npEmbedRosto = yhat
    
    print(f'-> Fim embeddingRostos a {time.ctime()}')
    return npEmbedRosto

def classifRostos(modelClassif, encoderIn, encoderOut, npEmbedRosto):
    
    print(f'-> Inicio classifRostos a {time.ctime()}')

    # VERIFICAR SE PRECISA OU NAO NORMALIZAR COM O ENCODER DO JOGO DE TREINAMENTO, VISTO QUE PARECE DAR UMA PREDICAO DE MAIOR
    # PROBABILIDADE SEM USAR A NORMALIZACAO
    # MAS TEM MAIS FALSOS POSITIVOS...
    #Se normaliza o array de embedding dos rostos, conforme feito para o jogo de treinamento do modelo de classificacao SVC
    XPred = encoderIn.transform(npEmbedRosto)
    #XPred = npEmbedRosto
    
    #Se realiza a predicao de cada rosto a parte do seu vector normalizado de embedding
    # Retorno um ID de classe de rosto dentro da lista de classes treinadas do modelo
    YPred = modelClassif.predict(XPred)
    # Por rosto, calculo da probabilidade de cada classe (a de probabilidade maior corresponde ao ID presente em YPred para o rosto)
    probPred = modelClassif.predict_proba(XPred)
    # Recuperacao, a parte do encoder do jogo de saida, do label da classe predita
    labelPred = encoderOut.inverse_transform(YPred)

    # Trannsformacao em listas para o retorno da funcao
    lstLabelPredRosto = labelPred.tolist()
    lstProbPredRosto = probPred.max(axis=1).round(2).tolist() #Se recupera so a probabilidade maior por rosto
                                                     # correspondendo a classe de ID presente em lstLabelPredRosto
    
    print(f'-> Fim classifRostos a {time.ctime()}')
    return lstLabelPredRosto, lstProbPredRosto

#Funcao para detecter e evantualmente classificar o/os eventuais rostos de uma foto, fazendo parte das classes reconhecidas pelo modelo treinado SVC
def detectClassifRostosFoto(pathPhoto, pathRosto, modelDetect=None, modelEmbed=None, modelClassif = None, \
                            encoderIn = None, encoderOut = None, limProbLabel = 0.7):
    print(f'CLASSIFICACAO ROSTOS da imagem {pathPhoto}')

    #Tamanho dos rostos esperados pelo modelo de embedding
    sizeRosto = (160, 160)

    try:
        #Se tratar-se bem de um arquivo
        if os.path.isfile(pathPhoto):
            #Nome do arquivo
            nameFile = os.path.basename(pathPhoto)
            print(nameFile)
            
            #Tentativa de carga do arquivo como imagem
            img = Image.open(pathPhoto)
            
            #Eventual rotacao da imagem para estar no bom sentido para a identificacao dos rostos
            img = orientacaoImagem(img)
    
            #Transformacao em array para passar ao extrator de rostos
            dataImg = np.asarray(img)
            
            #Extracao dos rostos da imagem e armazenamento opcional no storage
            _, lstDicRosto = extracaoRostos(modelDetect, nameFile, dataImg, None, limProbLabel, sizeRosto)
            
            #Se nenhum rosto foi achado na foto, saida para passar a foto seguinte
            if len(lstDicRosto) == 0:
                print(f'info: {nameFile} esta sem rosto identificado.')
                return None, None
            
            #Calculo da serie de valores caraterisando cada rosto via o modelo Facenet (processo de embedding)
            npEmbedRosto = embeddingRostos(modelEmbed, lstDicRosto)
            
            #Predicao da classe de cada rosto a parte da serie de embeddind destes rostos
            lstLabelPredRosto, lstProbPredRosto = classifRostos(modelClassif, encoderIn, encoderOut, npEmbedRosto)
            
            #Apresentacao da imagem com os rostos delimitados por rectangulo vermelho
            #plotImageComRostos(dataImg, lstDicRosto, lstLabelPredRosto, lstProbPredRosto, limProbLabel)
            
            #return lstDicRosto, npEmbedRosto, lstLabelPredRosto, lstProbPredRosto
            
            return lstLabelPredRosto, lstProbPredRosto
        else:
            print(f'info:{pathPhoto} corresponde a um repertorio.')
            return None, None
    except UnidentifiedImageError:
        print(f'Warning: O arquivo {nameFile} nao pode estar carregado porque nao esta reconhecido como imagem.')
        return None, None
    except OSError:
        print(f'Warning: O arquivo {nameFile} nao pode estar carregado porque gera um erro OS a pesar estar reconhecido como imagem.')
        #print(sys.exc_info()[0])
        print(sys.exc_info())
        raise
    except:
        print(nameFile)
        print("Unexpected error:", sys.exc_info()[0])
        raise

def predictPessoas(lstLabPesSel, dirNameSel, valOpcaoFindPes, valLimProb, progFindFoto,
                  modelDetect=None, modelEmbed=None, modelClassif = None, encoderIn = None, encoderOut = None):

    dicPesFind = dict()
    
    limProb = valLimProb / 100.
    print(f'Lista label pessoas pesquisadas {lstLabPesSel} com opcao {valOpcaoFindPes} e fracao limite probabilidade de {limProb}')
    
    #Lista dos labels selecionadas pel usuario encontradas nas fotos em funcao da opcao OR (qualquer) ou AND (todas)
    lstLabFind = list()
    
    # Carga dos modelos/encoders caso eles nao foram passados em parametro a funcao
    modelDetect, modelEmbed, modelClassif, encoderIn, encoderOut = getModelos(modelDetect, modelEmbed, modelClassif, encoderIn, encoderOut)
    
    # Pesquisa de todos os arquivos presentes na pasta de selecao de fotos
    os.chdir(dirNameSel)
    maskNameFile = '*'
    pathFiles = glob.iglob(f"{maskNameFile}", recursive=False)

    # Loop sobre cada um dos arquivos achados
    for pathFile in pathFiles:
        #  Detecao e tentativa classificacao eventuais rostos do arquivo  
        lstLabelPredRosto, lstProbPredRosto = \
        detectClassifRostosFoto(pathPhoto=pathFile, pathRosto=None, modelDetect=modelDetect, modelEmbed=modelEmbed, \
                                modelClassif = modelClassif, encoderIn = encoderIn, encoderOut = encoderOut, limProbLabel = limProb)
        #OBS: o limite de probabilidade passada via limProbLabel = limProb, indicada pelo usuario, serve para o filtro de rostos.
        #     Este mesmo limite esta na sequencia usada para filtrar as pessoas.
    
        #Se nao foram achados rostos ou se o arquivo nao esta uma foto
        if lstLabelPredRosto is None:
            # Se passa ao arquivo seguinte
            continue
        
        #Lista labels das pessoas preditas com uma probabilidade superior ao limite configurado em ratio
        lstLabPred = [lstLabelPredRosto[i] for i in range(len(lstProbPredRosto)) if lstProbPredRosto[i] > limProb]
        #Lista %probaabilidade das classes preditas com uma probabilidade superior ao limite configurado em ratio
        lstPercPred = [str(round(lstProbPredRosto[i]*100, 2))+'%' for i in range(len(lstProbPredRosto)) if lstProbPredRosto[i] > limProb]
        #Dico entre as 2 listas para conservar o %predicao por label classe
        dicLabPercPred = dict(zip(lstLabPred, lstPercPred)) 
        
        print(dicLabPercPred)
        
        #Cruzamento em funcao da opcao de pesquisa selecionada pelo usuario
        if valOpcaoFindPes == 'or':
            #S Alguma das classes preditas correspondem as classes selecionadas pelo usuario (cruzando as duas listas de label)
            lstLabFind = list(set(lstLabPesSel) & set(lstLabPred))
        if valOpcaoFindPes == 'and':
            # A totalidade das classes selecionadas pelo usuario devem haver sido preditas pelo modelo
            print(f'set lstLabPesSel {set(lstLabPesSel)} - set lstLabPred {set(lstLabPred)}')
            if set(lstLabPesSel) - set(lstLabPred) == set(): 
                lstLabFind = lstLabPesSel
            else:
                lstLabFind = list()
        
        #Se alguma classe foi achada
        if (len(lstLabFind) > 0):
            nameFile = os.path.basename(pathFile)
            print(f'- Foto {nameFile} com uma lista de pessoas preditas correspondendo as pessoas pesquisadas {lstLabFind}: ')
            lstPercFind = list()
            for i, labClasse in enumerate(lstLabFind):
                percPred = dicLabPercPred[labClasse] #Percentual de probabilidade desta classe
                print(f'-> Classe Lab {labClasse} com a probabilidade {percPred}')
                lstPercFind.append(percPred)   #Lista percentual probabilidade de cada label de classe predita correspondendo a uma classe selecionada
            
            #Visualizacao da imagem sem transformacao
            #print(f'---> Apresentaca da foto presente no caminho {dirNameSel}/{nameFile}')
            #img = load_img(f'{dirNameSel}/{nameFile}') #que faz fr' como estava antes...
            #plt.imshow(img)
            #plt.show()
            
            #Se adiciona os dados da foto ao dico de foto com uma classe predita correspondendo a uma classe selecionada pelo usuario
            # A chave esta o nome do arquivo foto e o valor o tuple da lista dos labels de classe preditas batendo com a lista correspondente de %predicao
            dicPesFind[nameFile] = (lstLabFind, lstPercFind)
    
    
    #Dico a retornar com os dados das fotos de pessoa predita correspondendo a uma das pessoas selecionadas pelo usuario
    return dicPesFind


def cruzaComPes(dicComFind, dicPesFind, valOpcaoFindComPes):
    
    print(f'Cruzamento3 com a opcao {valOpcaoFindComPes} das {len(dicComFind)} fotos com elementos comuns e das {len(dicPesFind)} com pessoas achadas.')

    dicComPesFind = dict()
    
    print(dicComFind)
    print(dicPesFind)
    
    #Cruzamento em funcao da opcao de pesquisa selecionada pelo usuario
    if valOpcaoFindComPes == 'or':
        # Union dos 2 dicionarios
        dicComPesFind = {**dicComFind, **dicPesFind}
        
        # Para as fotos presentes nos 2 dicos (de elemento comum e de pessoas), se junta as classes (label e probabilidade)
        for key in (dicComFind.keys() & dicPesFind.keys()):
            print(key)
            dicComPesFind[key] = (dicComFind[key][0]+dicPesFind[key][0], dicComFind[key][1]+dicPesFind[key][1])
    else:
        print('Intersecao...')
        # Se conserva so as fotos presentes nos 2 dicos (de elemento comum e de pessoas) e se junta as classes achadas (label e probabilidade)
        for key in (dicComFind.keys() & dicPesFind.keys()):
            print(key)
            dicComPesFind[key] = (dicComFind[key][0]+dicPesFind[key][0], dicComFind[key][1]+dicPesFind[key][1])
    
    print(f'Resultado com {valOpcaoFindComPes}: \n {dicComPesFind}')

    #Dico a retornar com os dados das fotos de elementos comuns e/ou pessoas preditos que ficaram depois do cruzamento
    return dicComPesFind
