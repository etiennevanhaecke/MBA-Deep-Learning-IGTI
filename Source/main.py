# import all components
# from the tkinter library
import tkinter as tk
from tkinter import ttk

# import filedialog module
from tkinter import filedialog

from PIL import Image, ImageTk

import datetime

import os

import time

import constants as c
import base_functions as bf

import re

import tkinter.messagebox

os.chdir("C:\\Users\\Utilisateur\\OneDrive\\00 - IGTI\\01 - MBA Deep Learning\\_2021\\DEV\\")
os.getcwd()
pthImg = "C:\\Users\\Utilisateur\\OneDrive\\00 - IGTI\\01 - MBA Deep Learning\\_2021\\DEV\\st-jean-de-luz-france.jpg"

#Carga e compilacao dos modelos
modelClas, modelPes = bf.loadModels()

#modelClas = bf.loadModel()

#Funcao para apresentar uma mensagem em uma popup
def popupmsg(msg):
    popup = tk.Toplevel(root)
    popup.wm_title("AVISO")
    label = ttk.Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Ok", command = popup.destroy)
    B1.pack()
    popup.bind('<Key-Return>', lambda e: B1.invoke())
    popup.mainloop()
    
def popupmsg2(msg):
    popup = tk.Toplevel()		  # Popup -> Toplevel()
    popup.title('AVISO')
    label = ttk.Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    ttk.Button(popup, text='OK', command=popup.destroy).pack(padx=10, pady=10)
    popup.transient(root) 	  # Réduction popup impossible 
    popup.grab_set()		  # Interaction avec fenetre jeu impossible
    root.wait_window(popup)   # Arrêt script principal
    
def popupmsg3(msg, tipo):
    if tipo == 'info':
        tkinter.messagebox.showinfo("AVISO", msg)
    if tipo == 'erro':
        tkinter.messagebox.showerror("ERRO", msg)
    if tipo == 'alerta':
        tkinter.messagebox.showwarning("ALERTA", msg)
        
def listPessoas(dirNameFace):
    
    #As pessoas disponiveis para treinamento de classificacao dos rostos e para predicao estao as que tem uma pasta com o nome dela
    # abaixo do repertorio de rostos indicado pelo usuario (inicialmente vai estar vazio se nao tem um repertorio default)
    
    return lstTagPes

#Se atualiza o box list das pessoaos disponiveis para treinamento/pesquisa
def updBoxLstPes():
    # Montagem da lista de nome das subpastas abaxo da pasta de rosto selecionada 
    #   que correspondem ao nome das pessoaos para treinamento/predica
    global lstTagPes
    lstTagPes = [f.name for f in os.scandir(dirNameFace) if f.is_dir() ]
    
    # Se muda o conteudo do boxlist das pessoas
    lstPessoaVar.set(lstTagPes)
    
    #Se a lista das pessoas nao esta vazia
    if lstTagPes:
        #Se alterna a cor de fundo das linhas do list box
        for i in range(0, len(lstTagPes), 2):
            lstbxPessoa.itemconfigure(i, background='#f0f0ff') #'#f0f0ff'

def designPessoas(mainFrame):
    varOpcaoFindPes = tk.StringVar(value='or')
    rbOrPes = ttk.Radiobutton(mainFrame, text="OR", variable=varOpcaoFindPes, value='or')
    rbAndPes = ttk.Radiobutton(mainFrame, text="AND", variable=varOpcaoFindPes, value='and')
    rbOrPes.grid(column=6, row=3, sticky=tk.E)
    rbAndPes.grid(column=7, row=3, sticky=tk.W)

    #Se ja esta definida a variavel do repertorio dos rostos
    lstTagPes = list()
    if varDirFace.get() != '':
        # Se recupera a lista de nome dos subrepertorios, que correspondem ao nome das pessoaos para treinamento/predica
        lstTagPes = [f.name for f in os.scandir(varDirFace.get()) if f.is_dir() ]
    lstPessoaVar = tk.StringVar(value=lstTagPes)
    lstbxPessoa = tk.Listbox(mainFrame, listvariable=lstPessoaVar, height=7, selectmode='extended', exportselection=False)
    #OBS: exportselection a False permite de selecionar em diferentes listbox sem perder as selecoes (senao fica so a ultima selecao)
    lstbxPessoa.grid(column=7, row=4, sticky=tk.W)
    # Colorize alternating lines of the listbox
    
    #Se a lista das pessoas nao esta vazia
    if lstTagPes:
        #Se alterna a cor de fundo das linhas do list box
        for i in range(0, len(lstTagPes), 2):
            lstbxPessoa.itemconfigure(i, background='#f0f0ff') #'#f0f0ff'
    scrLstbxPessoa = ttk.Scrollbar(mainFrame, orient=tk.VERTICAL, command=lstbxPessoa.yview)
    scrLstbxPessoa.grid(column=6, row=4, sticky=(tk.N, tk.S, tk.E))
    lstbxPessoa.configure(yscrollcommand=scrLstbxPessoa.set)
    
    return varOpcaoFindPes, lstPessoaVar, lstbxPessoa
        
# Function for opening the file explorer window
def browseFiles(*args):
    global fileNameSel
    fileNameSel = filedialog.askopenfilename(initialdir = "/", title = "Seleciona um Arquivo", 
                                          filetypes = (("Text files", "*.txt*"), ("all files","*.*")))
    # Change label contents
    varFileSel.set("Arquivo Selecionado: "+fileNameSel)
    
    
# Function for opening the path explorer window
def browseDir(origBut, *args):
    dirName = filedialog.askdirectory(initialdir = "/", title = "Seleciona um repertorio", 
                                        mustexist=True)
    # Change label contents
    if origBut == 'select':
        global dirNameSel
        dirNameSel = dirName
        varDirSel.set("Diretorio Selecionado: "+dirNameSel)
    else:
        if origBut == 'face':
            global dirNameFace
            dirNameFace = dirName
            
            # Se muda o label associaado ao butom de selecao do repertorio de rosto
            varDirFace.set("Diretorio Selecionado: "+dirNameFace)
            
            # Se o repertrio dos rostos esta definido
            if dirNameFace != '':
                # Change conteudo da list box das pessaos disponiveis para treinamento e pesquisa
                updBoxLstPes()                    
    
def closeApp():
    root.destroy()

    
def findFoto(flagComPes=0):
    progFindFoto['value'] = 0.
    
    #Se limpa o canvas das fotos achadas
    cnvTopFoto2.delete("all")
    
    if ( (not('dirNameSel' in globals())) or (dirNameSel == '') ):
        # Mensagem warning para o usuario remediar antes de lancar o treinamento
        popupmsg3('Selecionar antes, por favor, o repertorio das fotos a pesquisar', 'alerta')
        
        # Retorno none caso chamada desde a funcao pesquisa elemnto comun/pessoa
        return None
    #Senao
    else:
        # Pesquisa das fotos
        print(f'Pesquisa na pasta {dirNameSel} das classes comuns:')
        #progFindFoto.start()
        lstIDClasSel = lstbxClasse.curselection()
        for idx in lstIDClasSel:
            #time.sleep(1)
            #root.after(1000)
            print(f'Classe de label selecionada {c.lstTagClas[idx]} de ID classe {c.dicTagId[c.lstTagClas[idx]]}')
            #progFindFoto['value'] = progFindFoto['maximum'] / len(lstIDClasSel) * (idx+1)
            #progFindFoto.step(progFindFoto['maximum'] / len(lstClasseSel))

        #progFindFoto.stop()
        #Pesquisa das fotos correspondendo as classes selecionadas aplicando o modelo de predicao das classes comuns sobre cada foto da pasta selecionada
        dicImgFind = bf.predictFotos(lstIDClasSel, dirNameSel, varOpcaoFind.get(), modelClas, float(varLimProb.get()), progFindFoto)

        if flagComPes == 0:
            #Apresentacao destas fotos no frame separado topFoto com indicacao das classes encontradas e a probabilidade de predicao destas classes
            bf.showFotosFind(dicImgFind, cnvTopFoto2, dirNameSel, int(varNumCol.get()), int(varWidthImg.get()), int(varHeightImg.get()) )
            countImgFind = len(bf.lstImg)
            print(f'len lstImg: {countImgFind}')
            popupmsg3(f'{countImgFind} fotos encontradas', 'info')
        else:
            # A chamada com flagComPes = 1 esta feita quando se pesquisa tanto elemento comun que pessoa (findPesCom).
            # Neste caso a apresentacao das fotos esta controlada pela funcao findPesCom
            # Retorno das fotos com pessoas achadas
            return dicImgFind

def findFace():
    progFindFoto['value'] = 0.
    
    #Se limpa o canvas das fotos achadas
    cnvTopFoto2.delete("all")
    
    if ( (not('dirNameSel' in globals())) or (dirNameSel == '') ):
        # Mensagem warning para o usuario remediar antes de lancar o treinamento
        popupmsg3('Selecionar antes, por favor, o repertorio das fotos a pesquisar', 'alerta')
    #Senao
    else:
        if ( (not('dirNameFace' in globals())) or (dirNameFace == '') ):
            # Mensagem warning para o usuario remediar antes de lancar o treinamento
            popupmsg3('Selecionar antes, por favor, o repertorio dos rostos', 'alerta')
        #Senao
        else:
            # Se lanca a pesquisa dos rostos
            print(f'Pesquisa de rosto das fotos da pasta {dirNameSel} e copia no repertorio {dirNameFace}.')

            dicFaceSaveTot = bf.predictFaces(dirNameSel, dirNameFace, float(varLimProb.get()), progFindFoto)

            #Apresentacao destas fotos no frame separado topFoto com indicacao das classes encontradas e a probabilidade de predicao destas classes
            bf.showFotosFind(dicFaceSaveTot, cnvTopFoto2, dirNameFace, 5, 180, 180 )
            countFaceFind = len(dicFaceSaveTot)
            print(f'len dicFaceSaveTot: {countFaceFind}')
            popupmsg3(f'{countFaceFind} rostos encontradas', 'info')
    
    
def trainPessoa():
    print('trainPessoa')
    
    #testLogic = ('dirNameFace' in globals())
    #print(f'test dirNameFace in global: {testLogic}')
    #print(f'valor dirNameFace {dirNameFace}')
    
    #Se o repertorio dos rostos nao foi ainda selecionado (variavel global dirNameFace nao definida ou nula)
    # verif porque ~('dirNameFace' in globals()) segue a retornar True com dirNameFace definido
    if ( (not('dirNameFace' in globals())) or (dirNameFace == '') ):
        # Mensagem warning para o usuario remediar antes de lancar o treinamento
        popupmsg3('Selecionar antes, por favor, o repertorio dos rostos', 'alerta')
    #Senao
    else:
        # se lanca o treinamento
        progFindFoto['value'] = 0.
        nameModelPes, pathModel, accTrain, accVal = bf.trainPersons(dirNameFace, progFindFoto)
    
        popupmsg3(f'Modelo pesquisa pessoas treinado, conservado no arquivo {nameModelPes} do repertorio {pathModel}', 'info')
        popupmsg3(f'Accuracidade jogo de treinamento {accTrain}% e sobre o jogo de validacao {accVal}%', 'info')
        
    
def findPessoa(flagComPes=0):
    
    progFindFoto['value'] = 0.
    
    #Se limpa o canvas das fotos achadas
    cnvTopFoto2.delete("all")
    
    if ( (not('dirNameSel' in globals())) or (dirNameSel == '') ):
        # Mensagem warning para o usuario remediar antes de lancar o treinamento
        popupmsg3('Selecionar antes, por favor, o repertorio das fotos a pesquisar', 'alerta')
        
        # Retorno none caso chamada desde a funcao pesquisa elemnto comun/pessoa
        return None
    #Senao
    else:
        # Pesquisa das pessoas
        print(f'Pesquisa na pasta {dirNameSel} das pessoas:')
        #progFindFoto.start()
        lstIDPesSel = lstbxPessoa.curselection()
        #Precisa passar a lista dos labels da selecao, ja que a ordem destes labels na box list pode nao corresponder
        # a ordem de ID dos labels durante o treinamento, conservados no objeto encoder do treinamento
        lstLabPesSel = [lstTagPes[idx] for idx in lstIDPesSel]
        
        for idx in lstIDPesSel:
            #time.sleep(1)
            #root.after(1000)
            print(f'Pessoa de label selecionado {lstTagPes[idx]} de ID classe {idx}')
            #progFindFoto['value'] = progFindFoto['maximum'] / len(lstIDClasSel) * (idx+1)
            #progFindFoto.step(progFindFoto['maximum'] / len(lstClasseSel))
        
        dicPesFind = bf.predictPessoas(lstLabPesSel, dirNameSel, varOpcaoFindPes.get(), float(varLimProb.get()), progFindFoto)
        
        if flagComPes == 0:
            #Apresentacao destas fotos no frame separado topFoto com indicacao das classes encontradas e a probabilidade de predicao destas classes
            bf.showFotosFind(dicPesFind, cnvTopFoto2, dirNameSel, int(varNumCol.get()), int(varWidthImg.get()), int(varHeightImg.get()), 2)
            countPesFind = len(dicPesFind)
            print(f'len lstImg: {countPesFind}')
            popupmsg3(f'{countPesFind} fotos encontradas', 'info')
        else:
            # A chamada com flagComPes = 1 esta feita quando se pesquisa tanto elemento comun que pessoa (findPesCom).
            # Neste caso a apresentacao das fotos esta controlada pela funcao findPesCom
            # Retorno das fotos com pessoas achadas
            return dicPesFind
            
    
def findComPes():
    #Se lanca tanto a pesquisa das fotos de elementos comuns que das fotos de pessoas, indicando com o parametro
    # flagComPes a 1, a nao apresentacao das fotos destes as funcoes destas pesquisas
    
    # Pesquisa das fotos correspondendo aos elementos comuns selecionados pelo usuario
    dicComFind = findFoto(flagComPes=1)
    # Se transforma o ID das classes achadas em label das classes
    dicComFind = {key: ([c.dicIdTag[idx] for idx in dicComFind[key][0]], dicComFind[key][1]) for key in dicComFind.keys() }
    
    # Pesquisa das fotos correspondendo as pessoas selecionadas pelo usuario
    dicPesFind = findPessoa(flagComPes=1)
    
    # Se cruza as fotos achadas de elementos comuns e de pessoas em funcao da opcao or/and selecionada pelo usuario
    dicComPesFind = bf.cruzaComPes(dicComFind, dicPesFind, varOpcaoFindComPes.get())
        
    #Apresentacao destas fotos no frame separado topFoto com indicacao das classes encontradas e a probabilidade de predicao destas classes
    bf.showFotosFind(dicComPesFind, cnvTopFoto2, dirNameSel, int(varNumCol.get()), int(varWidthImg.get()), int(varHeightImg.get()), 2)
    countComPesFind = len(dicComPesFind)
    print(f'len countComPesFind: {countComPesFind}')
    popupmsg3(f'{countComPesFind} fotos encontradas', 'info')
    
root=tk.Tk()
root.title('Pesquisa por IA das fotos')
root.geometry("1100x600")

#Frame compativel coom o novo modulo ttk, o que nao esta o caso da janela principal root
mainFrame = ttk.Frame(root, padding="3 3 12 12", width = 400, height = 300)
mainFrame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
mainFrame['borderwidth'] = 20
mainFrame['relief'] = 'sunken'

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

#Set window background color
#mainFrame.(background = "white")

#Check input pelo usuario do tamamnho de apresentaca das imagens achadas
def checkNum(newVal):
    if (re.match('^[0-9]*$', newVal) is not None and len(newVal) <= 3):
        return True
    else:
        popupmsg3('Erro de digitacao. Tem que estar um numero inferior a 1000!', 'erro')
        return False
checkNum_wrapper = (root.register(checkNum), '%P')


#varFileSel = tk.StringVar()
#labFileSel = ttk.Label(mainFrame, text='Nenhum arquivo selecionado', width=50, textvariable=varFileSel, wraplength="10c")
#labFileSel.grid(column=0, row=0)
#butFileSel = ttk.Button(mainFrame, text='select foto file', command=browseFiles)
#butFileSel.grid(column=0, row=1)

varDirSel = tk.StringVar()
butDirSel = ttk.Button(mainFrame, text='Repertorio Fotos Pesquisa', command= lambda: browseDir('select'))
butDirSel.grid(column=2, row=0)
labDirSel = ttk.Label(mainFrame, text='Nenhum repertorio selecionado', width=70, textvariable=varDirSel, foreground='red')
labDirSel.grid(column=3, row=0, columnspan=4)

butDirSel.bind('<Enter>', lambda e: butDirSel.configure(underline=5))
#butDirSel.bind('<Enter>', lambda e: butDirSel.configure(text='Moved mouse inside'))
butDirSel.bind('<Leave>', lambda e: butDirSel.configure(underline=-1)) #

varDirFace = tk.StringVar()
butDirFace = ttk.Button(mainFrame, text='Repertorio Fotos Rostos', command= lambda: browseDir('face'))
butDirFace.grid(column=2, row=1)
labDirFace = ttk.Label(mainFrame, text='Nenhum repertorio selecionado', width=70, textvariable=varDirFace, foreground='green')
labDirFace.grid(column=3, row=1, columnspan=4)
    
sty = ttk.Style()
sty.configure('Danger.TFrame', background='red', borderwidth=5, relief='raised')
#mainFrame['style'] = 'Danger.TFrame'
root.bind('<Return>', lambda e: mainFrame.configure(style='Danger.TFrame'))

#img=Image.open(pthImg)
#img.thumbnail((300,300),Image.ANTIALIAS)
#img.thumbnail((200,200))
#img1 = ImageTk.PhotoImage(img)
#img2 = tk.PhotoImage(img)
#labImage = ttk.Label(mainFrame, image=img1, compound='top', text='image', anchor=tk.W)
#labImage.grid(column=0, row=3)

#varCumprido = tk.StringVar()
#varCumprido.set('fferkfpoekpekgokgpkgpgkp\nfrlrplgefrkfprkpkrefr\nfrefrfkkpr\nggke\nmf\nggjigg')
#labWrapped = ttk.Label(mainFrame, textvariable = varCumprido)
#labWrapped.grid(column=0, row=4)

#flagUseTag = tk.BooleanVar(value=False)
#chkUseTag = ttk.Checkbutton(mainFrame, text='Use fotos tags', variable=flagUseTag, onvalue=True, offvalue=False)
#chkUseTag.grid(column=4, row=1)

butFindFoto = ttk.Button(mainFrame, text='Pesquisa Comuns', command=findFoto)
butFindFoto.grid(column=1, row=2)
#root.bind('<Key-Return>', lambda e: butFindFoto.invoke())

butFindFace = ttk.Button(mainFrame, text='Extracao rostos', command=findFace)
butFindFace.grid(column=5, row=2)

butTrainPessoa = ttk.Button(mainFrame, text='Treinamento Pessoas', command=trainPessoa)
butTrainPessoa.grid(column=5, row=3)

butFindPessoa = ttk.Button(mainFrame, text='Pesquisa Pessoas', command=findPessoa)
butFindPessoa.grid(column=7, row=2, sticky=(tk.E))

labLimProb = ttk.Label(mainFrame, text='Limite Probabilidade', width=25, foreground='blue', anchor=tk.E)
labLimProb.grid(column=2, row=2, sticky=(tk.E))
varLimProb = tk.StringVar(value=c.limProb)
entLimProb = ttk.Entry(mainFrame, textvariable=varLimProb, width=2, validate='key', validatecommand=checkNum_wrapper)
entLimProb.grid(column=3, row=2, sticky=(tk.W))

labNumCol = ttk.Label(mainFrame, text='Num. Col Painel Fotos', width=25, foreground='blue', anchor=tk.E)
labNumCol.grid(column=2, row=3, sticky=tk.E)
varNumCol = tk.StringVar(value=c.numColImgFind)
cbNumCol = ttk.Combobox(mainFrame, textvariable=varNumCol, values=(1, 2, 3, 4, 5, 6), state = 'readonly', justify='center', width=2)
#cbNumCol.bind('<<ComboboxSelected>>', lambda e: cbNumCol.selection_clear())
cbNumCol.grid(column=3, row=3, sticky=tk.W)

labWidthImg = ttk.Label(mainFrame, text='Largura Fotos', width=25, foreground='blue', anchor=tk.E)
labWidthImg.grid(column=2, row=4, sticky=(tk.E,  tk.S))
varWidthImg = tk.StringVar(value=c.widthImgFind)
entWidthImg = ttk.Entry(mainFrame, textvariable=varWidthImg, width=3, validate='key', validatecommand=checkNum_wrapper)
entWidthImg.grid(column=3, row=4, sticky=(tk.W, tk.S))

labHeightImg = ttk.Label(mainFrame, text='Altura Fotos', width=25, foreground='blue', anchor=tk.E)
labHeightImg.grid(column=2, row=5, sticky=(tk.E,  tk.S))
varHeightImg = tk.StringVar(value=c.heightImgFind)
entHeightImg = ttk.Entry(mainFrame, textvariable=varHeightImg, width=3, validate='key', validatecommand=checkNum_wrapper)
entHeightImg.grid(column=3, row=5, sticky=(tk.W, tk.S))

progFindFoto = ttk.Progressbar(mainFrame, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=c.totStepProg) #indeterminate
progFindFoto.grid(column=1, row=5)

butQuit = ttk.Button(mainFrame, text="Saida", default="active", command=closeApp)
butQuit.grid(column=1, row=6)
root.bind('<Key-Escape>', lambda e: butQuit.invoke())

for child in mainFrame.winfo_children(): 
    child.grid_configure(padx=5, pady=5)
    #child['padding'] = [2, 2, 2, 2]
    #pass

varOpcaoFind = tk.StringVar(value='or')
rbOr = ttk.Radiobutton(mainFrame, text="OR", variable=varOpcaoFind, value='or')
rbAnd = ttk.Radiobutton(mainFrame, text="AND", variable=varOpcaoFind, value='and')
rbOr.grid(column=0, row=3, sticky=tk.E)
rbAnd.grid(column=1, row=3, sticky=tk.W)

lstClasseVar = tk.StringVar(value=c.lstTagClas)
lstbxClasse = tk.Listbox(mainFrame, listvariable=lstClasseVar, height=7, selectmode='extended', exportselection=False)
lstbxClasse.grid(column=1, row=4, sticky=tk.W)
# Colorize alternating lines of the listbox
for i in range(0, c.countClasses, 2):
    lstbxClasse.itemconfigure(i, background='#f0f0ff') #'#f0f0ff'
scrLstbxClasse = ttk.Scrollbar(mainFrame, orient=tk.VERTICAL, command=lstbxClasse.yview)
scrLstbxClasse.grid(column=0, row=4, sticky=(tk.N, tk.S, tk.E))
lstbxClasse.configure(yscrollcommand=scrLstbxClasse.set)

varOpcaoFindPes, lstPessoaVar, lstbxPessoa = designPessoas(mainFrame)

butFindComPes = ttk.Button(mainFrame, text='Pesquisa Comuns/Pessoas', command=findComPes)
butFindComPes.grid(column=4, row=5, sticky=(tk.W, tk.E))
varOpcaoFindComPes = tk.StringVar(value='or')
rbOrComPes = ttk.Radiobutton(mainFrame, text="OR", variable=varOpcaoFindComPes, value='or')
rbAndComPes = ttk.Radiobutton(mainFrame, text="AND", variable=varOpcaoFindComPes, value='and')
rbOrComPes.grid(column=3, row=6, sticky=tk.E)
rbAndComPes.grid(column=4, row=6, sticky=tk.W)
    
    
#topFoto = tk.Toplevel(root)
#topFoto.title('Fotos encontradas')
#topFoto.geometry('900x700-5+40')
#topFoto.update_idletasks()
#print(topFoto.geometry())
#topFoto.resizable(True, True)
#topFoto.minsize(300,300)
#topFoto.maxsize(1400,900)
#topFoto.grid_propagate(1) #Permite ao frame de se extender para apresentar as fotos

topFoto2 = tk.Toplevel(root)
topFoto2.title('Fotos encontradas V2')
topFoto2.geometry('1000x500-5+40')
#topFoto.update_idletasks()
#print(topFoto2.geometry())
topFoto2.rowconfigure(0, weight=1) 
topFoto2.columnconfigure(0, weight=1)
topFoto2.resizable(True, True)
topFoto2.minsize(200,200)
topFoto2.maxsize(1500,800)
topFoto2.grid_propagate(1) #Permite ao frame de se extender para apresentar as fotos

scrTopFoto2Hor=ttk.Scrollbar(topFoto2, orient=tk.HORIZONTAL)
scrTopFoto2Hor.grid(row=6, column=0, sticky="we")
scrTopFoto2Ver=ttk.Scrollbar(topFoto2, orient=tk.VERTICAL)
scrTopFoto2Ver.grid(row=0, column=6, sticky="ns")
cnvTopFoto2 = tk.Canvas(topFoto2, scrollregion=(0, 0, 24000, 24000), 
                        yscrollcommand=scrTopFoto2Ver.set, xscrollcommand=scrTopFoto2Hor.set)
cnvTopFoto2.grid(row=0, column=0, sticky="nsew") #added sticky
scrTopFoto2Ver.config(command=cnvTopFoto2.yview)
scrTopFoto2Hor.config(command=cnvTopFoto2.xview)
print(cnvTopFoto2)

root.mainloop()

