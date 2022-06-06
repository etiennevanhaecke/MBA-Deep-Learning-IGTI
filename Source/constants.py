#Lista de tags das 15 classes trabalhadas 
lstTagClas = ['arvore', 'ave', 'cachorro', 'caminhao', 'carro', 'casa', 'cavalo', \
              'gato', 'mar', 'montanha', 'ponte', 'praia', 'predio', 'rio', 'sol']
#Caso nao seja ainda feito, se ordena a lista por ordem alfabetico
lstTagClas.sort()
#print(type(lstTagClas))
#Criacao do dicionario tag-id label das classes
dicTagId = {lstTagClas[idx]:idx for idx in range(len(lstTagClas))}
#Criacao do dicionario id-tag label das classes
dicIdTag = {idx:lstTagClas[idx] for idx in range(len(lstTagClas))}

countClasses = len(lstTagClas)
#print(countClasses)

#Parametros
#Tamanho redimensionamento das fotogradias antes de transformar elas em array para passar na predicao do modelo
sizeImg = 128

#Limite de probabilidade a parte de qual se considera que a predicao de uma classe esta correta
limProb = 50 #0.5

#Numero colunas para apresentar as fotos encnntradas no frame independente topFoto
numColImgFind = 3
#Tamanho das fotos encontradas no frame independente topFoto
widthImgFind = 300
heightImgFind = 200

#Espaco entre as imagens verticalmente e horizontalmente
spaceImgFind=35 #10

#Total de step da barra de progressao
totStepProg = 100.