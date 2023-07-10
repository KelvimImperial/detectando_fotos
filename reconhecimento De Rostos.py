#imagens/px-people.jpg
#haarcascade_frontalface_default.xml

import cv2
import matplotlib.pyplot as plt

imagem=cv2.imread("px-people.jpg")
imagem=cv2.cvtColor(imagem,cv2.COLOR_BGR2RGB)

#Transformando a imagem em escala de cinza

imagem_cinza=cv2.cvtColor(imagem,cv2.COLOR_RGB2GRAY)

#Criando o classificador

classificador=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Contando a quantidade de faces na imagem

faces=classificador.detectMultiScale(imagem_cinza,1.3,5)

print("Quantidade de rostos na imagem:",len(faces))

#Colocando retangulos nos rostos

imagem_copy=imagem.copy()

for (x,y,w,h) in faces:

    cv2.rectangle(imagem_copy,(x,y),(x+w,y+h),(255,255,0),2)

plt.imshow(imagem_copy)
plt.show()




