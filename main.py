import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#tornillo C:\Users\POCHI\Documents\FACULTAD\Inteligencia Artificial 1\Final\imagenes\IMG_20230531_113513144.jpg
#clavo C:\Users\POCHI\Documents\FACULTAD\Inteligencia Artificial 1\Final\imagenes\IMG_20230531_113309287.jpg
#arandela C:\Users\POCHI\Documents\FACULTAD\Inteligencia Artificial 1\Final\imagenes\IMG_20230531_113535690.jpg
#tuerca C:\Users\POCHI\Documents\FACULTAD\Inteligencia Artificial 1\Final\imagenes\IMG_20230531_113606988.jpg

def procesar_imagen(imagen):
    img = cv2.imread('C:\\Users\\POCHI\\Documents\\FACULTAD\\Inteligencia Artificial 1\\Final\\imagenes\\IMG_20230531_113606988.jpg', cv2.IMREAD_UNCHANGED)
    imagen = cv2.resize(img, None, fx=0.15, fy=0.15)
    imagen = cv2.GaussianBlur(imagen, (7,7), 0)  ##Elimino ruido y suavio
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.Canny(imagen,0, 100) ##Resalto bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imagen = cv2.dilate(imagen, kernel, iterations=2)  #Proceso de dilatacion que ayuda a eliminar ruido
    contornos, jerarquia = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen, contornos, -1, (255,255,255), -1)
    imagen = cv2.erode(imagen, kernel, iterations=3)
    # Mostrar la imagen original y la imagen procesada
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(imagen, cmap='gray'), plt.title('Procesada')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return imagen


def guardar_datos():
    carpeta_imagenes = 'C:\\Users\\POCHI\\Documents\\FACULTAD\\Inteligencia Artificial 1\\Final\\imagenes'
    lista_imagenes = os.listdir(carpeta_imagenes)
    for nombre_imagen in lista_imagenes:
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
        img =procesar_imagen(ruta_imagen)
        momentos_hu = cv2.HuMoments(cv2.moments(img)).flatten()

    # Guardar los momentos de Hu en la base de datos o archivo
    # Aquí puedes guardar los momentos de Hu en una estructura de datos o archivo que funcione como base de datos


