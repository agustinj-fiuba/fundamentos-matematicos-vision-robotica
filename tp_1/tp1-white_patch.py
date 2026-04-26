import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Función que ejecuta el algoritmo de white_patch
def white_patch(img):
    r_max = np.max(img[:,:,0])
    g_max = np.max(img[:,:,1])
    b_max = np.max(img[:,:,2])
    # print(r_max, g_max, b_max) -> La imagen 4 tiene 255 en RGB máx, devuelve la misma imagen
    img[:,:,0] = cv2.multiply(img[:,:,0], 255/r_max)
    img[:,:,1] = cv2.multiply(img[:,:,1], 255/g_max)
    img[:,:,2] = cv2.multiply(img[:,:,2], 255/b_max)
    return img

# Lectura de las imagenes
paths = os.listdir("white_patch")
images = []
for path in paths:
    img = cv2.imread("white_patch/" + path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img_RGB)

# Muestro resultados
fig, ax = plt.subplots(2, 4)
for i in range(len(images)):
    ax[int(i/4), i % 4].imshow(white_patch(images[i]))

plt.show()