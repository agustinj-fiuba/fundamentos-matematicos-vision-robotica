import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leo imagenes en escala de grises
img_1 = cv2.imread("img1_tp.png", cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread("img2_tp.png", cv2.IMREAD_GRAYSCALE)

# IMAGEN 1 - GRADIENTE
fig, ax = plt.subplots(3, 2)
ax[0, 0].set_title("Sin ecualizar")
ax[0, 0].imshow(img_1, cmap="gray")

bins_1 = 256
ax[1, 0].hist(img_1.ravel(), bins_1)
ret, thresh = cv2.threshold(img_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ax[2, 0].imshow(thresh, cmap='gray', vmin=0, vmax=1)

img_1_eq = cv2.equalizeHist(img_1)
ax[0, 1].set_title("Ecualizada")
ax[0, 1].imshow(img_1_eq, cmap="gray")
ax[1, 1].hist(img_1_eq.ravel(), bins_1)
ret, thresh = cv2.threshold(img_1_eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ax[2, 1].imshow(thresh, cmap='gray', vmin=0, vmax=1)

plt.show()

# IMAGEN 2 - FLOR
fig, ax = plt.subplots(3, 2)
ax[0, 0].set_title("Sin ecualizar")
ax[0, 0].imshow(img_2, cmap="gray")

bins_2 = 256
ax[1, 0].hist(img_2.ravel(), bins_2)
ret, thresh = cv2.threshold(img_2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ax[2, 0].imshow(thresh, cmap='gray', vmin=0, vmax=1)

img_2_eq = cv2.equalizeHist(img_2)
ax[0, 1].set_title("Ecualizada")
ax[0, 1].imshow(img_2_eq, cmap="gray")
ax[1, 1].hist(img_2_eq.ravel(), bins_2)
ret, thresh = cv2.threshold(img_2_eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ax[2, 1].imshow(thresh, cmap='gray', vmin=0, vmax=1)

plt.show()
