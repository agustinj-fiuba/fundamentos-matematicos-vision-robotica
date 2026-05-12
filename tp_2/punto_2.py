import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Calculo MIS2 de la imagen ingresada (podrá ser frame completo o roi)
def calculo_MIS2(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # Convierto a gray
    height, width = img.shape[:2]
    img = img.astype(np.float64)
    I_base = img[:,:-2]
    I_desplazada = img[:,2:]
    diff = I_base - I_desplazada
    diff_sq = np.sum(diff**2)
    return diff_sq/(diff.shape[0]*diff.shape[1])

# Calculo ROI con porcentaje variable
def calculo_roi(img, percent):

    height, width = img.shape[:2]
    center_x = width//2
    center_y = height//2
    roi_width = int(width*np.sqrt(percent/100)) # Con el porcentaje obtengo el ancho
    roi_height = int(height*np.sqrt(percent/100)) # Con el porcentaje obtengo la altura
    y_bottom = center_y + roi_height//2
    y_top = center_y - roi_height//2
    x_right = center_x + roi_width//2
    x_left = center_x - roi_width//2

    roi = img[y_top:y_bottom, x_left:x_right] # Tomo la ROI de la imagen
    return roi, y_top, y_bottom, x_left, x_right

# Calculo cantidad de matrices de enfoque y posiciones, para luego ver promedio de la métrica, considero un ROI del 50%
def MIS2_matriz_enfoque(img, n, m):
    # Chequear n y m son impares
    if n % 2 != 0 and m % 2 != 0:
        roi, y_top, y_bottom, x_left, x_right = calculo_roi(img, 50)
        height, width = roi.shape[:2]
        grid_x = width//m
        grid_y = height//n

        total_matrices = 0
        total_matriz_enfoque = 0
        # Hago loop por la grilla y calculo la métrica para cada elemento
        for i in range(0,n,2):
            for j in range(0,m,2):
                y_start = i*grid_y
                y_end = (i+1)*grid_y
                x_start =j*grid_x
                x_end = (j+1)*grid_x
                matriz = roi[y_start:y_end,x_start:x_end]
                total_matriz_enfoque += calculo_MIS2(matriz)
                total_matrices += 1
                cv.rectangle(img, (x_start+x_left,y_start+y_top), (x_end+x_left,y_end+y_top), (0,255,0),2) # Hago bounding box en cada elemento de la grilla

        return total_matriz_enfoque/total_matrices # Promedio de los resultados de la métrica
    else:
        print("Verificar que N y M sean impares")
        return

# Inicio Video Capture
cap = cv.VideoCapture('focus_video.mov')

# Inicializo resultados
MIS2_frame_completo = []
MIS2_roi_centro = []
MIS2_matrices_enfoque = [[],[],[]]

# Imagenes con futuras bounding boxes
base_images = []
roi_images = []
focus_images = []

# Hagoo loop por Video Capture
while cap.isOpened():
    # Leo un frame
    ret, frame = cap.read()
    # Reviso que exista
    if ret:
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        base_images.append(img)
        # Hago copias para marcar en la imagen los diferentes métodos utilizados
        img_1 = img.copy()
        img_2 = img.copy()
        img_3 = img.copy()
        img_4 = img.copy()

        # Calculo con Frame Completo
        MIS2_frame_completo.append(calculo_MIS2(img))

        # Calculo con ROI
        roi, y_top, y_bottom, x_left, x_right = calculo_roi(img_1, 10)
        MIS2_roi_centro.append(calculo_MIS2(roi)) # el porcentaje va de 0 a 100
        cv.rectangle(img_1, (x_left, y_top), (x_right, y_bottom), (0,255,0), 2)
        roi_images.append(img_1)

        # Calculo con matriz de enfoque, se elige matriz según NxM buscado, deben ser impares
        MIS2_matrices_enfoque[0].append(MIS2_matriz_enfoque(img_2, 5, 5)) # 5x5
        MIS2_matrices_enfoque[1].append(MIS2_matriz_enfoque(img_3, 7, 5)) # 7x5
        MIS2_matrices_enfoque[2].append(MIS2_matriz_enfoque(img_4, 11, 11)) # 11x11
        focus_images.append(img_4)

    else:
        print("Termino de procesar video")
        break

cap.release()
cv.destroyAllWindows()

frames = np.arange(0, len(base_images))

# Organizo los elementos en una lista para ver los máximos
metodos = [np.array(MIS2_frame_completo), np.array(MIS2_roi_centro), 
           np.array(MIS2_matrices_enfoque[0]), np.array(MIS2_matrices_enfoque[1]), np.array(MIS2_matrices_enfoque[2])]
nombres = ["Frame completo", "ROI 10%", "5x5", "7x5", "11x11"]
frames_max = []
max_fm = []
tolerancia = 0.98

# Calculo de los máximos con umbral
print("Resultados del algoritmo:")
for j in range(len(metodos)):
    umbral_j = np.max(metodos[j])*tolerancia
    mascara_j = metodos[j] >= umbral_j
    frames_max.append(frames[mascara_j])
    max_fm.append(metodos[j][mascara_j])
    print(f"Frames enfocados según {nombres[j]}:", frames_max[j])


# Gráfico comparando las distintas métricas FM
layout = """
AB
AC
AD
"""
fig, ax = plt.subplot_mosaic(layout, figsize=(12,6))
# Ploteo resultados totales
for i in range(len(metodos)):
    # Ploteo métricas según método
    ax["A"].plot(frames, metodos[i], label=nombres[i], c=f"C{i}", zorder=1)
    # Ploteo máximos por algoritmo
    ax["A"].scatter(frames_max[i], max_fm[i], s=10, c=f"C{i}", marker="s", edgecolors='black', label=f"Máx {nombres[i]}", zorder=2)
ax["A"].set_title("Comparación diferentes métodos usando MIS2")
ax["A"].set_xlabel("Frames")
ax["A"].set_ylabel("MIS2")
ax["A"].legend(loc="lower right", fontsize=6)

# Muestro resultados
ax["B"].imshow(base_images[MIS2_frame_completo.index(max( MIS2_frame_completo))])
ax["B"].set_title("Frame completo con máxima métrica")
ax["C"].imshow(roi_images[MIS2_roi_centro.index(max(MIS2_roi_centro))])
ax["C"].set_title("Roi 10% con máxima métrica")
ax["D"].imshow(focus_images[MIS2_matrices_enfoque[2].index(max(MIS2_matrices_enfoque[2]))])
ax["D"].set_title("11x11 con máxima métrica")

plt.tight_layout()
plt.savefig("test.png",dpi=400)
plt.show()
