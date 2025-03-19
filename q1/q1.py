import cv2
import numpy as np

# Carregar o vídeo q1A.mp4 (conforme RM 94592)
cap = cv2.VideoCapture('q1/q1A.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para HSV para detecção de cores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Intervalos HSV ajustados para as cores do vídeo (baseado nas imagens)
    # Vermelho (círculo)
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    # Azul (quadrado)
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    # Máscaras para cada cor
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Encontrar contornos
    contours_red, _ = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(
        mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variáveis para formas de maior massa
    max_area_red = 0
    max_contour_red = None
    max_area_blue = 0
    max_contour_blue = None

    # R1: Detectar todas as formas geométricas por cor
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 100:  # Filtro de ruído
            # Contorno vermelho
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
            if area > max_area_red:
                max_area_red = area
                max_contour_red = contour

    for contour in contours_blue:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(frame, [contour], -1,
                             (255, 0, 0), 2)  # Contorno azul
            if area > max_area_blue:
                max_area_blue = area
                max_contour_blue = contour

    # R2: Identificar a forma de maior massa com retângulo verde
    if max_contour_red is not None and max_area_red > max_area_blue:
        x, y, w, h = cv2.boundingRect(max_contour_red)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)  # Retângulo verde
    elif max_contour_blue is not None:
        x, y, w, h = cv2.boundingRect(max_contour_blue)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # R3 e R4: Detectar colisão e ultrapassagem
    if max_contour_red is not None and max_contour_blue is not None:
        rect_red = cv2.boundingRect(max_contour_red)
        rect_blue = cv2.boundingRect(max_contour_blue)

        # Verificar colisão
        collision = (rect_red[0] < rect_blue[0] + rect_blue[2] and
                     rect_red[0] + rect_red[2] > rect_blue[0] and
                     rect_red[1] < rect_blue[1] + rect_blue[3] and
                     rect_red[1] + rect_red[3] > rect_blue[1])

        if collision:
            cv2.putText(frame, "COLISAO DETECTADA", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # R4: Verificar ultrapassagem da forma de maior massa
        if max_area_red > max_area_blue:
            larger = rect_red
            smaller = rect_blue
        else:
            larger = rect_blue
            smaller = rect_red

        # Ultrapassagem: quando a forma maior passa completamente a menor (horizontalmente)
        if (larger[0] > smaller[0] + smaller[2]) or (larger[0] + larger[2] < smaller[0]):
            cv2.putText(frame, "ULTRAPASSOU", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir o frame processado
    cv2.imshow('Jogo de Colisão', frame)

    # Sair com 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
