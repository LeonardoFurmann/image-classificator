from flask import Flask, render_template, request

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM
import cv2 as cv


app = Flask(__name__)

def segment_image(img_path, eraseLineCenter=0, eraseLineWidth=15, threshold=70, target_size=(150, 150)):
    # Carregar a imagem
    img = cv.imread(img_path, 0)
    if img is None:
        return None

    # Redimensionar a imagem para o tamanho desejado
    img = cv.resize(img, target_size)

    # Apagar a faixa na imagem
    img_erased = eraseMax(img, eraseLineCenter, eraseLineWidth)

    img_erased = eraseCorners(img, 0.2)

    #     # Aplicar a operação de Black-Hat para realçar as características
    # ker = 1300
    # kernel = np.ones((ker, ker), np.uint8)
    # blackhat = cv.morphologyEx(img_erased, cv.MORPH_BLACKHAT, kernel)

    # # Aplicar um limiar para segmentar as características
    # ret, thresh = cv.threshold(blackhat, threshold, 255, 0)

    # # Obter a máscara de canto
    # cmask = get_cmask(img_erased)

    # # Multiplicar a máscara de canto pelo limiar
    # mask = np.multiply(cmask, thresh).astype('uint8')

    # # Aplicar um filtro de mediana
    # median = cv.medianBlur(mask, 25)

    # # Aplicar a máscara de contorno
    # contour_mask = contourMask(median).astype('uint8')

    # # Aplicar um filtro de mediana na máscara de contorno
    # contour_mask = cv.medianBlur(contour_mask, 23)

    # Aplicar a máscara de contorno à imagem original
    result = img_erased #cv.bitwise_and(img, contour_mask)

    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    model = load_model('./train_model/temp_result_2.h5')
    img_height, img_width = 150, 150

    imageFile = request.files['imagefile']
    image_path = "./images/" + imageFile.filename
    imageFile.save(image_path)

    img = image.load_img(image_path, target_size=(img_height, img_width))

    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  

    predictions = model.predict(img_array)  
    predicted_class = np.argmax(predictions[0])  
    accuracy = predictions[0][predicted_class] * 100

    class_labels = {0: 'Normal', 1: 'Pneumonia'}
    predicted_label = class_labels[predicted_class]

    # model.build(None)

    # output_layer = model.layers[6].name

    # print(model.layers[6].name)

    # # Explicação usando GradCAM
    # explainer = GradCAM()
    # grid = explainer.explain((img_array, None), model, class_index=predicted_class, layer_name=output_layer)


    # # Ajustar o contraste da imagem Grad-CAM
    # grid_contrast = exposure.adjust_gamma(grid, gamma=1.5)  # Invertendo o valor do gamma para ajustar o contraste

    # # Aplicar operações de morfologia e limiarização na imagem Grad-CAM
    # ker = 1300
    # kernel = np.ones((ker, ker), np.uint8)
    # blackhat = cv.morphologyEx(grid_contrast, cv.MORPH_BLACKHAT, kernel)
    # threshold = 75
    # ret, thresh = cv.threshold(blackhat, threshold, 255, cv.THRESH_BINARY)

    # # Aplicar uma máscara para manter apenas os pixels azuis
    # blue_mask = (thresh == 255)[:, :, 0]  # Verifica se o canal azul está presente (255) em cada pixel
    # blue_only = np.zeros_like(thresh, dtype=np.uint8)  # Especificar explicitamente o tipo de saída
    # blue_only[blue_mask] = 255  # Define como branco apenas os pixels que correspondem à máscara azul

    # # Definir o peso da sobreposição
    # alpha = 1  # Ajuste o valor conforme necessário

    # # Combinar as duas imagens usando a função addWeighted
    # overlay = cv.addWeighted(img_array[0].astype(np.uint8), 1 - alpha, blue_only, alpha, 0)

    # # Segmentar áreas com valores de pixel acima de 250
    # pneumonia_points = np.where(grid_contrast > 254)

    # # Encontrar os pontos onde a imagem com pneumonia está preta
    # black_points = np.where(overlay == 0)

    # # Criar uma cópia da imagem original para desenhar o círculo
    # img_with_circle = np.copy(img)

    # # Desenhar um círculo nos pontos onde a imagem com pneumonia é preta
    # for point in zip(*black_points):
    #     cv.circle(img_with_circle, (point[1], point[0]), radius=5, color=(0, 0, 255), thickness=-1)  # Desenha um círculo vermelho

    # # Definir a transparência da imagem com o círculo
    # alpha = 0.06  # Nível de transparência

    # # Converter a imagem original para numpy array
    # img_array_np = image.img_to_array(img)

    # # Combinar as duas imagens usando a função addWeighted
    # img_with_circle = cv.addWeighted(img_with_circle.astype(np.uint8), alpha, img_array_np.astype(np.uint8), 1 - alpha, 0)

    # # Ajustar o contraste e o brilho da imagem com o círculo adicionado
    # adjusted_img_with_circle = adjust_contrast_brightness_gamma(img_with_circle, alpha=1.5, beta=-30, gamma=1.2)  # Aumentar contraste, diminuir o brilho e ajustar o gama

    # adjusted_axs0_img = adjust_contrast_brightness_gamma(img_array[0], gamma=1.2)

    # # Ajustar o contraste, brilho e gama da imagem axs[2]
    # adjusted_axs2_img = adjust_contrast_brightness_gamma(grid_contrast, alpha=1.5, beta=-30, gamma=1.2)

    # Ajustar a cor da sobreposição com base na acurácia
    if predicted_label == 'Pneumonia':
        if accuracy > 90:
            overlay_color = 'red'  # Mantenha a cor vermelha para alta acurácia
            legend_label = 'Alta Probabilidade\n de pneumonia'
        elif accuracy > 80:
            overlay_color = 'orange'  # Cor laranja para acurácia moderada
            legend_label = 'Moderada Probabilidade\n de pneumonia'
        elif accuracy > 70:
            overlay_color = 'yellow'  # Cor amarela para acurácia mais baixa
            legend_label = 'Baixa Probabilidade\n de pneumonia'
        else:
            overlay_color = 'green'  # Cor verde para baixa acurácia
            legend_label = 'Muito Baixa Probabilidade\n de pneumonia'
    else:  # Normal
        if accuracy > 90:
            overlay_color = 'green'  # Mantenha a cor verde para alta acurácia
            legend_label = 'Alta Probabilidade\n de normal'
        elif accuracy > 80:
            overlay_color = 'yellow' # Cor amarela para acurácia moderada
            legend_label = 'Moderada Probabilidade\n de normal'
        elif accuracy > 70:
            overlay_color = 'orange'  # Cor laranja para acurácia mais baixa
            legend_label = 'Baixa Probabilidade\n de normal'
        else:
            overlay_color = 'red'  # Cor vermelha para baixa acurácia
            legend_label = 'Muito Baixa Probabilidade\n de normal'

        # img, img_with_circle, adjusted_img_with_circle, adjusted_axs2_img = 

    return render_template('index.html', prediction= predicted_label,legend_label = legend_label)

if __name__ == '__main__':
    app.run(port=3000, debug=True)