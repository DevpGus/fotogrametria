from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2


#### Cost Functions
def mse(img1, img2):
    """
    Mean Squared Error.
    Retorna o erro médio do quadrado da unidade de intensidade dos pixels.
    
    """
    diff = img1.astype("float") - img2.astype("float")
    mse = np.mean(diff ** 2)
    return mse

def rmse(img1, img2):
    """
    Root Mean Squared Error.
    Retorna o erro médio na mesma unidade de intensidade dos pixels.

    """
    diff = img1.astype("float") - img2.astype("float")
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    return rmse

def ncc(img1, img2):
    """
    Normalized Cross-Correlation.
    Retorna um valor entre -1 e 1 (cosseno da similaridade).

    """
    # Transforma as imagens em vetores 1D.
    v1 = img1.flatten().astype("float")
    v2 = img2.flatten().astype("float")
    dot_product = np.dot(v1, v2)
    
    # Normas (magnitudes) dos vetores.
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

### Gerenciamento dos Gráficos
def on_press(event):
    global debug
    if event.key == 'enter':
        plt.close()
    elif event.key == '0':
        debug = False
        plt.close()

#### Estimate Functions
def estimate(img_base, img_zoom, interval, metric, debug=False):
    """
    Testa vários fatores de escala para encontrar qual faz a img_zoom voltar a ser idêntica à img_base.

    Args:
        img_base: Imagem base (i).
        img_zoom: Imagem zoom (i+1).
        interval: Intervalo de busca da Função de Custo. 
        metric: Define a Função de Custo ('MSE', 'RMSE' ou 'NCC').
        debug: Exibe um gráfico para avaliar a seleção da escala.
    
    Output:
        s: Melhor escala encontrada.
        err: Menor erro encontrado.

    """
    h, w = img_base.shape
    center = (w // 2, h // 2)
    
    errors = []
    scales = []
        
    init = time.time()
    
    # Loop de Otimização.
    margin = int(min(h, w) * 0.1)
    for s_test in interval:
        
        M = cv2.getRotationMatrix2D(center=center, angle=0, scale=s_test)
        img_base_transformed = cv2.warpAffine(img_base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Mascaramento.
        roi_base = img_base_transformed[margin:-margin, margin:-margin]
        roi_zoom = img_zoom[margin:-margin, margin:-margin]
        
        if metric == 'MSE':
            err = mse(roi_zoom, roi_base)
        elif metric == 'RMSE':
            err = rmse(roi_zoom, roi_base)
        elif metric == 'NCC':
            err = ncc(roi_zoom, roi_base)
        
        errors.append(err)
        scales.append(s_test)
        
    final = time.time()
    
    # MSE ou RMSE (Minimização de Erro)
    if metric in ['MSE', 'RMSE']:
        idx = np.argmin(errors) 

    # NCC (Maximização de Similaridade)
    elif metric == 'NCC':
        idx = np.argmax(errors)

    s = scales[idx]
    err = errors[idx]
    
    if debug:
        # Plotagem com Matplotlib.
        fig = plt.figure(figsize=(12, 5))
        plt.suptitle(f"Função de Custo: {metric} | Pressione '0' ou 'ESC' para continuar")

        # Gráfico 1: A Função de Custo.
        plt.plot(scales, errors, color='blue', linewidth=2)
        plt.axvline(x=s, color='red', linestyle='--', label=f'Detectado: {s:.4f}')
        
        plt.title("Função de Custo (Erro MSE vs Escala)")
        plt.xlabel("Fator de Escala Testado")
        plt.ylabel("Erro Médio Quadrático (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig.canvas.mpl_connect('key_press_event', on_press)

        plt.show()

    print(f"Otimização concluída em {final - init:.4f}s")
    print(f"Melhor Escala Encontrada: {s:.4f} (Erro: {err:.2f})")

    return s, err

def algorithm(images, metric, interval, debug=False):
    """
    Processa uma sequência de imagens alinhadas para determinar a escala relativa passo a passo.
    
    Args:
        images: Lista de numpy arrays (imagens em escala de cinza).
        metric: Define a Função de Custo ('MSE', 'RMSE' ou 'NCC').
        interval: Intervalo de iteração do método.
        debug: Exibe um gráfico para avaliar a seleção da escala.

    Returns:
        scales_step: Lista de fatores de escala entre i e i+1
        accumulated_scales: Lista da escala absoluta em relação à primeira imagem

    """

    print("\n[Algoritmo]")
    print(f"Função de Custo selecionada: {metric}")
    print(f"Total de imagens na pilha: {len(images)}")
    print(f"Iniciando iteração no intervalo I = [{interval[0]:.4f}, {interval[-1]:.4f}]")
    
    scales_step = []
    errors_min = []

    # Iterar sobre os pares.
    for i in range(len(images)-1):
        img_base = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_next = cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY)

        print("-------------------------------------------------------")
        print(f"[Processando par {i} -> {i+1}]...", end=" ")
        
        scale, error = estimate(
            img_base, 
            img_next, 
            interval,
            metric=metric,
            debug=debug
        )
        scales_step.append(scale)
        errors_min.append(error)
        
    # Calcular Escala Acumulada (Trajetória Z).
    accumulated_scale = [1.0] # A primeira imagem é a base 1.0

    for s in scales_step:
        nova_escala_total = accumulated_scale[-1] * s
        accumulated_scale.append(nova_escala_total)

    # Gráficos.
    if debug:

        fig = plt.figure(figsize=(12, 5))
        plt.suptitle(f"Função de Custo: {metric} | Pressione '0' ou 'ESC' para continuar")

        # Gráfico 1: Escala por Passo
        plt.subplot(1, 2, 1)
        plt.plot(range(len(scales_step)), scales_step, marker='o', linestyle='-')
        plt.title("Fator de Escala (s) por Passo (i)")
        plt.xlabel("Índice da Imagem (i)")
        plt.ylabel("Fator de Escala (s)")
        plt.grid(True)
        
        # Gráfico 2: Escala Acumulada
        plt.subplot(1, 2, 2)
        plt.plot(range(len(accumulated_scale)), accumulated_scale, marker='s', color='orange', linestyle='-')
        plt.title("Trajetória da Escala Acumulada")
        plt.xlabel("Índice da Imagem (i)")
        plt.ylabel("Escala Total Relativa à Imagem 0")
        plt.grid(True)

        plt.tight_layout()

        plt.show()

    scales_step.insert(0, 1)

    return scales_step, accumulated_scale