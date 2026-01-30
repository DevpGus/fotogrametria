import cv2
import csv
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_keypoint_density(aligned_images, scales, show=False):
    """
    Detecta Keypoints (AKAZE) em todas as camadas e analisa sua distribuição.
    Isso identifica quais imagens contêm a estrutura real da moeda.
    """
    if not aligned_images: return None
    
    os.system('cls')

    print(f"\nIniciando Análise de Densidade de Keypoints (AKAZE)")
    
    # Inicializa o detector AKAZE
    # threshold: Controla a sensibilidade. 
    akaze = cv2.AKAZE_create(threshold=0.001)
    
    keypoint_counts = []
    avg_responses = []
    
    # Loop de detecção
    print("\nProcessando imagens...")
    for i, img in enumerate(aligned_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kps = akaze.detect(gray, None)
        
        # Estatísticas
        count = len(kps)
        avg_resp = np.mean([kp.response for kp in kps]) if kps else 0
        
        keypoint_counts.append(count)
        avg_responses.append(avg_resp)
        
        if i % 10 == 0:
            print(f"Img {i}: {count} pontos detectados.")

    index = np.arange(len(aligned_images))
    counts = np.array(keypoint_counts)
    
    best_idx = np.argmax(counts)
    print(f"\nImagem com maior informação: Índice {best_idx} ({counts[best_idx]} pontos)")
    
    plt.figure(figsize=(15, 6))
    
    # Gráfico 1: Densidade de Keypoints (Quantidade)
    plt.subplot(1, 2, 1)
    plt.plot(index, counts, color='b', marker='o', linestyle='-', label='Qtd Keypoints')
    plt.axvline(best_idx, color='r', linestyle='--', label=f'Pico (Img {best_idx})')
    plt.title("Densidade de Informação (Foco Global)")
    plt.xlabel("Índice da Imagem (Z)")
    plt.ylabel("Número de Keypoints Detectados")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Relação com a Escala.
    plt.subplot(1, 2, 2)
    plt.plot(scales, counts, color='g', marker='x', linestyle='None')
    plt.title("Correlação: Escala vs. Quantidade de Detalhe")
    plt.xlabel("Fator de Escala (Zoom)")
    plt.ylabel("Número de Keypoints")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Gráfico 3: Keypoints da Imagem Vencedora.
    if show:
        best_img = aligned_images[best_idx].copy()
        kps_best = akaze.detect(cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY), None)
        img_kps = cv2.drawKeypoints(best_img, kps_best, None, color=(0,255,0), 
                                    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Visualização da Âncora (Img {best_idx})")
        plt.imshow(cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Retorna DataFrame para uso posterior
    df_kps = pd.DataFrame({
        'image_index': index,
        'keypoint_count': counts,
        'avg_response': avg_responses,
        'scale_val': scales if scales.all() else np.zeros_like(index)
    })
    
    return df_kps

def compute_focus_stacking(aligned_images, scales):
    """
    Realiza o Depth from Focus (DfF) para gerar a imagem nítida e mapas de dados.
    
    Parâmetros:
      aligned_images: Lista de imagens (numpy arrays) já alinhadas e recortadas.
      scales: Lista de floats com a escala de cada imagem.
      
    Retorna:
      Um dicionário com:
      - 'image': Imagem final (All-in-Focus).
      - 'depth_map': Mapa de índices (qual imagem venceu).
      - 'scale_map_pixelwise': Mapa com o valor de escala para cada pixel.
      - 'confidence_map': Mapa de nitidez (para filtrar ruído).
    """
    # Validação básica
    if not aligned_images or len(aligned_images) != len(scales):
        raise ValueError("O número de imagens e escalas deve ser igual.")

    os.system('cls')

    print(f"\nIniciando Depth from Focus em {len(aligned_images)} camadas")
    height, width = aligned_images[0].shape[:2]
    n_images = len(aligned_images)
    
    # 1. Alocação de Memória (Cubo de Nitidez)
    sharpness_cube = np.zeros((height, width, n_images), dtype=np.float32)
    
    # 2. Loop de Cálculo de Nitidez
    print("\nCalculando operador Laplaciano (ksize=5) com suavização espacial...")
    
    for i, img in enumerate(aligned_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)        
        score = np.absolute(lap)
        
        score_smooth = cv2.GaussianBlur(score, (13, 13), 0)
        sharpness_cube[:, :, i] = score_smooth

    # 3. Seleção de Camada (Winner-Takes-All)
    print("Selecionando camadas de máxima nitidez...")
    depth_index = np.argmax(sharpness_cube, axis=2)   # Mapa de Profundidade Bruto (Índices)
    confidence_map = np.max(sharpness_cube, axis=2)   # Mapa de Confiança (Valor da Derivada)

    # 4. Reconstrução da Imagem e Mapeamento de Escala
    print("Gerando imagem composta e mapa de escalas...")
    
    all_in_focus_img = np.zeros_like(aligned_images[0])    
    rows, cols = np.indices((height, width))
    
    stack_imgs = np.stack(aligned_images, axis=-1)     
    for c in range(3): 
        all_in_focus_img[:, :, c] = stack_imgs[:, :, c, :][rows, cols, depth_index]
    
    scale_map_pixelwise = np.array(scales)[depth_index]

    return {
        'image': all_in_focus_img,
        'depth_map': depth_index,
        'scale_map': scale_map_pixelwise,
        'confidence': confidence_map
    }

def calculate_depth_map(aligned_images, accmd_scales, motor_step):
    """
    Reconstrói o mapa de profundidade (Z) a partir de uma pilha de imagens,
    usando Depth from Focus combinado com a geometria de escala (Zoom).
    
    Args:
        aligned_images: Lista de arrays numpy (imagens em tons de cinza).
        accmd_scales: Lista de floats com a escala de cada imagem (S).
        motor_step: Deslocamento físico entre cada foto (Delta Z).
        
    Returns:
        depth_map: Matriz 2D com a distância Z calculada para cada pixel.
        index_map: Matriz 2D mostrando qual imagem tinha o melhor foco (para debug).
    """
    
    images = []
    for img in aligned_images:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img_gray)
        else:
            images.append(img)

    print("\nIniciando reconstrução 3D (Depth from Focus)")

    # 1. Empilhar imagens em um array 3D para processamento em lote (Altura, Largura, Numero_Imagens).
    stack = np.dstack(images)
    h, w, n_imgs = stack.shape
    
    print(f"Volume de Dados: {w}x{h} pixels com {n_imgs} camadas.")
    
    # 2. Calcular a Medida de Foco (Laplaciano) para todo o cubo
    focus_cube = np.zeros_like(stack, dtype=np.float32)
    
    print("\nCalculando nitidez (Laplaciano) por camada...")
    for i in range(n_imgs):
        img = stack[:, :, i]
        
        # Suavização leve.
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # Laplaciano: Detecta bordas rápidas (foco).
        lap = cv2.Laplacian(blur, cv2.CV_64F) 
        # Variância ou Magnitude absoluta do Laplaciano.
        focus_cube[:, :, i] = np.abs(lap)
        
    # 3. Encontrar o índice da imagem com maior nitidez para cada pixel.
    index_map = np.argmax(focus_cube, axis=2)
    
    # 4. Converter Índices para Profundidade Real (Z)
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    vector_scales = np.array(accmd_scales)
    vector_shift = np.arange(n_imgs) * motor_step
    
    for k in range(1, n_imgs):
        mask = (index_map == k)
        
        if np.sum(mask) == 0:
            continue
            
        s_k = vector_scales[k]
        dist_k = vector_shift[k]
        
        # Fórmula: Z = (s * DeltaZ) / (s - 1)
        if s_k > 1.0001: 
            z_value = (s_k * dist_k) / (s_k - 1.0)
            depth_map[mask] = z_value
        else:
            # Se a escala for 1 (sem zoom), a profundidade é indeterminada 
            # pela fórmula geométrica. Mantemos 0 ou definimos um valor padrão.
            pass

    print("Mapa de Profundidade calculado com sucesso.")
    return depth_map, index_map

def plot_depth(depth_map, index_map):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(index_map, cmap='nipy_spectral')
    plt.colorbar(label='Índice da Imagem (0 a N)')
    plt.title("Mapa de Índices")
    
    plt.subplot(1, 2, 2)
    # Mascarar zeros para visualização melhor (fundo preto)
    depth_masked = np.ma.masked_where(depth_map == 0, depth_map)
    plt.imshow(depth_masked, cmap='plasma_r') # plasma_r: cores quentes = perto
    plt.colorbar(label='Distância Z estimada (mm)')
    plt.title("Mapa de Profundidade Calculado (Z)")
    
    plt.tight_layout()
    plt.show()

def save_focus_results(results, output_dir="../results/depth_map"):
    """
    Exibe o painel de diagnóstico e salva os dados.
    
    """
    if results is None: return

    # Cria diretório
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nDiretório '{output_dir}' verificado.")

    # Extrai os mapas (Matrizes 2D)
    img_aif = results['image']           # BGR
    depth_map = results['depth_map']     # Índices Z
    scale_map = results['scale_map']     # Escalas
    conf_map = results['confidence']     # Laplaciano

    # Dashboard
    print("Gerando painel visual...")
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1); plt.title("All-in-Focus")
    plt.imshow(cv2.cvtColor(img_aif, cv2.COLOR_BGR2RGB)); plt.axis('off')
    
    plt.subplot(1, 4, 2); plt.title("Profundidade (Índice)")
    plt.imshow(depth_map, cmap='turbo'); plt.colorbar(); plt.axis('off')
    
    plt.subplot(1, 4, 3); plt.title("Escala (Zoom)")
    plt.imshow(scale_map, cmap='plasma'); plt.colorbar(); plt.axis('off')
    
    plt.subplot(1, 4, 4); plt.title("Confiança (Nitidez)")
    plt.imshow(conf_map, cmap='gray'); plt.colorbar(); plt.axis('off')
    
    plt.tight_layout()
    plt.show()
